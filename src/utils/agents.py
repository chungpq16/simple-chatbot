import os
from typing import Annotated, Dict, List, Tuple, TypedDict, Optional
import logging
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import langgraph
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define state types
class AgentState(TypedDict):
    """Represents the state of our agentic workflow"""
    question: str
    original_question: str
    rephrased_question: Optional[str]
    contexts: List[str]
    answer: Optional[str]
    grade: Optional[float]
    iteration: int
    max_iterations: int

class AgentAction(str, Enum):
    REPHRASE = "rephrase"  
    RETRIEVE = "retrieve" 
    RESPOND = "respond"
    EVALUATE = "evaluate"
    FINISH = "finish"

class AgentGraph:
    def __init__(
        self, 
        vectorstore: Chroma,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_iterations: int = 2,
        retrieval_k: int = 6,
        relevance_threshold: float = 0.7,
        use_azure: bool = False
    ):
        self.vectorstore = vectorstore
        
        # Initialize LLM - Force Azure OpenAI if use_azure=True
        if use_azure or (
            os.getenv("AZURE_OPENAI_API_KEY") and 
            os.getenv("AZURE_OPENAI_ENDPOINT") and 
            os.getenv("AZURE_OPENAI_API_VERSION") and
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        ):
            # Use Azure OpenAI
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", model_name)
            self.llm = AzureChatOpenAI(
                azure_deployment=deployment_name,
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                temperature=temperature
            )
            logger.info(f"Using Azure OpenAI for LLM with deployment {deployment_name}")
        else:
            # Only reach here if use_azure is False and Azure environment variables are missing
            if use_azure:
                error_msg = "Azure OpenAI was requested but environment variables are missing"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Fallback to regular OpenAI
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
            logger.info(f"Using OpenAI for LLM with model {model_name}")
            
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k}
        )
        self.max_iterations = max_iterations
        self.relevance_threshold = relevance_threshold
        self.graph = self._build_graph()
    
    def _query_rewriter(self, state: AgentState) -> AgentState:
        """Agent A: Rewrites and improves the user query for better retrieval"""
        
        # Skip rewriting on first iteration to try the original query first
        if state["iteration"] == 0:
            logger.info("First iteration, using original query")
            state["rephrased_question"] = state["question"]
            return state
        
        logger.info(f"Rewriting query, iteration: {state['iteration']}")
        
        # Prompt for query rewriting
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query reformulation assistant. 
Your task is to rewrite the user's query to make it more effective for retrieving relevant information.
Consider the original question, prior retrieved context (if any), and the current answer quality.
Make the query more specific, include any technical terms that might help with retrieval.
Do not try to answer the question, just rewrite it for better retrieval results.
Only output the rewritten query text. No explanations or other text."""),
            ("human", "Original question: {question}"),
            ("human", "Current retrieved context: {contexts}"),
            ("human", "Current answer: {answer}"),
            ("human", "Current grade: {grade}")
        ])
        
        # Generate the improved query
        chain = prompt | self.llm | StrOutputParser()
        improved_query = chain.invoke(state)
        
        # Update state
        state["rephrased_question"] = improved_query
        
        logger.info(f"Original query: {state['question']}")
        logger.info(f"Reformulated query: {improved_query}")
        
        return state
    
    def _retrieval(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents using the current query"""
        query = state.get("rephrased_question", state["question"])
        logger.info(f"Retrieving documents for query: {query}")
        
        # Perform retrieval
        docs = self.retriever.invoke(query)
        contexts = [doc.page_content for doc in docs]
        
        # Store contexts in state
        state["contexts"] = contexts
        logger.info(f"Retrieved {len(contexts)} document chunks")
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate a response using retrieved documents"""
        query = state.get("rephrased_question", state["question"])
        original_query = state["original_question"]
        contexts = state["contexts"]
        
        logger.info(f"Generating response for: {query}")
        
        # Prepare context for the model
        context_str = "\n\n".join(contexts)
        
        # Prompt for answering
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that provides accurate answers based on the given context.
If the context doesn't contain the information needed, admit that you don't know rather than making up an answer.
Base your response only on the provided context.
Be precise, clear and helpful."""),
            ("human", """Original question: {original_question}
Current question: {query}

Context:
{context}

Please provide a comprehensive answer to the question based solely on the context.""")
        ])
        
        # Generate the response
        chain = response_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "original_question": original_query,
            "query": query,
            "context": context_str
        })
        
        # Update state
        state["answer"] = answer
        logger.info("Generated response")
        
        return state
    
    def _evaluate_response(self, state: AgentState) -> AgentState:
        """Agent B: Evaluate the relevance and quality of the generated response"""
        original_query = state["original_question"]
        answer = state["answer"]
        
        logger.info("Evaluating response quality")
        
        # Prompt for evaluation
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator of answer quality.
Your task is to grade how well the provided answer addresses the original question.
Consider accuracy, completeness, and relevance to the question.

Grade on a scale from 0.0 to 1.0 where:
- 0.0: Completely irrelevant or incorrect answer
- 0.5: Partially addresses the question but has significant gaps or inaccuracies
- 0.7: Good answer that addresses most aspects of the question
- 1.0: Perfect answer that fully and accurately addresses the question

Return only a single number representing your grade."""),
            ("human", f"""Original question: {original_query}
Answer to evaluate: {answer}

Grade (0.0 to 1.0):""")
        ])
        
        # Generate the evaluation score
        chain = eval_prompt | self.llm | StrOutputParser()
        
        try:
            score = float(chain.invoke({}))
            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
        except:
            # Default score if parsing fails
            score = 0.5
        
        # Update state
        state["grade"] = score
        state["iteration"] += 1
        
        logger.info(f"Response quality score: {score}, Iteration: {state['iteration']}")
        
        return state
    
    def _should_rephrase(self, state: AgentState) -> str:
        """Decision node to determine next action based on quality evaluation"""
        grade = state.get("grade", 0.0)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)
        
        # If we've hit max iterations or have a good enough answer, finish
        if iteration >= max_iterations:
            logger.info(f"Reached max iterations ({max_iterations}), finishing")
            return AgentAction.FINISH
        
        # If quality is below threshold and we haven't exceeded iterations, rephrase
        if grade < self.relevance_threshold:
            logger.info(f"Grade {grade} below threshold {self.relevance_threshold}, rephrasing")
            return AgentAction.REPHRASE
        else:
            logger.info(f"Grade {grade} is good enough, finishing")
            return AgentAction.FINISH
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for agent workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes to the graph
        workflow.add_node(AgentAction.REPHRASE, self._query_rewriter)
        workflow.add_node(AgentAction.RETRIEVE, self._retrieval)
        workflow.add_node(AgentAction.RESPOND, self._generate_response)
        workflow.add_node(AgentAction.EVALUATE, self._evaluate_response)
        
        # Build the flow
        workflow.set_entry_point(AgentAction.REPHRASE)
        workflow.add_edge(AgentAction.REPHRASE, AgentAction.RETRIEVE)
        workflow.add_edge(AgentAction.RETRIEVE, AgentAction.RESPOND)
        workflow.add_edge(AgentAction.RESPOND, AgentAction.EVALUATE)
        
        # Conditional edges
        workflow.add_conditional_edges(
            AgentAction.EVALUATE,
            self._should_rephrase,
            {
                AgentAction.REPHRASE: AgentAction.REPHRASE,
                AgentAction.FINISH: END
            }
        )
        
        return workflow.compile()
    
    def run(self, question: str) -> Dict:
        """Run the agent workflow for a given user question"""
        logger.info(f"Processing question: {question}")
        
        # Initialize state
        initial_state = {
            "question": question,
            "original_question": question,
            "rephrased_question": None,
            "contexts": [],
            "answer": None,
            "grade": None,
            "iteration": 0,
            "max_iterations": self.max_iterations
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "question": question,
            "answer": result["answer"],
            "quality_grade": result["grade"],
            "final_query": result.get("rephrased_question", question),
            "iterations": result["iteration"]
        }