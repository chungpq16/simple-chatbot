# Agentic RAG Chatbot

A smart assistant built with LangGraph that can ingest and understand local PDF documents, using an agentic workflow to optimize query handling and response quality.

## Features

- **Document Processing**: Automatic PDF ingestion with hyperlink extraction and incremental updates
- **Agentic Architecture**: Uses LangGraph to coordinate multiple agents that collaborate
- **Query Optimization**: Query rewriter agent improves search results
- **Quality Control**: Answer evaluator agent grades responses and triggers refinement when needed
- **User Interface**: Simple Streamlit chat interface with detailed processing information

## Architecture

The system uses a graph-based agentic workflow:

1. **Document Processing**: PDF files are processed, text is extracted (including hyperlinks), and embeddings are generated using Ollama's nomic-embed-text model
2. **Vector Storage**: Documents are stored in a local ChromaDB vector database for efficient retrieval
3. **Agentic Flow**: 
   - User query → Query Rewriter Agent (Agent A) → Retrieval → Response Generation
   - Response → Quality Evaluator Agent (Agent B)
   - If quality is below threshold: return to Agent A for query refinement
   - If quality is acceptable or max iterations reached: return final answer

## Project Structure

```
agentic-rag-chatbot/
├── docs/                # Put your PDF documents here
├── vectordb/           # Vector database storage (auto-created)
├── src/
│   ├── app.py          # Streamlit application
│   ├── utils/
│       ├── document_processor.py   # PDF processing and embedding
│       ├── agents.py    # LangGraph agent definitions
├── requirements.txt    # Python dependencies
├── .env.example        # Configuration template
└── README.md           # This file
```

## Setup Instructions

1. **Clone the repository**:
```bash
git clone <repository-url>
cd agentic-rag-chatbot
```

2. **Set up a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
```
Then edit `.env` to add your OpenAI API key.

5. **Add PDF documents**:
Place your PDF files in the `docs/` directory.

6. **Run the application**:
```bash
streamlit run src/app.py
```

7. **Access the interface**:
Open your browser and go to the URL displayed in the terminal (usually http://localhost:8501).

## Usage

1. Add or update PDF files in the `docs/` folder
2. The system automatically detects new or changed files
3. Ask questions in the chat interface
4. View detailed processing information by expanding the "View processing details" section under each response

## Requirements

- Python 3.9+
- Ollama with the `nomic-embed-text:latest` model installed
- OpenAI API key
- Sufficient disk space for storing document embeddings

## Configuration Options

Edit the `.env` file to configure:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL_NAME`: Model to use (default: "gpt-3.5-turbo")
- `MAX_ITERATIONS`: Maximum query refinement attempts (default: 2)
- `RELEVANCE_THRESHOLD`: Quality threshold for responses (0.0-1.0, default: 0.7)

## Advanced Usage

### Forcing Reindex

If you need to force the system to reindex all documents:

1. Click the "Force Reindex Documents" button in the sidebar
2. Wait for the reindexing process to complete

### Customizing Embeddings

The system uses Ollama's nomic-embed-text model by default. To use a different model:

1. Edit the `embedding_model` parameter in `src/app.py`
2. Restart the application