# Pdf--summarizer--using--rag-pipeline
# RAG-based PDF Summarizer

A Retrieval-Augmented Generation (RAG) application that summarizes and answers questions about PDF documents. Upload one or more PDFs, and the app chunks, embeds, and indexes the content, then uses a large language model (LLM) to generate accurate, context-grounded summaries and answers.

## Features

- 📄 Upload and process single or multiple PDF documents
- 🔍 Semantic search over document content using vector embeddings
- 🧠 Context-aware summarization powered by an LLM
- 💬 Ask follow-up questions about the document (chat-style Q&A)
- ✂️ Automatic text chunking with configurable chunk size/overlap
- 📊 Source citation — see which page/section an answer came from
- ⚡ Caching of embeddings to avoid reprocessing the same document

## Architecture

```
                ┌─────────────┐
                │   PDF File  │
                └──────┬──────┘
                       │
                 Text Extraction
                       │
                       ▼
              ┌─────────────────┐
              │  Chunking &     │
              │  Preprocessing  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Embedding     │
              │     Model       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Vector Store  │◄────── Retrieval (top-k chunks)
              │  (e.g. FAISS/   │
              │  Chroma/Pinecone)│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   LLM (RAG)     │──────► Summary / Answer
              │  Prompt + Context│
              └─────────────────┘
```

## Tech Stack

| Component        | Technology (example)                 |
|-------------------|---------------------------------------|
| Language           | Python 3.10+                          |
| PDF Parsing        | PyPDF2 / pdfplumber / PyMuPDF         |
| Embeddings         | OpenAI / Sentence-Transformers        |
| Vector Store       | FAISS / ChromaDB / Pinecone           |
| LLM                | OpenAI GPT / Anthropic Claude / local model |
| Orchestration      | LangChain / LlamaIndex                |
| Frontend (optional)| Streamlit / Gradio / FastAPI + React  |

> Replace the above with your actual stack.

## Prerequisites

- Python 3.10 or higher
- pip / conda
- API key for your chosen LLM provider (e.g., `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- (Optional) Docker, if running via container

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/rag-pdf-summarizer.git
   cd rag-pdf-summarizer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_api_key_here
   VECTOR_DB_PATH=./data/vector_store
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

## Usage

### Run locally

```bash
python app.py
```

### Run with Streamlit UI

```bash
streamlit run app.py
```

### Run via CLI

```bash
python summarize.py --file path/to/document.pdf --mode summary
python summarize.py --file path/to/document.pdf --mode qa --question "What is the main conclusion?"
```

### Run with Docker

```bash
docker build -t rag-pdf-summarizer .
docker run -p 8501:8501 --env-file .env rag-pdf-summarizer
```

## Project Structure

```
rag-pdf-summarizer/
├── app.py                 # Main application entry point
├── summarize.py            # CLI script for summarization
├── src/
│   ├── loader.py           # PDF loading and text extraction
│   ├── chunker.py          # Text chunking logic
│   ├── embeddings.py       # Embedding generation
│   ├── vector_store.py     # Vector DB indexing and retrieval
│   ├── rag_pipeline.py     # RAG orchestration (retrieve + generate)
│   └── prompts.py          # Prompt templates
├── data/
│   └── vector_store/       # Persisted vector index
├── tests/                  # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

| Variable         | Description                            | Default |
|-------------------|-----------------------------------------|---------|
| `CHUNK_SIZE`       | Number of characters per text chunk     | 1000    |
| `CHUNK_OVERLAP`    | Overlap between consecutive chunks      | 200     |
| `TOP_K`            | Number of chunks retrieved per query    | 4       |
| `EMBEDDING_MODEL`  | Embedding model name                    | text-embedding-3-small |
| `LLM_MODEL`        | LLM used for generation                 | gpt-4o-mini |

## How It Works

1. **Extraction** – Text is extracted from the uploaded PDF, preserving page numbers for citation.
2. **Chunking** – Extracted text is split into overlapping chunks to preserve context across boundaries.
3. **Embedding** – Each chunk is converted into a vector embedding.
4. **Indexing** – Embeddings are stored in a vector database for fast similarity search.
5. **Retrieval** – When a summary or question is requested, the most relevant chunks are retrieved.
6. **Generation** – Retrieved chunks are passed to the LLM as context to generate a grounded summary/answer.

## Roadmap

- [ ] Multi-document cross-referencing
- [ ] Support for scanned PDFs via OCR
- [ ] Export summaries to Markdown/PDF
- [ ] User authentication and document history
- [ ] Streaming responses in UI

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a PR

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [LangChain](https://www.langchain.com/) / [LlamaIndex](https://www.llamaindex.ai/) for RAG orchestration
- [OpenAI](https://openai.com/) / [Anthropic](https://www.anthropic.com/) for LLM APIs
- [FAISS](https://github.com/facebookresearch/faiss) / [ChromaDB](https://www.trychroma.com/) for vector search
