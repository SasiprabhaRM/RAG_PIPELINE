# RAG_PIPELINE

A modular, production-grade Retrieval-Augmented Generation (RAG) pipeline built with LangChain, ChromaDB, and BERT embeddings. Supports multiple document formats and follows clean OOP design with PEP8 standards, logging, and pre-commit hooks.

# Features

Multi-format ingestion — PDF, CSV, Excel, HTML, and web URLs
Smart chunking — Markdown-aware chunking for PDFs, semantic BERT-based chunking for all other formats
Local embeddings — BERT (all-MiniLM-L6-v2) via langchain-huggingface, no API key required
OpenAI embeddings — Optional text-embedding-3-small support
ChromaDB vector store — Persistent local storage with similarity search
Similarity retrieval — Top-K document retrieval based on cosine similarity
Fully configurable — All settings driven by config/config.yaml
Pre-commit hooks — Ruff, Mypy, and file hygiene checks enforced on every commit

## Setup
1. Clone the repository
bashgit clone https://github.com/YOUR_USERNAME/RAG_PIPELINE.git
cd parsingtask
2. Create and activate virtual environment
bash python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

3. Install dependencies
bashpip install -r requirements.txt
pip install langchain-huggingface   # replaces deprecated HuggingFaceEmbeddings
pip install bandit[toml]            # for pre-commit bandit hook

4. Set up environment variables
Create a .env file in the project root:
env OPENAI_API_KEY=sk-your-key-here    


5. Configure config/config.yaml
yamlembedding:
  provider: bert                    # use 'openai' if you have API credits
  bert_model: all-MiniLM-L6-v2

## Usage
Run the pipeline
# PDF
python main.py --source data/samplepdf.pdf --query "What are the key findings?"

# CSV
python main.py --source data/employees.csv --query "Who works in Data Science?"

# Excel
python main.py --source data/data.xlsx --query "What are the totals?"

# HTML file
python main.py --source data/samplehtml.html --query "What is the main topic?"

# Web URL
python main.py --source https://en.wikipedia.org/wiki/Python_(programming_language) --query "Who created Python?
