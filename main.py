"""
RAG Pipeline with LangGraph — Entry Point.

Usage:
    python main.py --source <file_or_url> --query "your question"
    python main.py --source data/report.pdf --query "What are the key findings?"
    python main.py --source https://example.com --query "Summarise this page"
"""

import argparse
import io
import logging
import os
import sys
from typing import cast

import chromadb
import chromadb.config
import yaml
from dotenv import load_dotenv

from src.chunker import ChunkerFactory
from src.embedder import ChromaDBIndexer, EmbedderFactory
from src.llm import GroqLLM
from src.loader import LoaderFactory
from src.reranker import CrossEncoderReranker
from src.retriever import SimilarityRetriever
from src.router import RAGGraphBuilder

cast(io.TextIOWrapper, sys.stdin).reconfigure(encoding="utf-8")
cast(io.TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8")
# Suppress USER_AGENT warning from WebBaseLoader
os.environ.setdefault("USER_AGENT", "rag-pipeline/1.0")


def setup_logging(config: dict) -> None:
    """Configure root logger from config.yaml settings."""
    log_cfg = config.get("logging", {})
    log_file = log_cfg.get("log_file", "logs/app.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get(
            "format",
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        ),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(path: str = "config/config.yaml") -> dict:
    """
    Parse and return the YAML configuration file.

    Args:
        path: Path to the config YAML file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError:        If the config file is empty or invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: '{path}'")
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if config is None:
        raise ValueError(f"Config file is empty or invalid YAML: '{path}'")
    return config


def detect_format(source: str) -> str:
    """
    Derive a short format label from the source path or URL.

    Args:
        source: File path or URL string.

    Returns:
        Format label e.g. 'pdf', 'csv', 'xlsx', 'html', 'url'.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return "url"
    return os.path.splitext(source)[-1].lower().lstrip(".")


def ingest_source(source: str, config: dict, openai_key: str) -> ChromaDBIndexer:
    """
    Load, chunk, embed and index the source into ChromaDB.

    Args:
        source:     File path or URL to ingest.
        config:     Parsed configuration dictionary.
        openai_key: OpenAI API key (empty string if using BERT).

    Returns:
        A ChromaDBIndexer instance with indexed documents.
    """
    logger = logging.getLogger(__name__)
    embed_cfg = config["embedding"]

    # ── 1. LOAD ───────────────────────────────────────────────────────────
    print(f"\n[STEP 1] Loading source: {source}")
    logger.info("Pipeline started for source: %s", source)

    loader = LoaderFactory.get_loader(source)
    documents = loader.load()
    print(f"         ✓ Loaded {len(documents)} document(s)")

    if not documents:
        print("[ERROR] No documents were loaded. Check your file path.")
        logger.error("No documents loaded from: %s", source)
        sys.exit(1)

    print(f"         Preview: {documents[0].page_content[:200]!r}")

    # ── 2. CHUNK ──────────────────────────────────────────────────────────
    print("\n[STEP 2] Chunking documents…")
    source_format = detect_format(source)
    chunker_cfg = config["chunker"]

    chunker = ChunkerFactory.get_chunker(
        source_format=source_format,
        chunk_size=chunker_cfg["markdown_chunk_size"],
        chunk_overlap=chunker_cfg["markdown_chunk_overlap"],
        openai_api_key=openai_key,
        bert_model=embed_cfg["bert_model"],
        semantic_threshold=chunker_cfg["semantic_breakpoint_threshold"],
    )
    chunks = chunker.split(documents)
    print(f"         ✓ Created {len(chunks)} chunk(s)")

    if not chunks:
        print("[ERROR] Chunking produced 0 chunks. The document may be empty.")
        logger.error("Chunking produced 0 chunks for source: %s", source)
        sys.exit(1)

    print(f"         First chunk preview: {chunks[0].page_content[:200]!r}")

    # ── 3. EMBED + INDEX ──────────────────────────────────────────────────
    print("\n[STEP 3] Embedding & indexing into ChromaDB…")
    provider = embed_cfg["provider"]
    logger.info("Embedding provider: %s", provider)

    embeddings = EmbedderFactory.get_embeddings(
        provider=provider,
        openai_api_key=openai_key,
        bert_model=embed_cfg["bert_model"],
    )

    db_cfg = config["chromadb"]

    # Single shared client — prevents duplicate instance errors
    chroma_client = chromadb.PersistentClient(
        path=db_cfg["persist_directory"],
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )

    existing_names = [c.name for c in chroma_client.list_collections()]
    if db_cfg["collection_name"] in existing_names:
        print(f"         Deleting old collection '{db_cfg['collection_name']}'…")
        logger.info("Deleting existing collection: %s", db_cfg["collection_name"])
        chroma_client.delete_collection(db_cfg["collection_name"])

    indexer = ChromaDBIndexer(
        client=chroma_client,
        collection_name=db_cfg["collection_name"],
        embedding_function=embeddings,
    )
    indexer.index(chunks)
    print(f"         ✓ Indexed {len(chunks)} chunk(s) into ChromaDB")

    collection = chroma_client.get_collection(db_cfg["collection_name"])
    print(f"         ✓ ChromaDB collection count: {collection.count()}")

    return indexer


def build_graph(config: dict, indexer: ChromaDBIndexer, groq_key: str):
    """
    Assemble and compile the LangGraph RAG graph.

    Args:
        config:   Parsed configuration dictionary.
        indexer:  ChromaDBIndexer with indexed documents.
        groq_key: Groq API key for LLM calls.

    Returns:
        Compiled LangGraph app.
    """
    retriever_cfg = config["retriever"]
    reranker_cfg = config["reranker"]
    llm_cfg = config["llm"]
    router_cfg = config["router"]

    retriever = SimilarityRetriever(
        vectorstore=indexer.get_vectorstore(),
        top_k=retriever_cfg["top_k"],
    )
    reranker = CrossEncoderReranker(
        model_name=reranker_cfg["model"],
        top_k=reranker_cfg["top_k"],
    )
    llm = GroqLLM(
        model=llm_cfg["model"],
        api_key=groq_key,
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
    )

    builder = RAGGraphBuilder(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        router_prompt_template=router_cfg["prompt"],
    )
    return builder.build()


def run_pipeline(source: str, query: str, config: dict) -> None:
    """
    Run the full RAG pipeline via LangGraph.

    Args:
        source: File path or URL to ingest.
        query:  Natural language question.
        config: Parsed configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    load_dotenv()

    openai_key: str = os.getenv("OPENAI_API_KEY") or ""
    groq_key: str = os.getenv("GROQ_API_KEY") or ""

    if not groq_key:
        print("[ERROR] GROQ_API_KEY is not set in your .env file!")
        logger.error("GROQ_API_KEY missing.")
        sys.exit(1)

    if config["embedding"]["provider"] == "openai" and not openai_key:
        print("[ERROR] OPENAI_API_KEY is not set in your .env file!")
        logger.error("OPENAI_API_KEY missing.")
        sys.exit(1)

    # Ingest source into ChromaDB
    indexer = ingest_source(source, config, openai_key)

    # Build LangGraph
    print("\n[STEP 4] Building LangGraph RAG graph…")
    graph = build_graph(config, indexer, groq_key)
    print("         ✓ Graph compiled")

    # Invoke graph
    print(f"\n[STEP 5] Running graph for query: '{query}'")
    initial_state = {
        "query": query,
        "route": "",
        "documents": [],
        "reranked": [],
        "answer": "",
    }
    result = graph.invoke(initial_state)

    # Output
    route = result.get("route", "UNKNOWN")
    answer = result.get("answer", "No answer generated.")
    reranked = result.get("reranked", [])

    print("\n" + "═" * 60)
    print(f"  Query  : {query}")
    print(f"  Route  : {route}")
    print("═" * 60)

    if route == "DOCUMENT" and reranked:
        print(f"\n  📄 Top {len(reranked)} reranked chunk(s) used as context:\n")
        for i, doc in enumerate(reranked, 1):
            print(f"  [{i}] Source: {doc.metadata.get('source', 'N/A')}")
            print(f"       {doc.page_content[:200]}")
            print()

    print("  💬 Answer:\n")
    print(f"  {answer}")
    print("\n" + "═" * 60)
    logger.info("Pipeline complete — route=%s, answer=%d chars.", route, len(answer))


def main() -> None:
    """Parse CLI arguments and launch the RAG pipeline."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline with LangGraph — Load, Chunk, Index, Rerank, Answer"
    )
    parser.add_argument("--source", required=True, help="File path or URL to ingest")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    run_pipeline(args.source, args.query, config)


if __name__ == "__main__":
    main()
