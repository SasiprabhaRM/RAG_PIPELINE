"""
Embedding & ChromaDB Indexing Module.

Supports OpenAI embeddings and local BERT (sentence-transformers).
Stores and retrieves from a persistent ChromaDB collection.
"""

import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class EmbedderFactory:
    """Factory that returns the configured embedding model."""

    @staticmethod
    def get_embeddings(
        provider: str,
        openai_api_key: str = "",
        bert_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Return a LangChain-compatible embedding model instance.

        Args:
            provider:       Embedding backend — 'openai' or 'bert'.
            openai_api_key: API key (required when provider is 'openai').
            bert_model:     HuggingFace model name (used when provider is 'bert').

        Returns:
            A LangChain embeddings instance.

        Raises:
            ValueError: If an unknown provider string is supplied.
        """
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings(api_key=openai_api_key)

        if provider == "bert":
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info("Using BERT embeddings: %s", bert_model)
            return HuggingFaceEmbeddings(model_name=bert_model)

        raise ValueError(f"Unknown embedding provider: '{provider}'")


class ChromaDBIndexer:
    """
    Manages a ChromaDB collection for document indexing and retrieval.

    Accepts an externally created PersistentClient to avoid duplicate
    instance errors when the same persist path is used across the app.

    Args:
        client:             An existing chromadb.PersistentClient instance.
        collection_name:    Name of the ChromaDB collection to use.
        embedding_function: A LangChain-compatible embeddings instance.
    """

    def __init__(
        self,
        client: chromadb.PersistentClient,
        collection_name: str,
        embedding_function,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        logger.info("ChromaDB indexer ready (collection=%s)", collection_name)

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def index(self, chunks: list[Document]) -> None:
        """
        Embed and upsert document chunks into the ChromaDB collection.

        Args:
            chunks: List of LangChain Document objects to index.
        """
        logger.info("Indexing %d chunks into ChromaDB…", len(chunks))
        self.vectorstore.add_documents(chunks)
        logger.info("Indexing complete.")

    def get_vectorstore(self) -> Chroma:
        """Return the underlying Chroma vectorstore for retrieval."""
        return self.vectorstore
