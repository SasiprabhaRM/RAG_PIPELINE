"""
Retriever Module.

Performs similarity search against ChromaDB to return
the top-k most relevant document chunks for a query.
"""

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SimilarityRetriever:
    """
    Wraps ChromaDB with a similarity-search retriever.

    Args:
        vectorstore: Chroma instance from ChromaDBIndexer.
        top_k:       Number of results to return.
    """

    def __init__(self, vectorstore, top_k: int = 5) -> None:
        self.vectorstore = vectorstore
        self.top_k = top_k
        logger.info("SimilarityRetriever ready (top_k=%d)", top_k)

    def retrieve(self, query: str) -> list[Document]:
        """
        Run similarity search and return top-k documents.

        Args:
            query: Natural language query string.

        Returns:
            List of matching LangChain Document objects.
        """
        logger.info("Retrieving for query: '%s'", query)
        results = self.vectorstore.similarity_search(query, k=self.top_k)
        logger.info("Retrieved %d document(s)", len(results))
        return results
