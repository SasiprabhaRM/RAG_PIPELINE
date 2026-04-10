"""
Reranker Module.

Uses a CrossEncoder model (local, free) to rerank retrieved
document chunks by relevance to the user query.
"""

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks a list of documents using a CrossEncoder model.

    CrossEncoders score (query, document) pairs jointly, producing
    more accurate relevance scores than bi-encoder similarity alone.

    Args:
        model_name: HuggingFace CrossEncoder model identifier.
        top_k:      Number of top documents to return after reranking.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 3,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        logger.info(
            "CrossEncoderReranker initialised (model=%s, top_k=%d)",
            model_name,
            top_k,
        )

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Score and rerank documents by relevance to the query.

        Args:
            query:     User's natural language query.
            documents: Retrieved document chunks to rerank.

        Returns:
            Top-k documents sorted by descending relevance score.
        """
        if not documents:
            logger.warning("Reranker received 0 documents — skipping.")
            return []

        logger.info("Reranking %d document(s) for query: '%s'", len(documents), query)

        # Build (query, passage) pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        # Zip scores with documents and sort descending
        scored_docs = sorted(
            zip(scores, documents, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )

        top_docs = [doc for _, doc in scored_docs[: self.top_k]]
        logger.info("Reranker returned top %d document(s)", len(top_docs))
        return top_docs
