"""
Chunking Module.

Uses MarkdownTextSplitter for markdown/PDF output and
SemanticTextChunker for plain-text / tabular content.
"""

import logging

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as _LCSemanticChunker
from langchain_text_splitters import MarkdownTextSplitter

logger = logging.getLogger(__name__)


class BaseChunker:
    """Abstract base class for all chunking strategies."""

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks."""
        raise NotImplementedError


class MarkdownChunker(BaseChunker):
    """
    Splits markdown-rich content (e.g. PDF output) using
    MarkdownTextSplitter to respect heading boundaries.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self.splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(
            "MarkdownChunker initialised (size=%d, overlap=%d)",
            chunk_size,
            chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split markdown documents into heading-aware chunks."""
        chunks: list[Document] = self.splitter.split_documents(documents)
        logger.info("MarkdownChunker produced %d chunks", len(chunks))
        return chunks


class SemanticTextChunker(BaseChunker):
    """
    Splits content semantically using HuggingFace BERT embeddings
    (local, free, no API key required) to detect topic boundaries.

    Uses langchain-huggingface package to avoid deprecation warnings.
    """

    def __init__(
        self,
        bert_model: str = "all-MiniLM-L6-v2",
        threshold: int = 95,
    ) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(
            "Loading BERT embeddings for SemanticTextChunker (model=%s)…",
            bert_model,
        )
        embeddings = HuggingFaceEmbeddings(model_name=bert_model)

        # Alias avoids name collision with this class
        self.splitter = _LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold,
        )
        logger.info(
            "SemanticTextChunker initialised with BERT (threshold=%d)", threshold
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents using semantic topic boundaries."""
        chunks: list[Document] = self.splitter.split_documents(documents)
        logger.info("SemanticTextChunker produced %d chunks", len(chunks))
        return chunks


class ChunkerFactory:
    """
    Selects the appropriate chunking strategy based on source format.

    Routing logic:
        PDF  → MarkdownChunker    (preserves heading structure)
        else → SemanticTextChunker (topic-aware, local BERT)
    """

    @staticmethod
    def get_chunker(
        source_format: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        openai_api_key: str = "",
        bert_model: str = "all-MiniLM-L6-v2",
        semantic_threshold: int = 95,
    ) -> BaseChunker:
        """
        Return the correct chunker for the given source format.

        Args:
            source_format:      File type label e.g. 'pdf', 'csv', 'html'.
            chunk_size:         Max characters per chunk (markdown only).
            chunk_overlap:      Overlap between chunks (markdown only).
            openai_api_key:     Reserved for future OpenAI-based chunking.
            bert_model:         HuggingFace model for semantic chunking.
            semantic_threshold: Percentile threshold for topic boundaries.

        Returns:
            An instance of a BaseChunker subclass.
        """
        if source_format == "pdf":
            return MarkdownChunker(chunk_size, chunk_overlap)
        return SemanticTextChunker(bert_model, semantic_threshold)
