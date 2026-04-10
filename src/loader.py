"""
Document Loader Module.

Supports PDF (pdf4llm), CSV (CSVLoader), Excel (pandas),
HTML file (BSHTMLLoader), and web URL (WebBaseLoader).
"""

import logging
import os
from abc import ABC, abstractmethod

import pandas as pd
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

# ALL imports must be above this line to satisfy E402
logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load source content and return a list of LangChain Documents."""


class PDFDocumentLoader(BaseLoader):
    """
    Loads PDF files using the pdf4llm library for
    high-quality markdown extraction.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        logger.info("Loading PDF: %s", self.file_path)
        try:
            import pdf4llm

            text = pdf4llm.to_markdown(self.file_path)
            logger.info("PDF loaded successfully — %d chars", len(text) if text else 0)

            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": self.file_path,
                        "format": "pdf",
                    },
                )
            ]
        except ImportError as exc:
            logger.error("pdf4llm not found. Run: pip install pdf4llm | %s", exc)
            raise
        except Exception as exc:
            logger.error("PDF loading failed for '%s': %s", self.file_path, exc)
            raise


class CSVDocumentLoader(BaseLoader):
    """Loads CSV files using LangChain CSVLoader."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        logger.info("Loading CSV: %s", self.file_path)
        try:
            loader = CSVLoader(file_path=self.file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["format"] = "csv"
            return docs
        except Exception as exc:
            logger.error("CSV loading failed: %s", exc)
            raise


class ExcelDocumentLoader(BaseLoader):
    """Loads Excel (.xlsx / .xls) files using pandas."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        logger.info("Loading Excel: %s", self.file_path)
        try:
            xl = pd.ExcelFile(self.file_path)
            docs = []
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                text = df.to_string(index=False)
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": self.file_path,
                            "sheet": sheet,
                            "format": "excel",
                        },
                    )
                )
            return docs
        except Exception as exc:
            logger.error("Excel loading failed: %s", exc)
            raise


class HTMLFileLoader(BaseLoader):
    """Loads local HTML files using LangChain BSHTMLLoader."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        logger.info("Loading HTML file: %s", self.file_path)
        try:
            loader = BSHTMLLoader(self.file_path, open_encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["format"] = "html"
            return docs
        except Exception as exc:
            logger.error("HTML file loading failed: %s", exc)
            raise


class WebURLLoader(BaseLoader):
    """Loads a web page using LangChain WebBaseLoader."""

    def __init__(self, url: str) -> None:
        self.url = url

    def load(self) -> list[Document]:
        logger.info("Loading URL: %s", self.url)
        try:
            loader = WebBaseLoader(self.url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["format"] = "url"
            return docs
        except Exception as exc:
            logger.error("URL loading failed: %s", exc)
            raise


class LoaderFactory:
    """Factory for selecting the correct loader."""

    # 2. Use Type[BaseLoader] (Capital T) to indicate these are subclasses
    _LOADER_MAP: dict[str, type[BaseLoader]] = {
        ".pdf": PDFDocumentLoader,
        ".csv": CSVDocumentLoader,
        ".xlsx": ExcelDocumentLoader,
        ".xls": ExcelDocumentLoader,
        ".html": HTMLFileLoader,
        ".htm": HTMLFileLoader,
    }

    @staticmethod
    def get_loader(source: str) -> BaseLoader:
        if source.startswith(("http://", "https://")):
            return WebURLLoader(source)

        ext = os.path.splitext(source)[-1].lower()
        if ext not in LoaderFactory._LOADER_MAP:
            raise ValueError(f"Unsupported file type: '{ext}'")

        loader_cls = LoaderFactory._LOADER_MAP[ext]
        return loader_cls(source)  # type: ignore[call-arg]
