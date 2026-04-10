"""
LLM Module.

Wraps the Groq LLM (via langchain-groq) for both direct
general answers and document-grounded RAG answers.
"""

import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


class GroqLLM:
    """
    Wrapper around ChatGroq for two answer modes:
      - Direct: answer general queries from model knowledge.
      - Grounded: answer document queries using reranked context.

    Args:
        model:       Groq model name e.g. 'llama3-8b-8192'.
        api_key:     Groq API key.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens:  Maximum tokens in the response.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("GroqLLM initialised (model=%s)", model)

    def answer_general(self, query: str) -> str:
        """
        Answer a general query directly from model knowledge.

        Args:
            query: User's natural language question.

        Returns:
            LLM response string.
        """
        logger.info("GroqLLM answering general query: '%s'", query)
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. "
                    "Answer the user's question clearly and concisely."
                )
            ),
            HumanMessage(content=query),
        ]
        response = self.llm.invoke(messages)
        answer: str = response.content
        logger.info("GroqLLM general answer generated (%d chars)", len(answer))
        return answer

    def answer_with_context(
        self,
        query: str,
        documents: list[Document],
    ) -> str:
        """
        Answer a document query grounded in retrieved context.

        Args:
            query:     User's natural language question.
            documents: Reranked document chunks as context.

        Returns:
            LLM response string grounded in the provided documents.
        """
        logger.info(
            "GroqLLM answering document query with %d chunk(s): '%s'",
            len(documents),
            query,
        )
        context = "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'N/A')}\n{doc.page_content}"
            for doc in documents
        )
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. "
                    "Answer the user's question using ONLY the context provided below. "
                    "If the answer is not in the context, say so clearly.\n\n"
                    f"Context:\n{context}"
                )
            ),
            HumanMessage(content=query),
        ]
        response = self.llm.invoke(messages)
        answer = response.content
        logger.info("GroqLLM document answer generated (%d chars)", len(answer))
        return answer
