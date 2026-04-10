"""
LangGraph Router Module.

Defines the RAG graph with the following nodes:
  - router_node:    Classifies query as GENERAL or DOCUMENT.
  - retriever_node: Retrieves top-K chunks from ChromaDB.
  - reranker_node:  Reranks retrieved chunks via CrossEncoder.
  - llm_node:       Generates final answer via Groq LLM.

Edges:
  START → router_node
    → GENERAL  → llm_node → END
    → DOCUMENT → retriever_node → reranker_node → llm_node → END
"""

import logging
from typing import Literal

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.llm import GroqLLM
from src.reranker import CrossEncoderReranker
from src.retriever import SimilarityRetriever

logger = logging.getLogger(__name__)


# ── Graph State ───────────────────────────────────────────────────────────────


class GraphState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    query: str
    route: str
    documents: list[Document]
    reranked: list[Document]
    answer: str


# ── Graph Builder ─────────────────────────────────────────────────────────────


class RAGGraphBuilder:
    """
    Builds and compiles the LangGraph RAG pipeline.

    Args:
        retriever:  SimilarityRetriever instance.
        reranker:   CrossEncoderReranker instance.
        llm:        GroqLLM instance.
        router_prompt_template: Prompt template with {query} placeholder.
    """

    def __init__(
        self,
        retriever: SimilarityRetriever,
        reranker: CrossEncoderReranker,
        llm: GroqLLM,
        router_prompt_template: str,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.router_prompt_template = router_prompt_template
        logger.info("RAGGraphBuilder initialised.")

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _router_node(self, state: GraphState) -> GraphState:
        """
        Classify the query as GENERAL or DOCUMENT using Groq LLM.

        Args:
            state: Current graph state with 'query' populated.

        Returns:
            Updated state with 'route' set to 'GENERAL' or 'DOCUMENT'.
        """
        query = state["query"]
        logger.info("Router node — classifying query: '%s'", query)

        prompt = self.router_prompt_template.format(query=query)
        raw = self.llm.answer_general(prompt).strip().upper()

        # Normalise — default to DOCUMENT if response is ambiguous
        route = "GENERAL" if "GENERAL" in raw else "DOCUMENT"
        logger.info("Router decision: %s", route)

        return {**state, "route": route}

    def _retriever_node(self, state: GraphState) -> GraphState:
        """
        Retrieve top-K relevant chunks from ChromaDB.

        Args:
            state: Current graph state with 'query' populated.

        Returns:
            Updated state with 'documents' populated.
        """
        query = state["query"]
        logger.info("Retriever node — fetching chunks for: '%s'", query)
        docs = self.retriever.retrieve(query)
        logger.info("Retriever node — fetched %d chunk(s)", len(docs))
        return {**state, "documents": docs}

    def _reranker_node(self, state: GraphState) -> GraphState:
        """
        Rerank retrieved chunks using CrossEncoder.

        Args:
            state: Current graph state with 'documents' populated.

        Returns:
            Updated state with 'reranked' top-K documents.
        """
        query = state["query"]
        docs = state["documents"]
        logger.info("Reranker node — reranking %d chunk(s)", len(docs))
        reranked = self.reranker.rerank(query, docs)
        logger.info("Reranker node — top %d chunk(s) selected", len(reranked))
        return {**state, "reranked": reranked}

    def _llm_node(self, state: GraphState) -> GraphState:
        """
        Generate the final answer using Groq LLM.

        For GENERAL route: answers directly from model knowledge.
        For DOCUMENT route: answers grounded in reranked context.

        Args:
            state: Current graph state.

        Returns:
            Updated state with 'answer' populated.
        """
        query = state["query"]
        route = state.get("route", "DOCUMENT")
        logger.info("LLM node — generating answer (route=%s)", route)

        if route == "GENERAL":
            answer = self.llm.answer_general(query)
        else:
            answer = self.llm.answer_with_context(query, state["reranked"])

        logger.info("LLM node — answer generated (%d chars)", len(answer))
        return {**state, "answer": answer}

    # ── Routing Edge ──────────────────────────────────────────────────────────

    def _route_edge(self, state: GraphState) -> Literal["retriever_node", "llm_node"]:
        """
        Conditional edge: route after router_node.

        Returns:
            'llm_node' for GENERAL queries.
            'retriever_node' for DOCUMENT queries.
        """
        if state["route"] == "GENERAL":
            logger.info("Edge -> llm_node (GENERAL path)")
            return "llm_node"
        logger.info("Edge -> retriever_node (DOCUMENT path)")
        return "retriever_node"

    # ── Graph Compilation ─────────────────────────────────────────────────────

    def build(self):
        """
        Compile and return the LangGraph StateGraph.

        Returns:
            A compiled LangGraph app ready for invocation.
        """
        graph = StateGraph(GraphState)

        # Register nodes
        graph.add_node("router_node", self._router_node)
        graph.add_node("retriever_node", self._retriever_node)
        graph.add_node("reranker_node", self._reranker_node)
        graph.add_node("llm_node", self._llm_node)

        # Entry point
        graph.add_edge(START, "router_node")

        # Conditional edge after router
        graph.add_conditional_edges(
            "router_node",
            self._route_edge,
            {
                "llm_node": "llm_node",
                "retriever_node": "retriever_node",
            },
        )

        # Document path edges
        graph.add_edge("retriever_node", "reranker_node")
        graph.add_edge("reranker_node", "llm_node")

        # Both paths end here
        graph.add_edge("llm_node", END)

        compiled = graph.compile()
        logger.info("LangGraph RAG graph compiled successfully.")
        return compiled
