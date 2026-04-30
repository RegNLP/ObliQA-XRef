from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .eval import run_evaluation
from .fusion import RRFFusion
from .rerank import CrossEncoderReranker
from .xref_expand import XRefGraph, expand_retrieval_run, load_xref_graph

__all__ = [
    "run_evaluation",
    "BM25Retriever",
    "DenseRetriever",
    "RRFFusion",
    "CrossEncoderReranker",
    "XRefGraph",
    "load_xref_graph",
    "expand_retrieval_run",
]
"""
IR module: Indexing and retrieval for ObliQA-XRef.

Implements multiple retrieval methods:
  - BM25 (sparse/lexical)
  - Dense retrieval (E5, BGE)
  - RRF fusion
  - Cross-encoder reranking
"""

from .types import RetrievalRun, SearchResult

__all__ = [
    "SearchResult",
    "RetrievalRun",
    "BM25Retriever",
    "DenseRetriever",
    "RRFFusion",
    "CrossEncoderReranker",
    "XRefGraph",
    "load_xref_graph",
    "expand_retrieval_run",
]
