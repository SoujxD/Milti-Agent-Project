"""LangChain-native persistent retriever backed by ChromaDB.

This module exposes a single entry point, :func:`build_or_load_retriever`, that
returns a LangChain ``BaseRetriever``. The preferred backend is a persistent
ChromaDB vector store built with HuggingFace sentence-transformer embeddings.

If ChromaDB or the embedding model is unavailable (for example, no network to
download the model on first run), the function transparently falls back to
wrapping the existing TF-IDF retriever from :mod:`utils.retriever` behind a
minimal ``BaseRetriever`` subclass. This keeps every downstream caller
retriever-agnostic and preserves the project's no-API-key mock/offline path.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from utils.retriever import EcommerceRetriever


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_DOCS = 500


# ─────────────────────────────────────────
# Document construction
# ─────────────────────────────────────────
def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    normalized = {column: _normalize(column) for column in df.columns}
    for keyword in keywords:
        probe = _normalize(keyword)
        for column in df.columns:
            if probe in normalized[column]:
                return column
    return None


def _conversion_rate(series: pd.Series) -> float | None:
    """Best-effort conversion/revenue rate from bool, 0/1, or labelled columns."""
    if pd.api.types.is_bool_dtype(series):
        return float(series.mean())
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if not numeric.empty and set(numeric.unique()).issubset({0, 1}):
        return float(numeric.mean())
    lowered = series.astype(str).str.strip().str.lower()
    truthy = lowered.isin({"true", "yes", "y", "1", "purchase", "purchased", "converted"})
    if truthy.any():
        return float(truthy.mean())
    return None


def _row_to_document(row: pd.Series, row_id: int) -> Document:
    content = " | ".join(f"{column}={row[column]}" for column in row.index)
    return Document(page_content=content, metadata={"row_id": row_id})


def _summary_document(df: pd.DataFrame) -> Document:
    """A single dataset-level aggregate document to ground broad questions."""
    parts: list[str] = ["DATASET SUMMARY", f"total_rows={len(df)}"]

    revenue_col = _find_column(df, ["revenue", "converted", "conversion", "purchase", "purchased", "order"])
    if revenue_col is not None:
        rate = _conversion_rate(df[revenue_col])
        if rate is not None:
            parts.append(f"{revenue_col}_rate={rate:.4f}")

    traffic_col = _find_column(df, ["traffic", "channel", "source", "medium", "campaign", "acquisition"])
    if traffic_col is not None:
        top = df[traffic_col].astype(str).value_counts().head(3)
        top_str = ", ".join(f"{value} ({count})" for value, count in top.items())
        parts.append(f"top_{_normalize(traffic_col)}={top_str}")

    return Document(page_content=" | ".join(parts), metadata={"row_id": -1, "kind": "summary"})


def _build_documents(csv_path: str | Path) -> list[Document]:
    df = pd.read_csv(csv_path)
    sample = (
        df.sample(n=MAX_DOCS, random_state=42) if len(df) > MAX_DOCS else df
    )
    documents = [_row_to_document(row, int(idx)) for idx, row in sample.iterrows()]
    documents.append(_summary_document(df))
    return documents


# ─────────────────────────────────────────
# TF-IDF fallback retriever (LangChain-compatible)
# ─────────────────────────────────────────
class TfidfBackedRetriever(BaseRetriever):
    """Minimal ``BaseRetriever`` wrapping the legacy :class:`EcommerceRetriever`."""

    ecommerce_retriever: Any
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        results = self.ecommerce_retriever.retrieve(query, top_k=self.k)
        return [
            Document(
                page_content=item.text,
                metadata={"row_id": item.row_index, "score": item.score, "rank": item.rank},
            )
            for item in results
        ]


def _build_tfidf_retriever(csv_path: str | Path, k: int) -> BaseRetriever:
    df = pd.read_csv(csv_path)
    return TfidfBackedRetriever(ecommerce_retriever=EcommerceRetriever(df), k=k)


# ─────────────────────────────────────────
# Chroma backend
# ─────────────────────────────────────────
def _build_or_load_chroma(csv_path: str | Path, k: int, persist_dir: str | Path) -> BaseRetriever:
    # Imports are local so an ImportError here triggers the TF-IDF fallback.
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    with warnings.catch_warnings():
        # langchain-community emits a deprecation notice for HuggingFaceEmbeddings;
        # the spec pins this import, so silence the noise but keep using it.
        warnings.simplefilter("ignore")
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    persist_path = Path(persist_dir)
    if persist_path.exists() and any(persist_path.iterdir()):
        vectorstore = Chroma(persist_directory=str(persist_path), embedding_function=embeddings)
    else:
        persist_path.mkdir(parents=True, exist_ok=True)
        documents = _build_documents(csv_path)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_path),
        )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ─────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────
def build_or_load_retriever(
    csv_path: str | Path = "data/dataset.csv",
    k: int = 5,
    persist_dir: str | Path = "data/chroma_db",
) -> BaseRetriever:
    """Return a LangChain retriever, preferring a persistent Chroma store.

    Falls back to a TF-IDF-backed retriever when ChromaDB or the embedding
    model cannot be loaded, so callers never need an API key or network access.
    """
    try:
        return _build_or_load_chroma(csv_path, k, persist_dir)
    except Exception as exc:  # pragma: no cover - environment-dependent
        warnings.warn(
            f"Chroma/embeddings unavailable ({exc!r}); falling back to TF-IDF retriever.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _build_tfidf_retriever(csv_path, k)
