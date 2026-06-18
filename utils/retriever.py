"""Retrieval utilities for the RAG analyst agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


@dataclass(slots=True)
class RetrievalResult:
    """One retrieved text chunk plus similarity score."""

    rank: int
    score: float
    text: str
    row_index: int


class EcommerceRetriever:
    """Builds row-level retrievable chunks from the e-commerce dataset."""

    def __init__(self, dataframe: pd.DataFrame, text_columns: Iterable[str] | None = None) -> None:
        self.dataframe = dataframe.copy()
        self.text_columns = list(text_columns or dataframe.columns)
        self.documents = self._row_to_documents(self.dataframe)
        self.embedding_model = None
        self.faiss_index = None
        self.embeddings = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.backend = "tfidf"
        self._fit()

    def _should_use_embeddings(self) -> bool:
        """Use transformer embeddings only when explicitly enabled."""
        return os.getenv("ENABLE_SENTENCE_TRANSFORMERS", "").strip().lower() in {"1", "true", "yes"}

    def _load_sentence_transformer(self):
        """Import sentence-transformers lazily to avoid native-library startup crashes."""
        try:  # pragma: no cover - import availability is environment-specific
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None
        return SentenceTransformer

    @classmethod
    def from_csv(cls, csv_path: str) -> "EcommerceRetriever":
        return cls(pd.read_csv(csv_path))

    def _row_to_documents(self, dataframe: pd.DataFrame) -> list[str]:
        docs = []
        for _, row in dataframe.iterrows():
            docs.append(
                ", ".join(
                    f"{column}: {row[column]}"
                    for column in self.text_columns
                )
            )
        return docs

    def _fit(self) -> None:
        if faiss is not None and self._should_use_embeddings():
            sentence_transformer_cls = self._load_sentence_transformer()
            if sentence_transformer_cls is not None:
                try:
                    self.embedding_model = sentence_transformer_cls("all-MiniLM-L6-v2")
                    embeddings = self.embedding_model.encode(self.documents, show_progress_bar=False)
                    self.embeddings = np.asarray(embeddings, dtype="float32")
                    index = faiss.IndexFlatIP(self.embeddings.shape[1])
                    normalized = self.embeddings.copy()
                    faiss.normalize_L2(normalized)
                    index.add(normalized)
                    self.faiss_index = index
                    self.backend = "faiss"
                    return
                except Exception:
                    self.embedding_model = None
                    self.faiss_index = None
                    self.embeddings = None

        if TfidfVectorizer is not None and cosine_similarity is not None:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2048)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            self.backend = "tfidf"
            return

        self.backend = "token_overlap"

    def _expand_query(self, query: str) -> str:
        """Expand user phrasing toward common ecommerce schema variants."""
        lowered = query.lower()
        expansions = []
        synonym_map = {
            "purchase": "purchased purchase_count conversion revenue",
            "purchases": "purchased purchase_count conversion revenue",
            "buy": "purchased purchase_count conversion order",
            "sales": "purchased revenue conversion",
            "brand": "dominant_brand brand",
            "brands": "dominant_brand brand",
            "category": "dominant_category category",
            "categories": "dominant_category category",
            "product": "product unique_products dominant_category dominant_brand",
            "products": "product unique_products dominant_category dominant_brand",
            "customer": "customer segment user visitor dominant_brand dominant_category",
            "customers": "customer segment user visitor dominant_brand dominant_category",
            "channel": "channel source traffic dominant_category dominant_brand",
            "time": "event_month event_date time month",
            "month": "event_month month event_date",
            "cart": "cart_count basket",
            "engagement": "event_count cart_count unique_products avg_price",
        }
        for token, expansion in synonym_map.items():
            if token in lowered:
                expansions.append(expansion)
        return f"{query} {' '.join(expansions)}".strip()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the top matching dataset rows for the query."""
        expanded_query = self._expand_query(query)
        purchase_intent = any(term in expanded_query.lower() for term in ["purchase", "purchased", "conversion", "revenue", "order"])
        if self.backend == "faiss" and self.embedding_model is not None and self.faiss_index is not None:
            query_embedding = np.asarray(self.embedding_model.encode([expanded_query]), dtype="float32")
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            return [
                RetrievalResult(rank=rank + 1, score=float(scores[0][rank]), text=self.documents[idx], row_index=int(idx))
                for rank, idx in enumerate(indices[0])
            ]

        if self.backend == "tfidf":
            query_vector = self.vectorizer.transform([expanded_query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            query_tokens = {token.lower() for token in expanded_query.split() if token.strip()}
            similarities = []
            for document in self.documents:
                doc_tokens = {token.lower().strip(",.:") for token in document.split() if token.strip()}
                denominator = max(len(query_tokens | doc_tokens), 1)
                similarities.append(len(query_tokens & doc_tokens) / denominator)
            similarities = np.asarray(similarities)
        if purchase_intent:
            for candidate in ["purchased", "purchase_count", "revenue", "conversion", "converted"]:
                if candidate in self.dataframe.columns:
                    boost = pd.to_numeric(self.dataframe[candidate], errors="coerce").fillna(0).to_numpy(dtype=float)
                    similarities = similarities + (0.15 * np.clip(boost, 0, 1))
                    break
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            RetrievalResult(rank=rank + 1, score=float(similarities[idx]), text=self.documents[idx], row_index=int(idx))
            for rank, idx in enumerate(top_indices)
        ]

    def build_context(self, query: str, top_k: int = 5) -> tuple[str, list[RetrievalResult]]:
        results = self.retrieve(query, top_k=top_k)
        context = "\n".join(f"[{item.rank}] score={item.score:.3f} | {item.text}" for item in results)
        return context, results
