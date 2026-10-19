import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


@dataclass
class SearchHit:
    text: str
    source_url: str
    title: str = ""
    score: float = 0.0


class LocalVectorStore:
    """Vector store using FAISS when available, with a pure NumPy fallback."""

    def __init__(self, dim: int = 256):
        self.dim = max(1, dim)
        self.documents: List[str] = []
        self.source_urls: List[str] = []
        self.titles: List[str] = []
        self._vocab: Dict[str, int] = {}
        self._vectors: List[np.ndarray] = []
        self._index = None
        if faiss is not None:
            self._index = faiss.IndexFlatL2(self.dim)

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"\w+", text)]

    def _vectorize(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vector = np.zeros(self.dim, dtype=np.float32)
        if not tokens:
            return vector

        vocab_needed = sorted({token for token in tokens if token})
        for token in vocab_needed:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) % self.dim
            vector[self._vocab[token]] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def add_document(self, text: str, source_url: str = "", title: str = "") -> None:
        if not text or not text.strip():
            return
        vector = self._vectorize(text)
        self.documents.append(text.strip())
        self.source_urls.append(source_url)
        self.titles.append(title)
        self._vectors.append(vector)

        if self._index is not None:
            self._index.add(np.asarray([vector], dtype=np.float32))

    def search(self, query: str, top_k: int = 3) -> List[SearchHit]:
        if not self.documents:
            return []

        query_vec = self._vectorize(query)
        if self._index is not None:
            distances, indices = self._index.search(np.asarray([query_vec], dtype=np.float32), min(top_k, len(self.documents)))
            results: List[SearchHit] = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                score = 1.0 / (1.0 + float(distance))
                results.append(
                    SearchHit(
                        text=self.documents[int(idx)],
                        source_url=self.source_urls[int(idx)],
                        title=self.titles[int(idx)],
                        score=float(score),
                    )
                )
            return results

        scored: List[Tuple[float, int]] = []
        for idx, doc_vector in enumerate(self._vectors):
            similarity = float(np.dot(query_vec, doc_vector))
            scored.append((similarity, idx))

        scored.sort(key=lambda item: item[0], reverse=True)
        hits: List[SearchHit] = []
        for similarity, idx in scored[:top_k]:
            hits.append(
                SearchHit(
                    text=self.documents[idx],
                    source_url=self.source_urls[idx],
                    title=self.titles[idx],
                    score=float(similarity),
                )
            )
        return hits


class PersistentLocalVectorStore(LocalVectorStore):
    """Vector store that persists documents and vectors to disk."""

    def __init__(self, dim: int = 256, persist_dir: str = ".datlas_vectors"):
        super().__init__(dim=dim)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.persist_dir / "metadata.json"
        self.vectors_path = self.persist_dir / "vectors.npy"
        self._load()

    def _load(self) -> None:
        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.documents = payload.get("documents", [])
                self.source_urls = payload.get("source_urls", [])
                self.titles = payload.get("titles", [])
                self._vectors = []
                if self.vectors_path.exists():
                    vectors = np.load(self.vectors_path, allow_pickle=True)
                    if vectors.size > 0:
                        if vectors.ndim == 1:
                            self._vectors = [np.asarray(vectors, dtype=np.float32)]
                        else:
                            self._vectors = [np.asarray(vector, dtype=np.float32) for vector in vectors]
                if faiss is not None:
                    self._index = faiss.IndexFlatL2(self.dim)
                    if self._vectors:
                        self._index.add(np.asarray(self._vectors, dtype=np.float32))
            except Exception:
                self.documents = []
                self.source_urls = []
                self.titles = []
                self._vectors = []
                if faiss is not None:
                    self._index = faiss.IndexFlatL2(self.dim)

    def save(self) -> None:
        payload = {
            "documents": self.documents,
            "source_urls": self.source_urls,
            "titles": self.titles,
        }
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        if self._vectors:
            np.save(self.vectors_path, np.asarray(self._vectors, dtype=np.float32))
        else:
            if self.vectors_path.exists():
                self.vectors_path.unlink()

    def add_document(self, text: str, source_url: str = "", title: str = "") -> None:
        super().add_document(text=text, source_url=source_url, title=title)
        self.save()
