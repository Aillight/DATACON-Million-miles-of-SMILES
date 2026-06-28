from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

from backend.parsing.chunking import TextChunk


EmbeddingFn = Callable[[list[str]], Sequence[Sequence[float]]]
RetrievalMode = Literal["tfidf", "hybrid"]
DEFAULT_HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOMAIN_QUERY_TERMS = {
    "nanozymes": (
        "nanozyme nanoparticle nanoparticles nanomaterial material formula particle size diameter nm "
        "TEM SEM zeta potential hydrodynamic size Km Vmax kcat catalytic activity peroxidase oxidase "
        "catalase SOD Michaelis Menten IC50 cytotoxicity cell viability assay"
    ),
    "oxazolidinones": (
        "oxazolidinone compound antibacterial antimicrobial MIC pMIC minimum inhibitory concentration "
        "ug ml microgram ml activity Staphylococcus Enterococcus Mycobacterium table assay"
    ),
    "benzimidazoles": (
        "benzimidazole compound antibacterial antimicrobial MIC pMIC minimum inhibitory concentration "
        "ug ml microgram ml activity Staphylococcus Enterococcus Mycobacterium table assay"
    ),
}
DEFAULT_QUERY_TERMS = (
    "compound material property value unit table assay activity concentration size diameter kinetics"
)
SECTION_BONUS = {
    "results": 0.08,
    "results and discussion": 0.08,
    "discussion": 0.04,
    "experimental": 0.03,
    "experimental section": 0.03,
    "methods": 0.02,
    "materials and methods": 0.02,
}


@dataclass(frozen=True)
class RankedChunk:
    chunk: TextChunk
    rank: int
    score: float
    matched_terms: tuple[str, ...] = ()
    retrieval_method: str = "tfidf"
    tfidf_score: float = 0.0
    embedding_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunk"] = self.chunk.to_dict()
        return payload


def rank_chunks_for_extraction(
    chunks: list[TextChunk],
    domain: str,
    top_k: int,
    query_text: str | None = None,
    retrieval_mode: RetrievalMode = "tfidf",
    embedding_fn: EmbeddingFn | None = None,
    embedding_weight: float = 0.65,
) -> list[RankedChunk]:
    if top_k <= 0 or not chunks:
        return []

    query = query_text or query_for_domain(domain)
    texts = [chunk.text for chunk in chunks]
    tfidf_scores = vector_similarity_scores(query, texts)
    embedding_scores: list[float] | None = None
    retrieval_method = "tfidf"
    if retrieval_mode == "hybrid" and embedding_fn:
        embedding_scores = embedding_similarity_scores(query, texts, embedding_fn)
        if embedding_scores is not None:
            retrieval_method = "hybrid"
        else:
            retrieval_method = "tfidf_fallback"

    scored: list[tuple[int, float, tuple[str, ...], float, float | None]] = []
    query_terms = extract_query_terms(query)
    for position, chunk in enumerate(chunks):
        tfidf_score = tfidf_scores[position] if position < len(tfidf_scores) else 0.0
        embedding_score = embedding_scores[position] if embedding_scores and position < len(embedding_scores) else None
        score = combine_retrieval_scores(tfidf_score, embedding_score, embedding_weight)
        score += section_bonus(chunk.section_title)
        matches = matched_query_terms(chunk.text, query_terms)
        if matches:
            score += min(0.10, 0.01 * len(matches))
        scored.append((position, score, matches, tfidf_score, embedding_score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    selected = scored[: min(top_k, len(scored))]
    return [
        RankedChunk(
            chunk=chunks[position],
            rank=rank,
            score=round(float(score), 6),
            matched_terms=matches,
            retrieval_method=retrieval_method,
            tfidf_score=round(float(tfidf_score), 6),
            embedding_score=round(float(embedding_score), 6) if embedding_score is not None else None,
        )
        for rank, (position, score, matches, tfidf_score, embedding_score) in enumerate(selected, start=1)
    ]


def query_for_domain(domain: str) -> str:
    normalized = normalize_domain(domain)
    return DOMAIN_QUERY_TERMS.get(normalized, DEFAULT_QUERY_TERMS)


def normalize_domain(domain: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", domain.lower())


def vector_similarity_scores(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return keyword_overlap_scores(query, texts)

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w./%-]{2,}\b",
            max_features=12000,
        )
        matrix = vectorizer.fit_transform([query, *texts])
        scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
    except Exception:
        return keyword_overlap_scores(query, texts)
    return [float(score) if math.isfinite(float(score)) else 0.0 for score in scores]


def embedding_similarity_scores(query: str, texts: list[str], embedding_fn: EmbeddingFn) -> list[float] | None:
    try:
        matrix = coerce_embedding_matrix(embedding_fn([query, *texts]))
    except Exception:
        return None
    if len(matrix) != len(texts) + 1:
        return None
    query_embedding = matrix[0]
    return [cosine_similarity(query_embedding, embedding) for embedding in matrix[1:]]


def make_huggingface_embedding_function(
    model: str = DEFAULT_HF_EMBEDDING_MODEL,
    token: str | None = None,
    client: Any | None = None,
) -> EmbeddingFn:
    if client is None:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=model, token=token)

    def embed(texts: list[str]) -> Sequence[Sequence[float]]:
        response = client.feature_extraction(texts, normalize=True, truncate=True, model=model)
        return coerce_embedding_matrix(response)

    return embed


def combine_retrieval_scores(tfidf_score: float, embedding_score: float | None, embedding_weight: float) -> float:
    if embedding_score is None:
        return tfidf_score
    weight = min(1.0, max(0.0, embedding_weight))
    return (1.0 - weight) * tfidf_score + weight * embedding_score


def coerce_embedding_matrix(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("Embedding response is not a list.")
    if value and isinstance(value[0], (int, float)):
        return [[float(item) for item in value]]

    matrix: list[list[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, list):
            raise ValueError("Embedding row is not a list.")
        matrix.append([float(item) for item in row])
    return matrix


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(item) * float(item) for item in left))
    right_norm = math.sqrt(sum(float(item) * float(item) for item in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = dot / (left_norm * right_norm)
    return float(score) if math.isfinite(score) else 0.0


def keyword_overlap_scores(query: str, texts: list[str]) -> list[float]:
    query_terms = set(extract_query_terms(query))
    if not query_terms:
        return [0.0 for _ in texts]
    scores: list[float] = []
    for text in texts:
        text_terms = set(tokenize(text))
        scores.append(len(query_terms & text_terms) / len(query_terms))
    return scores


def section_bonus(section_title: str) -> float:
    normalized = re.sub(r"[^a-z0-9]+", " ", section_title.lower()).strip()
    return max((bonus for title, bonus in SECTION_BONUS.items() if title in normalized), default=0.0)


def matched_query_terms(text: str, query_terms: Iterable[str], limit: int = 8) -> tuple[str, ...]:
    text_terms = set(tokenize(text))
    matches: list[str] = []
    for term in query_terms:
        if term in text_terms and term not in matches:
            matches.append(term)
        if len(matches) >= limit:
            break
    return tuple(matches)


def extract_query_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    for token in tokenize(query):
        if len(token) < 2:
            continue
        if token not in terms:
            terms.append(token)
    return tuple(terms)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9./%-]*", text.lower())
