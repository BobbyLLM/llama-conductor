# rag.py
# version 1.0.1
"""
RAG layer for MoA router.

Responsibilities:
    - Embed queries with intfloat/e5-small-v2
    - Retrieve candidate chunks from Qdrant, filtered by KB tag(s)
    - Rerank candidates with cross-encoder/ms-marco-TinyBERT-L-2-v2
    - Build a compact FACTS block string for Serious / Mentats pipelines

Hardening / compatibility:
    - Uses NAMED VECTORS (VECTOR_NAME) consistently (ingest + query).
    - Uses Qdrant Universal Query API via client.query_points(...) (preferred).
    - Falls back to search_points(...) for older python clients if query_points is missing.
    - Ensures collection exists and *matches* expected named vector + dimension.
    - Does NOT silently delete data on mismatch by default (ALLOW_RECREATE_ON_MISMATCH = False).
"""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set, Tuple

from sentence_transformers import CrossEncoder, SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# NOTE (v1.0.1): RAG can be configured via router_config.yaml under the `rag:` key.
# All settings are optional; defaults below remain the source of truth if not set.

def _load_router_cfg(path: str = 'router_config.yaml'):
    try:
        import yaml  # local import to avoid hard dependency if unused
        p = Path(path)
        if not p.exists():
            return {}
        with p.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _apply_rag_cfg():
    cfg = _load_router_cfg().get('rag', {}) or {}

    # Connection / schema
    global QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, VECTOR_NAME
    QDRANT_HOST = str(cfg.get('qdrant_host', QDRANT_HOST))
    QDRANT_PORT = int(cfg.get('qdrant_port', QDRANT_PORT))
    QDRANT_COLLECTION = str(cfg.get('qdrant_collection', QDRANT_COLLECTION))
    VECTOR_NAME = str(cfg.get('vector_name', VECTOR_NAME))

    # Models (ranker/reranker)
    global EMBED_MODEL_NAME, RERANK_MODEL_NAME
    if 'embed_model' in cfg:
        EMBED_MODEL_NAME = str(cfg.get('embed_model') or EMBED_MODEL_NAME)
    if 'embed_model_name' in cfg:
        EMBED_MODEL_NAME = str(cfg.get('embed_model_name') or EMBED_MODEL_NAME)

    if 'rerank_model' in cfg:
        RERANK_MODEL_NAME = str(cfg.get('rerank_model') or RERANK_MODEL_NAME)
    if 'rerank_model_name' in cfg:
        RERANK_MODEL_NAME = str(cfg.get('rerank_model_name') or RERANK_MODEL_NAME)

    # Defaults
    global DEFAULT_TOP_K, DEFAULT_RERANK_TOP_N, DEFAULT_MAX_CHARS
    if 'top_k' in cfg:
        DEFAULT_TOP_K = int(cfg.get('top_k') or DEFAULT_TOP_K)
    if 'rerank_top_n' in cfg:
        DEFAULT_RERANK_TOP_N = int(cfg.get('rerank_top_n') or DEFAULT_RERANK_TOP_N)
    if 'max_chars' in cfg:
        DEFAULT_MAX_CHARS = int(cfg.get('max_chars') or DEFAULT_MAX_CHARS)

    global ALLOW_RECREATE_ON_MISMATCH
    if 'allow_recreate_on_mismatch' in cfg:
        ALLOW_RECREATE_ON_MISMATCH = bool(cfg.get('allow_recreate_on_mismatch'))




QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

QDRANT_COLLECTION = "moa_kb_docs"

# Named vector schema
VECTOR_NAME = "e5"

# Embedding model (e5-small-v2 is 384-dim)
EMBED_MODEL_NAME = "intfloat/e5-small-v2"

# Cross-encoder reranker
RERANK_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

# Defaults
DEFAULT_TOP_K = 6
DEFAULT_RERANK_TOP_N = 4
DEFAULT_MAX_CHARS = 1200

# Safety default: do not auto-delete/recreate a mismatched collection silently.
ALLOW_RECREATE_ON_MISMATCH = False

# Apply config overrides after defaults are defined (safe no-op if config is absent).
_apply_rag_cfg()

# Lazy-loaded singletons
_client: Optional[QdrantClient] = None
_embed_model: Optional[SentenceTransformer] = None
_rerank_model: Optional[CrossEncoder] = None


# ---------------------------------------------------------------------------
# Helpers: clients/models
# ---------------------------------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_rerank_model() -> Optional[CrossEncoder]:
    """Return a CrossEncoder reranker, or None if reranking is disabled.

    Disable by setting rag.rerank_model_name (or rag.rerank_model) to '', 'none', or 'disabled' in router_config.yaml.
    """
    global _rerank_model
    name = (RERANK_MODEL_NAME or '').strip().lower()
    if name in ('', 'none', 'disabled', 'off', 'false', '0'):
        return None
    if _rerank_model is None:
        _rerank_model = CrossEncoder(RERANK_MODEL_NAME)
    return _rerank_model


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------

def _format_qdrant_error(e: Exception) -> str:
    if isinstance(e, UnexpectedResponse):
        return f"{e}\nRaw response content:\n{getattr(e, 'content', b'')!r}"
    return str(e)


# ---------------------------------------------------------------------------
# Collection schema management
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectionSchema:
    vector_name: str
    dim: int
    distance: Distance = Distance.COSINE


def _desired_schema() -> CollectionSchema:
    dim = get_embed_model().get_sentence_embedding_dimension()
    return CollectionSchema(vector_name=VECTOR_NAME, dim=dim, distance=Distance.COSINE)


def ensure_collection(*, allow_recreate: bool = ALLOW_RECREATE_ON_MISMATCH) -> None:
    """
    Ensure QDRANT_COLLECTION exists and matches expected schema:
      - Named vector VECTOR_NAME must exist
      - Dimension must match embed model dim

    If missing -> create.
    If mismatch -> raise unless allow_recreate=True (then recreate).
    """
    client = get_qdrant_client()
    schema = _desired_schema()

    def recreate() -> None:
        print(
            f"[rag] (re)creating collection '{QDRANT_COLLECTION}' "
            f"vectors={{'{schema.vector_name}': dim={schema.dim}}}"
        )
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                schema.vector_name: VectorParams(size=schema.dim, distance=schema.distance)
            },
        )

    # Create if missing
    try:
        info = client.get_collection(QDRANT_COLLECTION)
    except Exception:
        recreate()
        return

    # Validate schema
    vectors = info.config.params.vectors

    # Named vectors case: dict-like
    if isinstance(vectors, dict):
        if schema.vector_name not in vectors:
            msg = (
                f"[rag] schema mismatch: collection '{QDRANT_COLLECTION}' missing named vector "
                f"'{schema.vector_name}'. Existing={list(vectors.keys())}"
            )
            if allow_recreate:
                print(msg)
                recreate()
                return
            raise RuntimeError(msg)

        existing = vectors[schema.vector_name]
        existing_dim = getattr(existing, "size", None)
        if existing_dim != schema.dim:
            msg = (
                f"[rag] schema mismatch: collection '{QDRANT_COLLECTION}' vector '{schema.vector_name}' "
                f"dim={existing_dim} but embed dim={schema.dim}"
            )
            if allow_recreate:
                print(msg)
                recreate()
                return
            raise RuntimeError(msg)

        return

    # Unnamed vector case: VectorParams
    existing_dim = getattr(vectors, "size", None)
    msg = (
        f"[rag] schema mismatch: collection '{QDRANT_COLLECTION}' uses UNNAMED vectors "
        f"(dim={existing_dim}), but code expects named vector '{schema.vector_name}' (dim={schema.dim})."
    )
    if allow_recreate:
        print(msg)
        recreate()
        return
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Query embedding + filters
# ---------------------------------------------------------------------------

def embed_query(query: str) -> List[float]:
    """
    Embed a query string using e5-small-v2.

    Per e5 convention, queries should be prefixed with "query: ".
    """
    model = get_embed_model()
    vec = model.encode(f"query: {query}", normalize_embeddings=True)
    return vec.tolist()


def _kb_filter(attached_kbs: Set[str]) -> Filter:
    attached = {k.strip() for k in attached_kbs if (k or "").strip()}
    if not attached:
        # caller should already handle this, but keep it safe
        return Filter()

    if len(attached) == 1:
        kb_name = next(iter(attached))
        return Filter(
            must=[FieldCondition(key="kb", match=MatchValue(value=kb_name))]
        )

    return Filter(
        must=[FieldCondition(key="kb", match=MatchAny(any=list(attached)))]
    )


# ---------------------------------------------------------------------------
# Retrieval (Universal Query API preferred)
# ---------------------------------------------------------------------------

def semantic_search(query: str, attached_kbs: Set[str], top_k: int = DEFAULT_TOP_K) -> List[Any]:
    """
    Run a semantic search in Qdrant restricted to kb in attached_kbs.

    Preferred: Universal Query API via client.query_points(..., using=VECTOR_NAME)
    Fallback: search_points(...) with SearchRequest (older python client).
    """
    query = (query or "").strip()
    if not query or not attached_kbs:
        return []

    try:
        ensure_collection()
    except Exception as e:
        print(f"[rag] ensure_collection failed: {e}")
        return []

    client = get_qdrant_client()
    query_vec = embed_query(query)
    kb_filter = _kb_filter(attached_kbs)

    # ----------------------------
    # Preferred: query_points()
    # ----------------------------
    if hasattr(client, "query_points"):
        try:
            res = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vec,           # dense vector
                using=VECTOR_NAME,         # named vector selector
                query_filter=kb_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            # QueryResponse has .points
            points = getattr(res, "points", None)
            return points or []
        except Exception as e:
            print(f"[rag] query_points failed: {_format_qdrant_error(e)}")
            return []

    # ----------------------------
    # Fallback: search_points()
    # ----------------------------
    try:
        from qdrant_client.models import SearchRequest
    except Exception as e:
        print(f"[rag] cannot import SearchRequest for fallback: {e}")
        return []

    try:
        # Try named-vector dict form first (often the most compatible)
        req = SearchRequest(
            vector={VECTOR_NAME: query_vec},
            filter=kb_filter,
            limit=top_k,
            with_payload=True,
        )
        res = client.search_points(collection_name=QDRANT_COLLECTION, query=req)
        return getattr(res, "result", res) or []
    except TypeError:
        # Some client versions want (name, vector) tuple
        try:
            req = SearchRequest(
                vector=(VECTOR_NAME, query_vec),
                filter=kb_filter,
                limit=top_k,
                with_payload=True,
            )
            res = client.search_points(collection_name=QDRANT_COLLECTION, query=req)
            return getattr(res, "result", res) or []
        except Exception as e:
            print(f"[rag] search_points fallback failed (tuple): {_format_qdrant_error(e)}")
            return []
    except Exception as e:
        print(f"[rag] search_points fallback failed (dict): {_format_qdrant_error(e)}")
        return []


# ---------------------------------------------------------------------------
# Reranking + block building
# ---------------------------------------------------------------------------

def rerank(query: str, hits: Iterable[Any], top_n: int = DEFAULT_RERANK_TOP_N) -> List[Any]:
    """
    Rerank raw Qdrant hits with a cross-encoder and return top_n.

    If reranking is disabled, returns the first top_n hits in original order.

    hits: iterable of Qdrant scored points (expects .payload['text']).
    """
    hits_list = list(hits)
    if not hits_list or top_n <= 0:
        return []

    rerank_model = get_rerank_model()
    if rerank_model is None:
        return hits_list[:top_n]

    passages: List[str] = []
    idx_map: List[int] = []

    for i, h in enumerate(hits_list):
        payload = getattr(h, 'payload', None) or {}
        text = str(payload.get('text', '')).strip()
        if not text:
            continue
        passages.append(text)
        idx_map.append(i)

    if not passages:
        return []

    pairs = [(query, p) for p in passages]
    try:
        scores = rerank_model.predict(pairs)
        scores_list = list(getattr(scores, 'tolist', lambda: scores)())
    except Exception as e:
        print(f'[rag] rerank failed: {e}')
        return hits_list[:top_n]

    scored: List[Tuple[int, float]] = list(zip(idx_map, scores_list))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [hits_list[idx] for idx, _ in scored[:top_n]]


def build_rag_block(
    query: str,
    attached_kbs: Set[str],
    *,
    top_k: int = DEFAULT_TOP_K,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """
    High-level helper: build a FACTS block from RAG.

    - Uses e5 to embed the query.
    - Searches Qdrant collection restricted to kb in attached_kbs.
    - Reranks with TinyBERT cross-encoder.
    - Returns a single string containing the top passages, separated by blank lines.

    Fail-open: returns "" if anything fails.
    """
    query = (query or "").strip()
    if not query or not attached_kbs:
        return ""

    hits = semantic_search(query, attached_kbs, top_k=top_k)
    if not hits:
        return ""

    top_hits = rerank(query, hits, top_n=rerank_top_n)
    if not top_hits:
        return ""

    pieces: List[str] = []
    total_chars = 0

    for h in top_hits:
        payload = getattr(h, "payload", None) or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            continue

        if max_chars > 0 and total_chars >= max_chars:
            break

        remaining = max_chars - total_chars if max_chars > 0 else len(text)
        snippet = text if remaining >= len(text) else text[: max(0, remaining)]
        if not snippet:
            break

        pieces.append(snippet)
        total_chars += len(snippet) + 2  # "\n\n"

    return "\n\n".join(pieces) if pieces else ""

# ---------------------------------------------------------------------------
# Public config access (back-compat)
# ---------------------------------------------------------------------------

def get_rag_config() -> dict:
    """Return the effective RAG configuration (defaults + YAML overrides).

    This exists for backwards compatibility with modules that import
    `get_rag_config` from `rag`.

    It MUST be safe to call at import time (no model loading).
    """
    # Read raw config (may be empty)
    raw = _load_router_cfg().get('rag', {}) or {}

    # Build effective config using current globals (already include overrides)
    effective = {
        'qdrant_host': QDRANT_HOST,
        'qdrant_port': QDRANT_PORT,
        'qdrant_collection': QDRANT_COLLECTION,
        'vector_name': VECTOR_NAME,
        'embed_model_name': EMBED_MODEL_NAME,
        'rerank_model_name': RERANK_MODEL_NAME,
        'top_k': DEFAULT_TOP_K,
        'rerank_top_n': DEFAULT_RERANK_TOP_N,
        'max_chars': DEFAULT_MAX_CHARS,
        'allow_recreate_on_mismatch': ALLOW_RECREATE_ON_MISMATCH,
    }

    # Preserve any additional user-provided keys for transparency
    for k, v in raw.items():
        if k not in effective:
            effective[k] = v

    return effective

