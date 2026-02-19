# vault_ops.py
"""Vault promotion and SUMM pipeline operations."""

import hashlib
import os
import re
import shutil
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from .config import (
    cfg_get,
    SUMM_PROMPT_PATH,
    VAULT_KB_NAME,
    VAULT_CHUNK_WORDS,
    VAULT_OVERLAP_WORDS,
    KB_PATHS,
)


# ---------------------------------------------------------------------------
# Filesystem SUMM pipeline
# ---------------------------------------------------------------------------

_SUPPORTED_RAW_EXTS = {".md", ".txt", ".pdf", ".html", ".htm"}
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_FACT_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|had|has|have|found|show|shows|reported|released|"
    r"permitted|requires|need|priced|continued|include|includes|apply|applied|emit)\b",
    re.I,
)
_NUM_RE = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\b")
_YEAR_RE = re.compile(r"\b(?:19\d{2}|20\d{2})\b")
_MONEY_RE = re.compile(r"\b(?:percent|million|billion|trillion|usd|gbp|eur)\b|%|\$", re.I)
_CAPWORD_RE = re.compile(r"\b[A-Z][a-z]+\b")
_SHORT_CUTOFF_WORDS = 420
_LONG_CUTOFF_WORDS = 1200
_SHORT_RATIO = 0.90
_MID_RATIO = 0.38
_LONG_RATIO = 0.50
_MIN_WORDS = 160
_MAX_SENT_CAP = 44
_MIN_SENT_CAP = 8
_AVG_SENT_WORDS = 22
_MMR_LAMBDA = 0.72
_BUDGET_SLACK = 1.15
_FLOOR_KEEP = 10
_LEAD_BOOST = 0.20
_TAIL_BOOST = 0.05
_NUM_BOOST = 2.40
_MONEY_BOOST = 1.00
_YEAR_BOOST = 0.25
_ENT_BOOST = 0.30
_VERB_BOOST = 0.10
_COVER_TOP_K = 16
_COVER_MIN_TOKEN_LEN = 5
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
    "could", "did", "do", "does", "doing", "for", "from", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
}


def sha256_file(path: str) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text_file(path: str) -> str:
    """Read text file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_text(path: str) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pypdf not installed ({e})")

    reader = PdfReader(path)
    pieces: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            pieces.append(t.strip())
    return "\n\n".join(pieces).strip()


def read_html_text(path: str) -> str:
    """Extract text from HTML file (cheap strip)."""
    raw = read_text_file(path)
    raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
    raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_raw_to_text(path: str) -> str:
    """Read file to text based on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf_text(path)
    if ext in (".html", ".htm"):
        return read_html_text(path)
    return read_text_file(path)


def load_summ_prompt() -> str:
    """Load SUMM prompt from file."""
    with open(SUMM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _clean_summary_text(text: str) -> str:
    s = text or ""
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", s)
    s = s.replace("×", "x").replace("–", "-").replace("—", "-")
    s = re.sub(r"[`*_>#]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _split_sentences(text: str) -> List[str]:
    cleaned = _clean_summary_text(text)
    if not cleaned:
        return []
    parts = _SENT_SPLIT_RE.split(cleaned)
    if len(parts) <= 1:
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p and p.strip()]


def _is_header_like(sentence: str) -> bool:
    s = (sentence or "").strip()
    if not s:
        return True
    tokens = re.findall(r"[A-Za-z]+", s)
    if not tokens:
        return True
    if len(tokens) <= 8 and s[-1] not in ".!?":
        return True
    title_case_count = sum(1 for t in tokens if t[:1].isupper())
    if tokens and (title_case_count / len(tokens)) > 0.75 and s[-1] not in ".!?":
        return True
    return False


def _is_noise_caption(sentence: str) -> bool:
    s = (sentence or "").strip().lower()
    if "photograph:" in s:
        return True
    return s.startswith("thermal video") or s.startswith("emissions coming") or s.startswith("snow on the ground")


def _tokenize(sentence: str) -> List[str]:
    out: List[str] = []
    for tok in _WORD_RE.findall((sentence or "").lower()):
        if len(tok) > 2 and tok not in _STOPWORDS:
            out.append(tok)
    return out


def _factual_signal(sentence: str) -> float:
    score = 0.0
    if _NUM_RE.search(sentence):
        score += _NUM_BOOST
    if _MONEY_RE.search(sentence):
        score += _MONEY_BOOST
    score += _YEAR_BOOST * len(_YEAR_RE.findall(sentence))
    if len(_CAPWORD_RE.findall(sentence)) >= 2:
        score += _ENT_BOOST
    if _FACT_VERB_RE.search(sentence):
        score += _VERB_BOOST
    return score


def _choose_target_words(src_words: int) -> int:
    if src_words < _SHORT_CUTOFF_WORDS:
        ratio = _SHORT_RATIO
    elif src_words < _LONG_CUTOFF_WORDS:
        ratio = _MID_RATIO
    else:
        ratio = _LONG_RATIO
    return max(_MIN_WORDS, int(src_words * ratio))


def _mmr_select(sentences: List[str], scores: List[float], max_sent: int, lam: float) -> List[int]:
    token_sets = [set(_tokenize(s)) for s in sentences]
    selected: List[int] = []
    while len(selected) < min(max_sent, len(sentences)):
        best_idx = None
        best_val = -1e18
        for i in range(len(sentences)):
            if i in selected:
                continue
            relevance = scores[i]
            redundancy = 0.0
            for j in selected:
                a = token_sets[i]
                b = token_sets[j]
                if a and b:
                    redundancy = max(redundancy, len(a & b) / max(1, len(a | b)))
            val = lam * relevance - (1.0 - lam) * redundancy
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)
    return sorted(selected)


def _top_coverage_terms(sentences: List[str]) -> Set[str]:
    c = Counter()
    for s in sentences:
        for tok in _tokenize(s):
            if len(tok) >= _COVER_MIN_TOKEN_LEN and not tok.isdigit():
                c[tok] += 1
    return {tok for tok, _ in c.most_common(_COVER_TOP_K)}


def _extractive_sentences(text: str) -> List[str]:
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p and p.strip()]
    sentences: List[str] = []
    seen: Set[str] = set()

    for paragraph in raw_paragraphs:
        for sent in _split_sentences(paragraph):
            s = sent.strip()
            if len(s) < 28:
                continue
            if _is_header_like(s) or _is_noise_caption(s):
                continue
            norm = re.sub(r"\s+", " ", s).strip().lower()
            if norm in seen:
                continue
            seen.add(norm)
            sentences.append(s)

    if not sentences:
        for sent in _split_sentences(text):
            s = sent.strip()
            if len(s) < 28:
                continue
            if _is_header_like(s) or _is_noise_caption(s):
                continue
            norm = re.sub(r"\s+", " ", s).strip().lower()
            if norm in seen:
                continue
            seen.add(norm)
            sentences.append(s)

    if not sentences:
        return []

    freq = Counter()
    for sent in sentences:
        freq.update(_tokenize(sent))

    scores: List[float] = []
    n_sent = len(sentences)
    for i, sent in enumerate(sentences):
        toks = _tokenize(sent)
        base = sum(freq[t] for t in toks) / max(1, len(toks))
        pos = 0.0
        if i < max(1, n_sent // 8):
            pos += _LEAD_BOOST
        if i > n_sent * 0.85:
            pos += _TAIL_BOOST
        scores.append(base + _factual_signal(sent) + pos)

    src_words = len(_WORD_RE.findall(_clean_summary_text(text)))
    target_words = _choose_target_words(src_words)
    max_sent = min(_MAX_SENT_CAP, max(_MIN_SENT_CAP, target_words // _AVG_SENT_WORDS))

    selected = _mmr_select(sentences, scores, max_sent=max_sent, lam=_MMR_LAMBDA)

    # Ensure early/mid/late coverage if possible.
    if sentences:
        n = len(sentences)
        bands = [range(0, max(1, n // 3)), range(n // 3, max(n // 3 + 1, 2 * n // 3)), range(2 * n // 3, n)]
        for band in bands:
            if any(i in band for i in selected):
                continue
            candidates = [i for i in band if i not in selected]
            if candidates:
                selected.append(max(candidates, key=lambda idx: scores[idx]))

    selected_set: Set[int] = set(selected)
    covered_tokens: Set[str] = set()
    for i in selected_set:
        covered_tokens.update(_tokenize(sentences[i]))
    desired_terms = _top_coverage_terms(sentences)

    budget = int(target_words * _BUDGET_SLACK)
    used_words = sum(len(_WORD_RE.findall(sentences[i])) for i in selected_set)
    candidates = [i for i in range(len(sentences)) if i not in selected_set]
    candidates.sort(key=lambda idx: scores[idx], reverse=True)

    for i in candidates:
        if used_words >= budget or len(selected_set) >= _MAX_SENT_CAP:
            break
        sent_tokens = set(_tokenize(sentences[i]))
        gain = len((sent_tokens & desired_terms) - covered_tokens)
        if gain <= 0:
            continue
        w = len(_WORD_RE.findall(sentences[i]))
        if used_words + w > budget:
            continue
        selected_set.add(i)
        used_words += w
        covered_tokens.update(sent_tokens)

    ranked = sorted(selected_set, key=lambda idx: scores[idx], reverse=True)
    kept: List[int] = []
    kept_words = 0
    for i in ranked:
        w = len(_WORD_RE.findall(sentences[i]))
        if kept_words + w <= budget or len(kept) < _FLOOR_KEEP:
            kept.append(i)
            kept_words += w

    kept = sorted(set(kept))
    return [sentences[i] for i in kept]


def _render_extractive_summary(src_path: str, sentences: List[str]) -> str:
    src_name = os.path.basename(src_path)
    lines = [
        f"# SUMM: {src_name}",
        "method: extractive-stdlib-tuned-v3",
        "",
        "## Extracted Sentences",
    ]
    if not sentences:
        lines.append("- [No extractable sentences found in source text.]")
    else:
        lines.extend(f"- {s}" for s in sentences)
    return "\n".join(lines).strip() + "\n"


def summ_one(src_path: str) -> str:
    """Return deterministic extractive SUMM markdown content for src_path."""
    body = read_raw_to_text(src_path)

    # Guard: avoid processing insane payloads.
    max_chars = int(cfg_get("summ.max_input_chars", 120_000))
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars] + "\n\n...[truncated]...\n"

    sentences = _extractive_sentences(body)
    return _render_extractive_summary(src_path, sentences)


def summ_new_in_kb(kb_name: str, folder: str) -> Dict[str, Any]:
    """Create SUMM_*.md for each supported doc in kb folder, moving originals to /original/."""
    kb = (kb_name or "").strip()
    if not kb:
        return {"summ_created": 0, "summ_skipped": 0, "notes": ["kb name missing"]}

    if not folder or not os.path.isdir(folder):
        return {"summ_created": 0, "summ_skipped": 0, "notes": [f"folder missing: {folder}"]}

    notes: List[str] = []
    created = 0
    skipped = 0

    original_dir = os.path.join(folder, "original")
    os.makedirs(original_dir, exist_ok=True)

    for root, _, files in os.walk(folder):
        # skip /original
        if os.path.abspath(root).lower().startswith(os.path.abspath(original_dir).lower()):
            continue

        for fn in files:
            path = os.path.join(root, fn)
            if not os.path.isfile(path):
                continue

            # Only summarize direct raw docs
            if fn.startswith("SUMM_") and fn.lower().endswith(".md"):
                continue

            ext = os.path.splitext(fn)[1].lower()
            if ext not in _SUPPORTED_RAW_EXTS:
                continue

            base = os.path.splitext(fn)[0]
            summ_name = f"SUMM_{base}.md"
            summ_path = os.path.join(root, summ_name)

            if os.path.exists(summ_path):
                skipped += 1
                continue

            try:
                sha = sha256_file(path)
            except Exception as e:
                notes.append(f"sha fail: {fn}: {e}")
                skipped += 1
                continue

            try:
                summ_md = summ_one(path)
            except Exception as e:
                notes.append(f"SUMM failed: {fn}: {e}")
                skipped += 1
                continue

            header = (
                "<!--\n"
                f"source_file: {fn}\n"
                f"source_rel_path: {os.path.relpath(path, folder)}\n"
                f"source_sha256: {sha}\n"
                f"summ_created_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                "pipeline: SUMM\n"
                "-->\n\n"
            )

            try:
                with open(summ_path, "w", encoding="utf-8") as f:
                    f.write(header)
                    f.write((summ_md or "").strip() + "\n")
            except Exception as e:
                notes.append(f"write fail: {summ_name}: {e}")
                skipped += 1
                continue

            # move original to /original
            try:
                dest = os.path.join(original_dir, fn)
                # avoid overwrite
                if os.path.exists(dest):
                    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
                    dest = os.path.join(original_dir, f"{base}_{ts}{ext}")
                shutil.move(path, dest)
            except Exception as e:
                notes.append(f"move original fail: {fn}: {e}")

            created += 1

    return {"summ_created": created, "summ_skipped": skipped, "notes": notes}


# ---------------------------------------------------------------------------
# Vault promotion (Qdrant)
# ---------------------------------------------------------------------------


def word_chunks(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    """Split text into overlapping word chunks."""
    words = (text or "").split()
    if not words:
        return []

    chunk_words = max(50, int(chunk_words or 600))
    overlap_words = max(0, int(overlap_words or 175))
    if overlap_words >= chunk_words:
        overlap_words = max(0, chunk_words // 4)

    out: List[str] = []
    i = 0
    step = max(1, chunk_words - overlap_words)
    while i < len(words):
        out.append(" ".join(words[i : i + chunk_words]))
        i += step
    return out


def ensure_qdrant_ready() -> Tuple[Any, Any, str, str]:
    """Return (client, embed_model, collection, vector_name)."""
    from .rag import (
        QDRANT_COLLECTION,
        VECTOR_NAME,
        ensure_collection,
        get_embed_model,
        get_qdrant_client,
    )

    ensure_collection()
    return get_qdrant_client(), get_embed_model(), QDRANT_COLLECTION, VECTOR_NAME


def qdrant_delete_for_vault_file(client: Any, *, vault_kb: str, source_kb: str, rel_path: str) -> int:
    """Best-effort delete of existing vault points for a given source_kb+rel_path."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

        flt = Filter(
            must=[
                FieldCondition(key="kb", match=MatchValue(value=vault_kb)),
                FieldCondition(key="source_kb", match=MatchValue(value=source_kb)),
                FieldCondition(key="source_rel_path", match=MatchValue(value=rel_path)),
            ]
        )
        res = client.delete(
            collection_name=cfg_get("rag.collection", None) or "moa_kb_docs",
            points_selector=flt,
            wait=True,
        )
        return 0
    except Exception:
        return 0


def move_summ_to_vault(source_kbs: set, *, newest_only: bool = False) -> Dict[str, Any]:
    """Promote SUMM_*.md from source_kbs into Qdrant under kb=VAULT_KB_NAME.

    newest_only=True promotes only the newest SUMM_*.md per source KB.
    """
    source_kbs = {k.strip() for k in (source_kbs or set()) if k and k.strip()}
    if not source_kbs:
        return {"files": 0, "chunks": 0, "notes": ["no source KBs provided"]}

    # Ensure SUMM.md exists (so users have it next to router)
    if not os.path.isfile(SUMM_PROMPT_PATH):
        return {"files": 0, "chunks": 0, "notes": [f"SUMM.md missing at {SUMM_PROMPT_PATH}"]}

    notes: List[str] = []
    files_done = 0
    chunks_done = 0

    try:
        client, embed_model, collection, vector_name = ensure_qdrant_ready()
    except Exception as e:
        return {"files": 0, "chunks": 0, "notes": [f"Qdrant not ready: {e}"]}

    # import here to avoid hard dependency on qdrant types at startup
    try:
        from qdrant_client.models import PointStruct  # type: ignore
    except Exception as e:
        return {"files": 0, "chunks": 0, "notes": [f"qdrant_client missing: {e}"]}

    for kb in sorted(source_kbs):
        folder = KB_PATHS.get(kb)
        if not folder or not os.path.isdir(folder):
            notes.append(f"KB folder missing: {kb} -> {folder}")
            continue

        # find SUMM_*.md files
        summ_files: List[str] = []
        for root, _, files in os.walk(folder):
            if "original" in {p.lower() for p in root.split(os.sep)}:
                continue
            for fn in files:
                if fn.startswith("SUMM_") and fn.lower().endswith(".md"):
                    summ_files.append(os.path.join(root, fn))

        if newest_only and summ_files:
            # Keep only the single newest SUMM file for this KB.
            summ_files = [max(summ_files, key=lambda p: os.path.getmtime(p))]

        for summ_path in sorted(summ_files):
            rel_path = os.path.relpath(summ_path, folder)

            # replace-by-file: delete existing vault points for this source file
            qdrant_delete_for_vault_file(client, vault_kb=VAULT_KB_NAME, source_kb=kb, rel_path=rel_path)

            try:
                text = read_text_file(summ_path)
            except Exception as e:
                notes.append(f"read fail: {kb}:{rel_path}: {e}")
                continue

            # Strip HTML comment header if present
            if text.lstrip().startswith("<!--"):
                end = text.find("-->")
                if end != -1:
                    text = text[end + 3 :]

            text = (text or "").strip()
            if not text:
                continue

            chunks = word_chunks(text, VAULT_CHUNK_WORDS, VAULT_OVERLAP_WORDS)
            if not chunks:
                continue

            points: List[Any] = []
            # deterministic ids so re-promote is stable
            ns = uuid.UUID("12345678-1234-5678-1234-567812345678")

            for idx, ch in enumerate(chunks):
                try:
                    vec = embed_model.encode(["passage: " + ch], normalize_embeddings=True)[0]
                    vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                except Exception as e:
                    notes.append(f"embed fail: {kb}:{rel_path}#{idx}: {e}")
                    continue

                # id is uuid5(kb|rel_path|idx|hashprefix)
                key = f"{kb}|{rel_path}|{idx}|{hashlib.sha1(ch[:120].encode('utf-8', errors='ignore')).hexdigest()}"
                try:
                    pid = str(uuid.uuid5(ns, key)) if uuid else key
                except Exception:
                    pid = key

                payload = {
                    "kb": VAULT_KB_NAME,
                    "text": ch,
                    "source_kb": kb,
                    "source_rel_path": rel_path,
                    "file": os.path.basename(summ_path),
                    "path": summ_path,
                }

                points.append(
                    PointStruct(
                        id=pid,
                        vector={vector_name: vec_list},
                        payload=payload,
                    )
                )

            if points:
                try:
                    client.upsert(collection_name=collection, points=points, wait=True)
                    files_done += 1
                    chunks_done += len(points)
                except Exception as e:
                    notes.append(f"upsert fail: {kb}:{rel_path}: {e}")

    return {"files": files_done, "chunks": chunks_done, "notes": notes}
