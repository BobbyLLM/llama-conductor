# vault_ops.py
"""Vault promotion and SUMM pipeline operations."""

import hashlib
import os
import re
import shutil
import time
import uuid
from typing import Any, Dict, List, Tuple

from .config import (
    cfg_get,
    SUMM_PROMPT_PATH,
    VAULT_KB_NAME,
    VAULT_CHUNK_WORDS,
    VAULT_OVERLAP_WORDS,
    KB_PATHS,
)
from .model_calls import call_model_prompt


# ---------------------------------------------------------------------------
# Filesystem SUMM pipeline
# ---------------------------------------------------------------------------

_SUPPORTED_RAW_EXTS = {".md", ".txt", ".pdf", ".html", ".htm"}


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


def summ_one(src_path: str) -> str:
    """Return SUMM markdown content for src_path."""
    prompt = load_summ_prompt()
    body = read_raw_to_text(src_path)

    # Guard: avoid sending insane payloads
    max_chars = int(cfg_get("summ.max_input_chars", 120_000))
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars] + "\n\n...[truncated]...\n"

    full_prompt = (
        f"{prompt.rstrip()}\n\n"
        "---\n"
        "INPUT_TEXT:\n"
        f"{body}\n"
        "---\n"
    )

    return call_model_prompt(
        role="thinker",
        prompt=full_prompt,
        max_tokens=int(cfg_get("summ.max_tokens", 900)),
        temperature=float(cfg_get("summ.temperature", 0.2)),
        top_p=float(cfg_get("summ.top_p", 0.9)),
    )


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


def move_summ_to_vault(source_kbs: set) -> Dict[str, Any]:
    """Promote SUMM_*.md from source_kbs into Qdrant under kb=VAULT_KB_NAME."""
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
