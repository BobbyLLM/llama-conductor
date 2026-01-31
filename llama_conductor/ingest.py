# ingest.py
# version 1.0.4
"""KB summarisation (SUMM) + Vault promotion (Qdrant).

This file replaces the old "ingest everything into Qdrant" behavior.

Current workflow:
  1) Filesystem KBs (folders) are the day-to-day, inspectable knowledge.
     - You drop raw docs into a KB folder.
     - You run `>>summ <kb>` (router command) to generate SUMM_*.md files.
     - The original file is moved into `<kb>/original/`.
     - Serious/Fun retrieval uses SUMM_*.md directly (filesystem retrieval).

  2) Vault is Qdrant (kb label: vault_kb_name, default "vault").
     - `>>move to vault` chunks+embeds SUMM_*.md and writes to Qdrant as kb="vault".
     - Replace-by-file semantics: promoting a SUMM file overwrites its prior vault chunks.

Important:
  - `summ_*` never writes to Qdrant.
  - `move_to_vault` is the *only* place in this module that writes to Qdrant.

Router integration:
  - router passes a `call_model(role, prompt, ...)` function for SUMM generation.
  - move_to_vault uses rag.py helpers for Qdrant + embeddings.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Qdrant + embedding helpers (Vault only)
from .rag import (
    QDRANT_COLLECTION,
    VECTOR_NAME,
    ensure_collection,
    get_embed_model,
    get_qdrant_client,
)

try:
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover
    qmodels = None  # type: ignore


# --------------------------
# File reading helpers
# --------------------------

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s)
    s = re.sub(r"(?is)<.*?>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_text(path: Path) -> str:
    # Optional dependency: pypdf
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


SUPPORTED_EXTS = {".md", ".txt", ".html", ".htm", ".pdf"}


def load_document_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf_text(path)
    if ext in (".html", ".htm"):
        return _strip_html(_read_text_file(path))
    if ext in (".md", ".txt"):
        return _read_text_file(path)
    return ""


# --------------------------
# SUMM prompt
# --------------------------

def load_summ_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8", errors="replace").strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _summ_output_name(src: Path) -> str:
    # SUMM_<stem>.md
    # (stem includes dots; keep it stable)
    safe_stem = src.stem
    return f"SUMM_{safe_stem}.md"


def _extract_existing_source_sha(summ_md_text: str) -> Optional[str]:
    # header line: Source-SHA256: <hash>
    m = re.search(r"(?im)^\s*Source-SHA256:\s*([0-9a-f]{64})\s*$", summ_md_text or "")
    return m.group(1) if m else None


def write_summ_file(
    out_path: Path,
    *,
    source_rel_path: str,
    source_sha256: str,
    model_id: str,
    summ_text: str,
) -> None:
    header = [
        "---",
        f"Source-File: {source_rel_path}",
        f"Source-SHA256: {source_sha256}",
        f"SUMM-Created: {_now_iso()}",
        f"SUMM-Model: {model_id}",
        "---",
        "",
    ]
    out_path.write_text("\n".join(header) + (summ_text.strip() + "\n"), encoding="utf-8")


# --------------------------
# Chunking for Vault promotion
# --------------------------

def chunk_text_sliding(
    text: str,
    *,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if chunk_chars <= 0:
        return [t]

    chunks: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = t[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(j - max(0, overlap_chars), i + 1)
    return chunks


def strip_summ_frontmatter(summ_text: str) -> str:
    """Remove leading YAML-ish frontmatter block starting with --- and ending with ---."""
    txt = (summ_text or "").lstrip()
    if not txt.startswith("---"):
        return summ_text
    # Find the second --- delimiter
    parts = txt.split("\n")
    if len(parts) < 3:
        return summ_text
    # find end delimiter line index >0
    end = None
    for idx in range(1, len(parts)):
        if parts[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return summ_text
    return "\n".join(parts[end + 1 :]).lstrip()


# --------------------------
# Public API: SUMM
# --------------------------

CallModelFn = Callable[..., str]


@dataclass
class SummResult:
    kb: str
    processed: int
    created: int
    skipped: int
    errors: int
    details: List[str]


def summ_kb(
    kb_name: str,
    kb_folder: str,
    *,
    call_model: CallModelFn,
    summarizer_role: str = "thinker",
    summ_prompt_path: Optional[str] = None,
    move_originals: bool = True,
) -> SummResult:
    """Generate SUMM_*.md for each supported doc in kb folder.

    - Uses SUMM.md prompt file (default: alongside this ingest.py).
    - Writes SUMM_<stem>.md into kb folder.
    - Moves originals into kb/original/ (optional).
    """
    folder = Path(kb_folder)
    details: List[str] = []
    if not folder.exists():
        return SummResult(kb=kb_name, processed=0, created=0, skipped=0, errors=1, details=[f"[summ] missing folder: {kb_folder}"])

    prompt_path = Path(summ_prompt_path) if summ_prompt_path else (Path(__file__).parent / "SUMM.md")
    if not prompt_path.exists():
        return SummResult(kb=kb_name, processed=0, created=0, skipped=0, errors=1, details=[f"[summ] missing SUMM prompt: {prompt_path}"])

    summ_prompt = load_summ_prompt(prompt_path)

    original_dir = folder / "original"
    if move_originals:
        original_dir.mkdir(parents=True, exist_ok=True)

    processed = created = skipped = errors = 0

    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if p.name.startswith("SUMM_") and p.suffix.lower() == ".md":
            continue
        if p.name.lower() == "readme.md":
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue

        processed += 1

        try:
            raw_bytes = _read_bytes(p)
            src_sha = _sha256_bytes(raw_bytes)
            src_rel = p.name  # keep simple & portable; router shows kb + filename anyway

            out_name = _summ_output_name(p)
            out_path = folder / out_name

            # Idempotence: if SUMM exists with same source sha, skip
            if out_path.exists():
                existing = out_path.read_text(encoding="utf-8", errors="replace")
                existing_sha = _extract_existing_source_sha(existing)
                if existing_sha == src_sha:
                    skipped += 1
                    details.append(f"[summ:{kb_name}] SKIP (unchanged) {p.name} -> {out_name}")
                    # Still archive original if user left it in root
                    if move_originals:
                        try:
                            shutil.move(str(p), str(original_dir / p.name))
                        except Exception:
                            pass
                    continue

            doc_text = load_document_text(p)
            if not doc_text.strip():
                errors += 1
                details.append(f"[summ:{kb_name}] ERROR (no text) {p.name}")
                continue

            # Compose prompt
            full_prompt = (
                f"{summ_prompt.strip()}\n\n"
                f"---\n"
                f"SOURCE_FILE: {src_rel}\n"
                f"SOURCE_SHA256: {src_sha}\n"
                f"---\n\n"
                f"{doc_text.strip()}\n"
            )

            # call local model
            summ_out = (call_model(role=summarizer_role, prompt=full_prompt, max_tokens=2048) or "").strip()
            if not summ_out:
                errors += 1
                details.append(f"[summ:{kb_name}] ERROR (empty model output) {p.name}")
                continue

            # write SUMM file with provenance
            model_id = summarizer_role
            write_summ_file(
                out_path,
                source_rel_path=src_rel,
                source_sha256=src_sha,
                model_id=model_id,
                summ_text=summ_out,
            )
            created += 1
            details.append(f"[summ:{kb_name}] OK {p.name} -> {out_name}")

            # archive original
            if move_originals:
                try:
                    shutil.move(str(p), str(original_dir / p.name))
                except Exception:
                    # don't fail summarisation if move fails
                    details.append(f"[summ:{kb_name}] WARN could not move original: {p.name}")

        except Exception as e:
            errors += 1
            details.append(f"[summ:{kb_name}] ERROR {p.name}: {e}")

    return SummResult(kb=kb_name, processed=processed, created=created, skipped=skipped, errors=errors, details=details)


def summ_all(
    kb_paths: Dict[str, str],
    *,
    call_model: CallModelFn,
    summarizer_role: str = "thinker",
    summ_prompt_path: Optional[str] = None,
) -> List[SummResult]:
    out: List[SummResult] = []
    for kb, folder in (kb_paths or {}).items():
        out.append(
            summ_kb(
                kb,
                folder,
                call_model=call_model,
                summarizer_role=summarizer_role,
                summ_prompt_path=summ_prompt_path,
            )
        )
    return out


# Backwards compatible names (older router used ingest_*):
ingest_kb = summ_kb
ingest_all = summ_all


# --------------------------
# Public API: MOVE TO VAULT
# --------------------------

@dataclass
class VaultMoveResult:
    vault_kb_name: str
    files_considered: int
    files_indexed: int
    chunks_upserted: int
    files_skipped: int
    errors: int
    details: List[str]


def _list_summ_files_in_folder(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.name.startswith("SUMM_") and p.suffix.lower() == ".md"]
    files.sort(key=lambda x: x.name.lower())
    return files


def _parse_summ_provenance(summ_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (source_file, source_sha256) from the SUMM header."""
    src_file = None
    src_sha = None
    m1 = re.search(r"(?im)^\s*Source-File:\s*(.+?)\s*$", summ_text or "")
    if m1:
        src_file = m1.group(1).strip()
    m2 = re.search(r"(?im)^\s*Source-SHA256:\s*([0-9a-f]{64})\s*$", summ_text or "")
    if m2:
        src_sha = m2.group(1).strip()
    return src_file, src_sha


def _uuid5(namespace: str, name: str) -> str:
    import uuid

    ns = uuid.uuid5(uuid.NAMESPACE_DNS, namespace)
    return str(uuid.uuid5(ns, name))


def move_to_vault(
    *,
    kb_names: Sequence[str],
    kb_paths: Dict[str, str],
    vault_kb_name: str = "vault",
    chunk_chars: int = 900,
    overlap_chars: int = 120,
) -> VaultMoveResult:
    """Index SUMM_*.md from KB folders into Qdrant (kb=vault_kb_name).

    Replace-by-file semantics: for each SUMM file, delete prior vault chunks for the same
    (source_sha256 OR summ filename), then upsert new chunks.
    """
    details: List[str] = []
    files_considered = files_indexed = chunks_upserted = files_skipped = errors = 0

    # Setup Qdrant collection
    try:
        client = get_qdrant_client()
        ensure_collection(client)
    except Exception as e:
        return VaultMoveResult(
            vault_kb_name=vault_kb_name,
            files_considered=0,
            files_indexed=0,
            chunks_upserted=0,
            files_skipped=0,
            errors=1,
            details=[f"[vault] ERROR init qdrant: {e}"],
        )

    embedder = None
    try:
        embedder = get_embed_model()
    except Exception as e:
        return VaultMoveResult(
            vault_kb_name=vault_kb_name,
            files_considered=0,
            files_indexed=0,
            chunks_upserted=0,
            files_skipped=0,
            errors=1,
            details=[f"[vault] ERROR load embed model: {e}"],
        )

    if qmodels is None:
        return VaultMoveResult(
            vault_kb_name=vault_kb_name,
            files_considered=0,
            files_indexed=0,
            chunks_upserted=0,
            files_skipped=0,
            errors=1,
            details=["[vault] ERROR qdrant_client.http.models not available"],
        )

    # Collect SUMM files
    summ_files: List[Tuple[str, Path]] = []
    for kb in kb_names:
        folder = kb_paths.get(kb)
        if not folder:
            details.append(f"[vault] WARN unknown kb: {kb}")
            continue
        for sf in _list_summ_files_in_folder(Path(folder)):
            summ_files.append((kb, sf))

    if not summ_files:
        return VaultMoveResult(
            vault_kb_name=vault_kb_name,
            files_considered=0,
            files_indexed=0,
            chunks_upserted=0,
            files_skipped=0,
            errors=0,
            details=["[vault] No SUMM_*.md files found in selected KB(s)."],
        )

    for kb, sf in summ_files:
        files_considered += 1
        try:
            txt = sf.read_text(encoding="utf-8", errors="replace")
            src_file, src_sha = _parse_summ_provenance(txt)
            body = strip_summ_frontmatter(txt).strip()
            if not body:
                files_skipped += 1
                details.append(f"[vault] SKIP empty SUMM body: {kb}/{sf.name}")
                continue

            # Delete prior chunks for this file (replace-by-file)
            # Filter: kb==vault_kb_name AND (source_sha256==src_sha OR summ_file==sf.name)
            must = [qmodels.FieldCondition(key="kb", match=qmodels.MatchValue(value=vault_kb_name))]
            should = []
            if src_sha:
                should.append(qmodels.FieldCondition(key="source_sha256", match=qmodels.MatchValue(value=src_sha)))
            should.append(qmodels.FieldCondition(key="summ_file", match=qmodels.MatchValue(value=sf.name)))

            flt = qmodels.Filter(must=must, should=should) if should else qmodels.Filter(must=must)
            try:
                client.delete(collection_name=QDRANT_COLLECTION, points_selector=flt, wait=True)
            except Exception:
                # Non-fatal: proceed to upsert anyway
                pass

            # Chunk + embed
            chunks = chunk_text_sliding(body, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
            if not chunks:
                files_skipped += 1
                details.append(f"[vault] SKIP no chunks: {kb}/{sf.name}")
                continue

            # E5 convention: "passage: ..."
            passages = [f"passage: {c}" for c in chunks]
            vecs = embedder.encode(passages, normalize_embeddings=True).tolist()

            points = []
            for i, (c, v) in enumerate(zip(chunks, vecs)):
                pid = _uuid5("moa.vault", f"{vault_kb_name}|{sf.name}|{src_sha or ''}|{i}")
                payload = {
                    "kb": vault_kb_name,
                    "from_kb": kb,
                    "summ_file": sf.name,
                    "source_file": src_file or "",
                    "source_sha256": src_sha or "",
                    "chunk_index": i,
                    "text": c,
                    "indexed_at": _now_iso(),
                }
                points.append(
                    qmodels.PointStruct(
                        id=pid,
                        vector={VECTOR_NAME: v},
                        payload=payload,
                    )
                )

            client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
            files_indexed += 1
            chunks_upserted += len(points)
            details.append(f"[vault] OK {kb}/{sf.name} -> {vault_kb_name} ({len(points)} chunks)")

        except Exception as e:
            errors += 1
            details.append(f"[vault] ERROR {kb}/{sf.name}: {e}")

    return VaultMoveResult(
        vault_kb_name=vault_kb_name,
        files_considered=files_considered,
        files_indexed=files_indexed,
        chunks_upserted=chunks_upserted,
        files_skipped=files_skipped,
        errors=errors,
        details=details,
    )
