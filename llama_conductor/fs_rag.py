# fs_rag.py
# version 1.0.4

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SUMM_FILE_RE = re.compile(r"\b(SUMM_[A-Za-z0-9._-]+\.md)\b", re.IGNORECASE)


@dataclass
class FsHit:
    score: float
    kb: str
    file: str
    rel_path: str
    snippet: str


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _norm_space(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _extract_target_summ_file(query: str) -> str:
    """Return explicitly requested SUMM filename from query, else empty string."""
    m = _SUMM_FILE_RE.search(query or "")
    if not m:
        return ""
    return m.group(1).strip().lower()


def _iter_summ_files(root: str) -> List[str]:
    """Return absolute paths of SUMM_*.md under root (excluding /original/)."""
    out: List[str] = []
    if not root or not os.path.isdir(root):
        return out

    for dirpath, _, filenames in os.walk(root):
        # Skip originals subfolder
        if "original" in {p.lower() for p in dirpath.split(os.sep)}:
            continue

        for fn in filenames:
            if not fn.lower().endswith(".md"):
                continue
            if not fn.startswith("SUMM_"):
                continue
            out.append(os.path.join(dirpath, fn))

    out.sort()
    return out


def _split_blocks(md_text: str) -> List[str]:
    """Split markdown into moderately sized blocks for scoring."""
    max_block = 1200
    overlap = 240

    def _window_text(s: str) -> List[str]:
        txt = (s or "").strip()
        if not txt:
            return []
        if len(txt) <= max_block:
            return [txt]
        out: List[str] = []
        step = max(1, max_block - overlap)
        i = 0
        while i < len(txt):
            chunk = txt[i : i + max_block].strip()
            if chunk:
                out.append(chunk)
            i += step
        return out

    text = (md_text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]

    blocks: List[str] = []
    for b in raw_blocks:
        if len(b) <= max_block:
            blocks.append(b)
            continue

        sub = [s.strip() for s in re.split(r"\n(?=#+\s)", b) if s.strip()]
        if not sub:
            blocks.extend(_window_text(b))
            continue

        for s in sub:
            blocks.extend(_window_text(s))

    return blocks


def _score_block(q_tokens: Set[str], block: str) -> float:
    if not q_tokens:
        return 0.0

    btoks = _tokenize(block)
    if not btoks:
        return 0.0

    bset = set(btoks)
    overlap = len(q_tokens & bset)

    score = overlap / max(1.0, float(len(q_tokens)))

    # Boost if query tokens appear in headings
    for ln in (block or "").splitlines()[:6]:
        ln_s = ln.strip()
        if ln_s.startswith("#"):
            ln_toks = set(_tokenize(ln_s))
            score += 0.12 * len(q_tokens & ln_toks)

    # Slight boost if block contains code fences (often operationally useful)
    if "```" in block:
        score += 0.05

    # Penalize ultra-short blocks
    if len(block) < 80:
        score *= 0.6

    return float(score)


def search_fs(
    query: str,
    attached_kbs: Set[str],
    kb_paths: Dict[str, str],
    *,
    top_k: int = 8,
    max_chars: int = 2400,
    max_blocks_per_file: int = 3,
) -> List[FsHit]:
    """Search attached KB folders for relevant SUMM_*.md blocks."""
    q = _norm_space(query)
    if not q or not attached_kbs:
        return []

    q_tokens = set(_tokenize(q))
    if not q_tokens:
        return []
    target_summ_file = _extract_target_summ_file(q)

    hits: List[FsHit] = []

    for kb in sorted(attached_kbs):
        root = kb_paths.get(kb)
        if not root:
            continue

        for fpath in _iter_summ_files(root):
            if target_summ_file and os.path.basename(fpath).lower() != target_summ_file:
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    md = f.read()
            except Exception:
                continue

            blocks = _split_blocks(md)
            scored: List[Tuple[float, str]] = []
            for b in blocks:
                s = _score_block(q_tokens, b)
                if s > 0:
                    scored.append((s, b))

            if not scored:
                continue

            scored.sort(key=lambda t: (-t[0], t[1][:40]))
            rel = os.path.relpath(fpath, root)
            fn = os.path.basename(fpath)

            for s, b in scored[: max(1, max_blocks_per_file)]:
                hits.append(
                    FsHit(
                        score=s,
                        kb=kb,
                        file=fn,
                        rel_path=rel,
                        snippet=b.strip(),
                    )
                )

    if not hits:
        return []

    hits.sort(key=lambda h: (-h.score, h.kb, h.file, h.rel_path))

    # Clip and respect max_chars
    out: List[FsHit] = []
    total = 0
    for h in hits:
        if len(out) >= max(1, top_k):
            break
        snippet = h.snippet.strip()
        if not snippet:
            continue

        overhead = 80  # provenance budget
        if max_chars > 0 and total + len(snippet) + overhead > max_chars:
            remaining = max_chars - total - overhead
            if remaining <= 0:
                break
            snippet = snippet[:remaining].rstrip()
            if not snippet:
                break

        out.append(
            FsHit(
                score=h.score,
                kb=h.kb,
                file=h.file,
                rel_path=h.rel_path,
                snippet=snippet,
            )
        )
        total += len(snippet) + overhead

    return out


def build_fs_facts_block(
    query: str,
    attached_kbs: Set[str],
    kb_paths: Dict[str, str],
    *,
    top_k: int = 8,
    max_chars: int = 2400,
) -> str:
    """Build a FACTS block string from filesystem KBs."""
    hits = search_fs(query, attached_kbs, kb_paths, top_k=top_k, max_chars=max_chars)
    if not hits:
        return ""

    parts: List[str] = []
    for h in hits:
        parts.append(f"- [kb={h.kb} file={h.file}] {h.snippet}")

    return "\n\n".join(parts)
