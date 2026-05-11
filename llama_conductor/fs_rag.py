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


def find_summ_file_matches(
    filename: str,
    attached_kbs: Set[str],
    kb_paths: Dict[str, str],
) -> List[Tuple[str, str, str, str]]:
    """Find SUMM filename matches in attached KBs.

    Returns list of tuples:
      (kb, abs_path, rel_path, actual_filename)
    """
    target = (filename or "").strip().lower()
    if not target:
        return []
    out: List[Tuple[str, str, str, str]] = []
    for kb in sorted(attached_kbs):
        root = kb_paths.get(kb)
        if not root or not os.path.isdir(root):
            continue
        for fpath in _iter_summ_files(root):
            fn = os.path.basename(fpath)
            if fn.lower() != target:
                continue
            rel = os.path.relpath(fpath, root)
            out.append((kb, fpath, rel, fn))
    return out


def find_summ_file_candidates(
    term: str,
    attached_kbs: Set[str],
    kb_paths: Dict[str, str],
) -> List[Tuple[str, str, str, str]]:
    """Find candidate SUMM files by partial filename match.

    Matching is case-insensitive and checks:
    - full filename (with .md),
    - filename stem,
    - filename stem with optional leading 'SUMM_' removed.
    """
    q = (term or "").strip().lower()
    if not q:
        return []
    if q.endswith(".md"):
        q = q[:-3]
    if q.startswith("summ_"):
        q = q[5:]
    q = q.strip()
    if not q:
        return []

    out: List[Tuple[str, str, str, str]] = []
    for kb in sorted(attached_kbs):
        root = kb_paths.get(kb)
        if not root or not os.path.isdir(root):
            continue
        for fpath in _iter_summ_files(root):
            fn = os.path.basename(fpath)
            fn_l = fn.lower()
            stem = fn_l[:-3] if fn_l.endswith(".md") else fn_l
            stem_no_summ = stem[5:] if stem.startswith("summ_") else stem
            if q in fn_l or q in stem or q in stem_no_summ:
                rel = os.path.relpath(fpath, root)
                out.append((kb, fpath, rel, fn))
    out.sort(key=lambda t: (t[0], t[3].lower(), t[2].lower()))
    return out


def _strip_summ_comment_header(md_text: str) -> str:
    txt = (md_text or "").replace("\r\n", "\n").replace("\r", "\n").lstrip()
    if not txt.startswith("<!--"):
        return txt
    end = txt.find("-->")
    if end == -1:
        return txt
    return txt[end + 3 :].lstrip()


def _extract_summ_sentences(md_text: str) -> List[str]:
    """Extract bullet sentences from '## Extracted Sentences' section."""
    txt = _strip_summ_comment_header(md_text)
    lines = txt.splitlines()
    in_section = False
    out: List[str] = []
    for raw in lines:
        ln = (raw or "").strip()
        if not in_section:
            if ln.lower().startswith("## extracted sentences"):
                in_section = True
            continue
        if ln.startswith("## "):
            break
        if ln.startswith("- "):
            s = ln[2:].strip()
            if s:
                out.append(s)
    return out


def build_locked_summ_facts_block(
    *,
    query: str,
    kb: str,
    file: str,
    rel_path: str,
    abs_path: str,
    max_chars: int = 12000,
) -> Tuple[str, int]:
    """Build deterministic facts block from one locked SUMM file.

    Returns: (facts_block, fact_line_count)
    """
    del query  # lock mode is deterministic by file scope; query is not used for ranking.

    if not abs_path or not os.path.isfile(abs_path):
        return ("", 0)

    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            md = f.read()
    except Exception:
        return ("", 0)

    sentences = _extract_summ_sentences(md)
    if not sentences:
        txt = _strip_summ_comment_header(md).strip()
        if not txt:
            return ("", 0)
        line = f"- [kb={kb} file={file}] {txt}"
        if max_chars > 0 and len(line) > max_chars:
            line = line[:max_chars].rstrip()
        return (line, 1 if line else 0)

    parts: List[str] = []
    used = 0
    count = 0
    for s in sentences:
        line = f"- [kb={kb} file={file}] {s}"
        if max_chars > 0 and used + len(line) + 2 > max_chars:
            break
        parts.append(line)
        used += len(line) + 2
        count += 1

    if not parts:
        return ("", 0)
    return ("\n\n".join(parts), count)


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
