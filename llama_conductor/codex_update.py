#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


CONTRADICTION_PREFIX = "**[CONTRADICTION]**"
RESOLVED_PREFIX = "**[RESOLVED]**"
CONFLICT_WARNING = "[Codex: contradiction marker present on this page. Content shown; review recommended.]"
INDEX_SECTIONS = ("Topics", "Entities", "Concepts", "Sources")
PAGE_DIRS = ("topics", "entities", "concepts", "sources")
_HTML_COMMENT_BLOCK_RE = re.compile(r"<!--.*?-->", re.S)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-]", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def slugify(value: str) -> str:
    slug = normalize_key(value)
    return slug or "untitled"


def first_sentence(text: str) -> str:
    t = " ".join((text or "").strip().split())
    if not t:
        return ""
    m = re.search(r"\.(?=\s|$)", t)
    if m:
        return t[: m.end()].strip()[:120]
    return t[:120]


def sanitize_summ_ingest_text(text: str) -> str:
    """Strip HTML comments from SUMM ingest text, including unclosed comment tails."""
    t = str(text or "")
    # Remove properly closed HTML comments first.
    t = _HTML_COMMENT_BLOCK_RE.sub(" ", t)
    # Remove dangling unclosed comments to EOF.
    while True:
        start = t.find("<!--")
        if start < 0:
            break
        t = t[:start]
    # Normalize whitespace after removals.
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text_utf8(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8", newline="\n")


def atomic_write_text_utf8(path: Path, text: str) -> None:
    ensure_parent(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def append_text_utf8(path: Path, text: str) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(text)


def delete_path(path: Path) -> None:
    if path.exists():
        path.unlink()


def clean_marker_line(line: str) -> str:
    trimmed = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line.strip())
    return trimmed


def is_unresolved_contradiction(line: str) -> bool:
    return clean_marker_line(line).startswith(CONTRADICTION_PREFIX)


def is_resolved_contradiction(line: str) -> bool:
    return clean_marker_line(line).startswith(RESOLVED_PREFIX)


def page_conflict_stale(text: str) -> bool:
    for line in (text or "").splitlines():
        if is_unresolved_contradiction(line):
            return True
    return False


def parse_headings(markdown: str) -> List[str]:
    out: List[str] = []
    for line in (markdown or "").splitlines():
        m = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if m:
            out.append(m.group(1).strip())
    return out


def parse_first_heading(markdown: str) -> Optional[str]:
    hs = parse_headings(markdown)
    return hs[0] if hs else None


def extract_original_ref(summ_path: Path) -> str:
    original_dir = summ_path.parent / "original"
    if original_dir.exists() and original_dir.is_dir():
        files = sorted([p for p in original_dir.iterdir() if p.is_file()])
        if files:
            return str(files[0]).replace("\\", "/")
    return str((summ_path.parent / "original" / "UNKNOWN_ORIGINAL").as_posix())


@dataclass
class IndexEntry:
    section: str
    title: str
    rel_path: str
    description: str


def parse_index(index_text: str) -> List[IndexEntry]:
    entries: List[IndexEntry] = []
    current_section = "Topics"
    bullet_re = re.compile(r"^\s*-\s*\[([^\]]+)\]\(([^)]+)\)\s*(?:[-—]\s*(.*))?$")
    for line in (index_text or "").splitlines():
        sec = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if sec:
            current_section = sec.group(1).strip()
            continue
        b = bullet_re.match(line)
        if b:
            entries.append(
                IndexEntry(
                    section=current_section,
                    title=b.group(1).strip(),
                    rel_path=b.group(2).strip(),
                    description=(b.group(3) or "").strip(),
                )
            )
    return entries


def render_index(entries: Sequence[IndexEntry]) -> str:
    grouped: Dict[str, List[IndexEntry]] = {k: [] for k in INDEX_SECTIONS}
    for e in entries:
        sec = e.section if e.section in grouped else "Topics"
        grouped[sec].append(e)
    for sec in grouped:
        grouped[sec] = sorted(grouped[sec], key=lambda x: (normalize_key(x.title), x.rel_path))
    lines: List[str] = ["# Codex Index", ""]
    for sec in INDEX_SECTIONS:
        if sec == "Sources" and not grouped[sec]:
            continue
        lines.append(f"## {sec}")
        for e in grouped[sec]:
            desc = f" — {e.description}" if e.description else ""
            lines.append(f"- [{e.title}]({e.rel_path}){desc}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def page_title(text: str, fallback: str) -> str:
    m = re.search(r"^\s*#\s+(.+?)\s*$", text or "", flags=re.M)
    if m:
        return m.group(1).strip()
    return fallback


def page_summary(text: str, fallback_title: str) -> str:
    m = re.search(r"^\s*\*\*Summary\*\*:\s*(.+?)\s*$", text or "", flags=re.M)
    if m:
        return m.group(1).strip()
    return fallback_title


def section_for_relpath(rel_path: str) -> str:
    p = rel_path.replace("\\", "/").lower()
    if p.startswith("entities/"):
        return "Entities"
    if p.startswith("concepts/"):
        return "Concepts"
    if p.startswith("sources/"):
        return "Sources"
    return "Topics"


def normal_usable_page(page_path: Path, indexed: bool) -> Tuple[bool, str]:
    if not indexed:
        return False, "not_indexed"
    if not page_path.exists():
        return False, "missing_path"
    try:
        text = read_text_utf8(page_path)
    except Exception:
        return False, "unreadable_text"
    if not re.search(r"^\s*#\s+.+$", text, flags=re.M):
        return False, "missing_title"
    if "SUMM_" not in text:
        return False, "missing_summ_ref"
    if not re.search(r"original", text, flags=re.I):
        return False, "missing_original_ref"
    return True, "ok"


def rebuild_usable_page(page_path: Path) -> Tuple[bool, str]:
    if not page_path.exists():
        return False, "missing_path"
    try:
        text = read_text_utf8(page_path)
    except Exception:
        return False, "unreadable_text"
    if not re.search(r"^\s*#\s+.+$", text, flags=re.M):
        return False, "missing_title"
    if "SUMM_" not in text:
        return False, "missing_summ_ref"
    if not re.search(r"original", text, flags=re.I):
        return False, "missing_original_ref"
    return True, "ok"


def collect_page_paths(codex_root: Path) -> List[Path]:
    pages: List[Path] = []
    for folder in PAGE_DIRS:
        d = codex_root / folder
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.md")):
            pages.append(p)
    return pages


def build_identity_targets(codex_root: Path, entries: Sequence[IndexEntry]) -> Dict[str, Set[str]]:
    targets: Dict[str, Set[str]] = {}
    for e in entries:
        key = normalize_key(e.title)
        targets.setdefault(key, set()).add(e.rel_path)
        stem = normalize_key(Path(e.rel_path).stem)
        targets.setdefault(stem, set()).add(e.rel_path)
    for p in collect_page_paths(codex_root):
        rel = p.relative_to(codex_root).as_posix()
        stem = normalize_key(p.stem)
        targets.setdefault(stem, set()).add(rel)
    return targets


def extract_candidate_terms(summ_path: Path, summ_text: str, kb_name: Optional[str]) -> List[str]:
    terms: List[str] = []
    if kb_name:
        terms.append(kb_name)
    if summ_path.parent.name:
        terms.append(summ_path.parent.name)
    terms.append(summ_path.stem)
    first = parse_first_heading(summ_text)
    if first:
        terms.append(first)
    terms.extend(parse_headings(summ_text))
    return [t for t in terms if t and t.strip()]


def choose_fallback_title(summ_path: Path, summ_text: str) -> str:
    headings = parse_headings(summ_text)
    first = headings[0] if headings else None
    meta_title = ""
    if first:
        h = str(first).strip()
        m = re.match(r"^\s*sum[m]?\s*:\s*(.+?)\s*$", h, flags=re.I)
        if m:
            meta_title = str(m.group(1) or "").strip()
        else:
            h = re.sub(r"\.md\s*$", "", h, flags=re.I).strip()
            if h:
                return h

    # If first heading is SUMM meta-header, prefer a meaningful subject heading below it.
    if meta_title:
        skip = {
            "extracted sentences",
            "source refs",
            "source references",
            "main content",
            "contradictions / open questions",
            "contradictions",
            "open questions",
        }
        for h in headings[1:]:
            hs = str(h or "").strip()
            if not hs:
                continue
            hs_norm = re.sub(r"\s+", " ", hs).strip().lower()
            if hs_norm in skip:
                continue
            if hs_norm.startswith("summ:"):
                continue
            hs_clean = re.sub(r"\.md\s*$", "", hs, flags=re.I).strip()
            if hs_clean:
                return hs_clean

        # No meaningful heading present. Fall back to first extracted sentence subject.
        for line in (summ_text or "").splitlines():
            s = str(line or "").strip()
            if not s.startswith("- "):
                continue
            sentence = s[2:].strip()
            m_subj = re.match(
                r"^(?:The|An?|This|That)\s+([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3})\b",
                sentence,
            )
            if m_subj:
                cand = str(m_subj.group(1) or "").strip()
                if cand:
                    return cand
            m_is = re.match(r"^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3})\s+is\b", sentence)
            if m_is:
                cand = str(m_is.group(1) or "").strip()
                if cand:
                    return cand

        meta_title = re.sub(r"\.md\s*$", "", meta_title, flags=re.I).strip()
        if meta_title:
            return meta_title
    stem = summ_path.stem
    stem = re.sub(r"^SUMM[_\-]?", "", stem, flags=re.I)
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem.title() or "Untitled"


def ensure_codex_bootstrap(codex_root: Path, dry_run: bool) -> None:
    if dry_run:
        return
    codex_root.mkdir(parents=True, exist_ok=True)
    for folder in ("topics", "entities", "concepts"):
        (codex_root / folder).mkdir(parents=True, exist_ok=True)
    idx = codex_root / "index.md"
    if not idx.exists():
        write_text_utf8(idx, render_index([]))
    log = codex_root / "log.md"
    if not log.exists():
        write_text_utf8(log, "# Codex Log\n\n")


def load_index_entries(index_path: Path) -> List[IndexEntry]:
    text = read_text_utf8(index_path)
    return parse_index(text)


def page_rel_to_section_slug(title: str, codex_root: Path) -> Tuple[str, bool]:
    slug = slugify(title)
    rel = Path("topics") / f"{slug}.md"
    candidate = codex_root / rel
    if not candidate.exists():
        return rel.as_posix(), False
    i = 1
    while True:
        rel_i = Path("topics") / f"{slug}-{i}.md"
        if not (codex_root / rel_i).exists():
            return rel_i.as_posix(), True
        i += 1


def parse_source_refs_block(summ_path: Path, summ_text: str) -> Tuple[str, str]:
    summ_ref = summ_path.name
    original = extract_original_ref(summ_path)
    return summ_ref, original


def ensure_base_page(
    page_path: Path,
    page_title_text: str,
    dry_run: bool,
    summ_ref: str,
    original_ref: str,
) -> None:
    if dry_run:
        return
    if page_path.exists():
        return
    text = (
        f"# {page_title_text}\n\n"
        f"**Summary**: {page_title_text}\n\n"
        f"**Derived from**:\n"
        f"- {summ_ref}\n"
        f"- {original_ref}\n\n"
        f"**Last updated**: {utc_date()}\n\n"
        f"**Related pages**:\n\n"
        f"---\n\n"
        f"## Main content\n\n"
        f"## Contradictions / open questions\n\n"
        f"## Source refs\n\n"
    )
    atomic_write_text_utf8(page_path, text)


def split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    current = "__preamble__"
    buf: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if m:
            sections[current] = "\n".join(buf).strip("\n")
            current = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip("\n")
    return sections


def compose_sections(sections: Dict[str, str]) -> str:
    order = ["__preamble__", "Main content", "Contradictions / open questions", "Source refs"]
    lines: List[str] = []
    if sections.get("__preamble__", ""):
        lines.append(sections["__preamble__"].rstrip())
        lines.append("")
    for sec in order[1:]:
        lines.append(f"## {sec}")
        lines.append("")
        body = sections.get(sec, "").strip("\n")
        if body:
            lines.append(body.rstrip())
            lines.append("")
    for sec, body in sections.items():
        if sec in order:
            continue
        lines.append(f"## {sec}")
        lines.append("")
        if body:
            lines.append(body.rstrip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_SUMM_SUBSECTION_HEADING_RE = re.compile(r"^\s*###\s+From\s+(SUMM_[^\s]+\.md)\s*$", re.I)


def _stale_stems_from_refs(stale_summ_refs: Set[str]) -> Set[str]:
    """Extract base stems from stale SUMM refs for paired original-path matching.

    E.g. ``SUMM_drop-fixture.md`` → ``drop-fixture``.
    """
    stems: Set[str] = set()
    for ref in stale_summ_refs:
        name = Path(ref).stem                       # e.g. SUMM_drop-fixture
        stem = re.sub(r"^SUMM[_\-]?", "", name, flags=re.I)  # e.g. drop-fixture
        if stem:
            stems.add(stem.lower())
    return stems


def _strip_matching_refs_block(text: str, stale_summ_refs: Set[str], *, header_pattern: str) -> str:
    if not stale_summ_refs:
        return text
    stale_stems = _stale_stems_from_refs(stale_summ_refs)
    lines = (text or "").splitlines()
    out: List[str] = []
    in_block = False
    for line in lines:
        if re.match(header_pattern, line, flags=re.I):
            in_block = True
            out.append(line)
            continue
        if in_block:
            stripped = str(line or "").strip()
            if stripped.startswith("- "):
                ref = stripped[2:].strip()
                if ref in stale_summ_refs:
                    continue
                if Path(ref).stem.lower() in stale_stems:
                    continue
                out.append(line)
                continue
            if stripped == "":
                out.append(line)
                continue
            in_block = False
        out.append(line)
    return "\n".join(out).strip("\n")


def _strip_matching_bullet_refs(text: str, stale_summ_refs: Set[str]) -> str:
    if not stale_summ_refs:
        return text
    stale_stems = _stale_stems_from_refs(stale_summ_refs)
    out: List[str] = []
    for line in (text or "").splitlines():
        stripped = str(line or "").strip()
        if stripped.startswith("- "):
            ref = stripped[2:].strip()
            if ref in stale_summ_refs:
                continue
            if Path(ref).stem.lower() in stale_stems:
                continue
        out.append(line)
    return "\n".join(out).strip("\n")


def _remove_stale_subsections(main_body: str, *, install_root: Path) -> Tuple[str, List[str], int]:
    lines = (main_body or "").splitlines()
    blocks: List[Tuple[str, List[str]]] = []
    current_ref = ""
    current_lines: List[str] = []
    for line in lines:
        m = _SUMM_SUBSECTION_HEADING_RE.match(str(line or ""))
        if m:
            if current_lines:
                blocks.append((current_ref, current_lines))
            current_ref = str(m.group(1) or "").strip()
            current_lines = [line]
            continue
        if current_lines:
            current_lines.append(line)
        else:
            blocks.append(("", [line]))
    if current_lines:
        blocks.append((current_ref, current_lines))

    kept_lines: List[str] = []
    stale_refs: List[str] = []
    removed_count = 0
    for summ_ref, block_lines in blocks:
        if not summ_ref:
            kept_lines.extend(block_lines)
            continue
        ref_path = _resolve_ref_path(summ_ref, install_root=install_root)
        if ref_path is None:
            stale_refs.append(summ_ref)
            removed_count += 1
            continue
        kept_lines.extend(block_lines)
    return "\n".join(kept_lines).strip("\n"), stale_refs, removed_count


def _remaining_summ_subsection_refs(main_body: str) -> List[str]:
    refs: List[str] = []
    for line in (main_body or "").splitlines():
        m = _SUMM_SUBSECTION_HEADING_RE.match(str(line or ""))
        if m:
            refs.append(str(m.group(1) or "").strip())
    return refs


def add_or_update_related_pages(sections: Dict[str, str], links: Sequence[str]) -> None:
    existing: List[str] = []
    body = sections.get("__preamble__", "")
    rel_pat = re.compile(r"^\s*-\s*\[\[([^\]]+)\]\]\s*$")
    out_lines: List[str] = []
    in_related = False
    found_related_header = False
    for line in body.splitlines():
        if re.match(r"^\s*\*\*Related pages\*\*:\s*$", line):
            in_related = True
            found_related_header = True
            out_lines.append(line)
            continue
        if in_related:
            m = rel_pat.match(line)
            if m:
                existing.append(m.group(1).strip())
                out_lines.append(line)
                continue
            if line.strip() == "":
                out_lines.append(line)
                continue
            in_related = False
        out_lines.append(line)
    existing_set = {e for e in existing if e}
    if not found_related_header:
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        out_lines.append("**Related pages**:")
    for l in links:
        if l not in existing_set:
            out_lines.append(f"- [[{l}]]")
            existing_set.add(l)
    sections["__preamble__"] = "\n".join(out_lines).strip("\n")


def update_last_updated_preamble(preamble: str) -> str:
    line = f"**Last updated**: {utc_date()}"
    if re.search(r"^\s*\*\*Last updated\*\*:\s*.+$", preamble, flags=re.M):
        return re.sub(r"^\s*\*\*Last updated\*\*:\s*.+$", line, preamble, flags=re.M)
    if preamble.strip():
        return preamble.rstrip() + "\n\n" + line
    return line


def append_subsection_and_refs(
    page_path: Path,
    summ_path: Path,
    summ_text: str,
    dry_run: bool,
) -> Tuple[str, str, List[str]]:
    def _extract_relevant_summ_body(raw: str, *, max_chars: int = 800) -> str:
        txt = str(raw or "").replace("\ufeff", "").strip()
        if not txt:
            return ""
        lines = txt.splitlines()

        # Prefer explicit extracted content section from SUMM files.
        start = -1
        for i, ln in enumerate(lines):
            if re.match(r"^\s*##\s+Extracted Sentences\s*$", str(ln or ""), flags=re.I):
                start = i + 1
                break
        if start >= 0:
            body_lines: List[str] = []
            for ln in lines[start:]:
                if re.match(r"^\s*##\s+", str(ln or "")):
                    break
                body_lines.append(ln.rstrip())
            body = "\n".join(body_lines).strip()
        else:
            # Fallback: strip SUMM meta-header lines and keep body.
            kept: List[str] = []
            for ln in lines:
                s = str(ln or "").strip()
                if re.match(r"^\s*#\s*SUMM\s*:\s*", s, flags=re.I):
                    continue
                if re.match(r"^\s*method\s*:\s*", s, flags=re.I):
                    continue
                if re.match(r"^\s*##\s*Source refs\s*$", s, flags=re.I):
                    break
                kept.append(ln.rstrip())
            body = "\n".join(kept).strip()

        if not body:
            return ""
        if len(body) > max_chars:
            cut = body[:max_chars]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            body = cut.rstrip() + "..."
        return body

    original_text = read_text_utf8(page_path)
    sections = split_sections(original_text)
    sections.setdefault("Main content", "")
    sections.setdefault("Contradictions / open questions", "")
    sections.setdefault("Source refs", "")
    summ_ref, original_ref = parse_source_refs_block(summ_path, summ_text)
    digest = _extract_relevant_summ_body(summ_text, max_chars=800) or "Imported update."
    heading = f"### From {summ_ref}"
    new_block = (
        f"{heading}\n"
        f"{digest}\n\n"
        f"Source refs (this update):\n"
        f"- {summ_ref}\n"
        f"- {original_ref}\n"
    )
    main = sections.get("Main content", "").strip("\n")
    if main:
        main = main.rstrip() + "\n\n" + new_block
    else:
        main = new_block
    sections["Main content"] = main.strip("\n")

    contradictions_added = 0
    contradiction_lines = []
    for line in summ_text.splitlines():
        cleaned = clean_marker_line(line)
        if cleaned.startswith(CONTRADICTION_PREFIX) or cleaned.startswith(RESOLVED_PREFIX):
            contradiction_lines.append(f"- {cleaned}")
            if cleaned.startswith(CONTRADICTION_PREFIX):
                contradictions_added += 1
    if contradiction_lines:
        cbody = sections.get("Contradictions / open questions", "").strip("\n")
        add = "\n".join(contradiction_lines)
        sections["Contradictions / open questions"] = (cbody + "\n" + add).strip("\n") if cbody else add

    sbody = sections.get("Source refs", "").strip("\n")
    refs_to_add = [f"- {summ_ref}", f"- {original_ref}"]
    existing_refs = set(s.strip() for s in sbody.splitlines() if s.strip().startswith("- "))
    for ref in refs_to_add:
        if ref not in existing_refs:
            sbody = (sbody + "\n" + ref).strip("\n") if sbody else ref
    sections["Source refs"] = sbody
    sections["__preamble__"] = update_last_updated_preamble(sections.get("__preamble__", ""))
    updated_text = compose_sections(sections)
    if not dry_run:
        write_text_utf8(page_path, updated_text)
    summ_headings = parse_headings(summ_text)
    return updated_text, new_block, [heading] + summ_headings


def extract_crosslink_candidates_from_block(block_text: str, headings: Sequence[str]) -> Set[str]:
    candidates: Set[str] = set()
    for h in headings:
        candidates.add(normalize_key(h))
    for tok in re.split(r"\s+", block_text or ""):
        cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", tok).strip()
        if len(cleaned) < 3:
            continue
        candidates.add(normalize_key(cleaned))
    return {c for c in candidates if c}


def compute_crosslinks(
    codex_root: Path,
    page_rel: str,
    new_block_text: str,
    new_headings: Sequence[str],
    index_entries: Sequence[IndexEntry],
) -> List[str]:
    targets = build_identity_targets(codex_root, index_entries)
    page_norm_stem = normalize_key(Path(page_rel).stem)
    out: Set[str] = set()
    candidates = extract_crosslink_candidates_from_block(new_block_text, new_headings)
    for c in candidates:
        for rel in sorted(targets.get(c, set())):
            if normalize_key(Path(rel).stem) == page_norm_stem:
                continue
            if not (codex_root / rel).exists():
                continue
            out.add(Path(rel).stem)
    return sorted(out)


def update_index_for_pages(
    codex_root: Path,
    index_entries: List[IndexEntry],
    touched_rel_paths: Sequence[str],
    dry_run: bool,
) -> Tuple[int, int]:
    by_rel: Dict[str, IndexEntry] = {e.rel_path: e for e in index_entries}
    added = 0
    updated = 0
    for rel in touched_rel_paths:
        p = codex_root / rel
        if not p.exists():
            continue
        text = read_text_utf8(p)
        title = page_title(text, Path(rel).stem.replace("-", " ").title())
        summary = page_summary(text, title)
        desc = first_sentence(summary)
        sec = section_for_relpath(rel)
        if rel in by_rel:
            e = by_rel[rel]
            if e.title != title or e.description != desc or e.section != sec:
                e.title = title
                e.description = desc
                e.section = sec
                updated += 1
        else:
            index_entries.append(IndexEntry(sec, title, rel, desc))
            added += 1
    if not dry_run:
        idx_path = codex_root / "index.md"
        write_text_utf8(idx_path, render_index(index_entries))
    return added, updated


def append_log_entry(
    log_path: Path,
    summ_path: Path,
    original_ref: str,
    pages_updated: Sequence[str],
    notes: Sequence[str],
    dry_run: bool,
) -> None:
    lines = [f"## {utc_now()}", f"- SUMM: {summ_path.as_posix()}", f"- Original: {original_ref}", "- Updated pages:"]
    if pages_updated:
        for p in pages_updated:
            lines.append(f"  - {p}")
    else:
        lines.append("  - (none)")
    if notes:
        lines.append("- Notes:")
        for n in notes:
            lines.append(f"  - {n}")
    lines.append("")
    entry = "\n".join(lines)
    if not dry_run:
        append_text_utf8(log_path, entry)


def make_telemetry_path(repo_root: Path, summ_stem: str) -> Path:
    telemetry_dir = repo_root / "total_recall" / "codex"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return telemetry_dir / f"codex_update_{slugify(summ_stem)}_{stamp}.json"


def write_telemetry(repo_root: Path, summ_stem: str, payload: Dict[str, object]) -> Path:
    path = make_telemetry_path(repo_root, summ_stem)
    write_text_utf8(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def command_rebuild(codex_root: Path, dry_run: bool = False) -> Dict[str, object]:
    ensure_codex_bootstrap(codex_root, dry_run=dry_run)
    skipped: List[str] = []
    valid_entries: List[IndexEntry] = []
    for page in collect_page_paths(codex_root):
        ok, reason = rebuild_usable_page(page)
        rel = page.relative_to(codex_root).as_posix()
        if not ok:
            skipped.append(f"{rel} :: {reason}")
            continue
        text = read_text_utf8(page)
        title = page_title(text, page.stem.replace("-", " ").title())
        desc = first_sentence(page_summary(text, title))
        valid_entries.append(IndexEntry(section_for_relpath(rel), title, rel, desc))
    if not dry_run:
        write_text_utf8(codex_root / "index.md", render_index(valid_entries))
        if skipped:
            log_path = codex_root / "log.md"
            ensure_parent(log_path)
            if not log_path.exists():
                write_text_utf8(log_path, "# Codex Log\n\n")
            for item in skipped:
                append_text_utf8(
                    log_path,
                    f"## {utc_now()}\n- Skipped: {item.split(' :: ')[0]}\n- Reason: {item.split(' :: ')[1]}\n\n",
                )
    entry_keys = [f"{e.section}:{e.title}:{e.rel_path}" for e in valid_entries]
    return {"rebuilt_entries": len(valid_entries), "skipped": skipped, "entry_keys": entry_keys}


def command_doctor(codex_root: Path) -> Dict[str, object]:
    findings: List[str] = []
    idx = codex_root / "index.md"
    if not idx.exists():
        findings.append("missing:index.md")
        return {"findings": findings}
    try:
        index_text = read_text_utf8(idx)
    except Exception:
        findings.append("unreadable:index.md")
        return {"findings": findings}
    entries = parse_index(index_text)
    for e in entries:
        page = codex_root / e.rel_path
        if not page.exists():
            findings.append(f"dead_ref:{e.rel_path}")
            continue
        ok, reason = normal_usable_page(page, indexed=True)
        if not ok:
            findings.append(f"unusable:{e.rel_path}:{reason}")
    for page in collect_page_paths(codex_root):
        rel = page.relative_to(codex_root).as_posix()
        if not any(x.rel_path == rel for x in entries):
            findings.append(f"orphan:{rel}")
    return {"findings": findings}


def _parse_last_updated_date(text: str) -> Optional[date]:
    m = re.search(r"^\s*\*\*Last updated\*\*:\s*(\d{4}-\d{2}-\d{2})\s*$", str(text or ""), flags=re.M)
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(1))
    except Exception:
        return None


def _extract_source_ref_lines(text: str) -> List[str]:
    refs: List[str] = []
    for line in str(text or "").splitlines():
        s = str(line or "").strip()
        if s.startswith("- "):
            refs.append(s[2:].strip())
    return refs


def _extract_wikilinks(text: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"\[\[([^\]]+)\]\]", str(text or "")):
        t = str(m.group(1) or "").strip()
        if t:
            out.append(t)
    return out


def _resolve_ref_path(ref: str, *, install_root: Path) -> Optional[Path]:
    r = str(ref or "").strip()
    if not r:
        return None
    p = Path(r)
    if p.is_absolute():
        return p if p.exists() else None
    rel = (install_root / r).resolve()
    if rel.exists():
        return rel
    # Basename SUMM refs are usually under docz/kb/**.
    if "/" not in r and "\\" not in r:
        try:
            for hit in (install_root / "docz" / "kb").rglob(r):
                if hit.exists():
                    return hit
        except Exception:
            return None
    return None


def command_lint(codex_root: Path) -> Dict[str, object]:
    findings: List[str] = []
    install_root = codex_root.parent.resolve()
    idx = codex_root / "index.md"
    if not idx.exists():
        findings.append("LINT:schema:index.md:missing_index")
        findings.append(f"LINT:summary:all:count={len(findings)}")
        return {"findings": findings}
    try:
        entries = parse_index(read_text_utf8(idx))
    except Exception:
        findings.append("LINT:schema:index.md:unreadable_index")
        findings.append(f"LINT:summary:all:count={len(findings)}")
        return {"findings": findings}

    indexed_rel: Set[str] = set(e.rel_path for e in entries)
    pages = collect_page_paths(codex_root)
    page_texts: Dict[str, str] = {}
    stem_to_rel: Dict[str, str] = {}
    links_by_rel: Dict[str, Set[str]] = {}

    for p in pages:
        rel = p.relative_to(codex_root).as_posix()
        try:
            txt = read_text_utf8(p)
        except Exception:
            findings.append(f"LINT:schema:{rel}:unreadable_page")
            continue
        page_texts[rel] = txt
        stem_to_rel[normalize_key(Path(rel).stem)] = rel
        links_by_rel[rel] = {normalize_key(x) for x in _extract_wikilinks(txt)}

    # 1) Orphan pages
    for rel in sorted(page_texts.keys()):
        if rel not in indexed_rel:
            findings.append(f"LINT:orphan:{rel}:not_in_index")

    # 2) Missing backlinks
    for rel_a in sorted(page_texts.keys()):
        out_targets = links_by_rel.get(rel_a, set())
        stem_a = normalize_key(Path(rel_a).stem)
        for tgt in sorted(out_targets):
            rel_b = stem_to_rel.get(tgt)
            if not rel_b or rel_b == rel_a:
                continue
            if stem_a not in links_by_rel.get(rel_b, set()):
                findings.append(f"LINT:missing_backlink:{rel_a}:to={rel_b}")

    # 3) Contradictions (unresolved markers)
    for rel, txt in sorted(page_texts.items()):
        if any(is_unresolved_contradiction(ln) for ln in txt.splitlines()):
            findings.append(f"LINT:contradiction:{rel}:unresolved_marker")

    # 4) Stale pages (timestamp-stale per 17.1)
    for rel, txt in sorted(page_texts.items()):
        lu = _parse_last_updated_date(txt)
        if lu is None:
            continue
        stale_details: List[str] = []
        seen_paths: Set[str] = set()
        for ref in _extract_source_ref_lines(txt):
            low = ref.lower()
            ref_path = _resolve_ref_path(ref, install_root=install_root)
            if ref_path is None:
                continue
            ref_norm = ref_path.as_posix()
            if ref_norm in seen_paths:
                continue
            if ("summ_" in low) or ("original/" in low) or ("original\\" in low):
                try:
                    mdate = datetime.fromtimestamp(ref_path.stat().st_mtime, timezone.utc).date()
                except Exception:
                    continue
                if mdate > lu:
                    stale_details.append(f"{ref_norm}>{lu.isoformat()}")
                    seen_paths.add(ref_norm)
        if stale_details:
            findings.append(f"LINT:stale:{rel}:{'; '.join(stale_details[:3])}")

    # 5) Uncited claims (no SUMM source ref)
    for rel, txt in sorted(page_texts.items()):
        if "SUMM_" not in txt:
            findings.append(f"LINT:uncited_claim:{rel}:missing_summ_ref")

    # 6) Schema violations (title, derived-from, last-updated)
    for rel, txt in sorted(page_texts.items()):
        miss: List[str] = []
        if not re.search(r"^\s*#\s+.+$", txt, flags=re.M):
            miss.append("title")
        if not re.search(r"^\s*\*\*Derived from\*\*:\s*$", txt, flags=re.M):
            miss.append("derived-from")
        if not re.search(r"^\s*\*\*Last updated\*\*:\s*\d{4}-\d{2}-\d{2}\s*$", txt, flags=re.M):
            miss.append("last-updated")
        if miss:
            findings.append(f"LINT:schema:{rel}:missing_{','.join(miss)}")

    # 7) Missing original refs
    for rel, txt in sorted(page_texts.items()):
        has_original = False
        for ref in _extract_source_ref_lines(txt):
            low = ref.lower()
            if ("original/" in low) or ("original\\" in low):
                has_original = True
                break
        if not has_original:
            findings.append(f"LINT:missing_original_ref:{rel}:no_archived_original_link")

    findings.append(f"LINT:summary:all:count={len(findings)}")
    return {"findings": findings}


def command_clean(codex_root: Path, dry_run: bool = False) -> Dict[str, object]:
    ensure_codex_bootstrap(codex_root, dry_run=dry_run)
    install_root = codex_root.parent.resolve()
    log_path = codex_root / "log.md"
    index_path = codex_root / "index.md"

    pre_entries: List[IndexEntry] = []
    if index_path.exists():
        try:
            pre_entries = parse_index(read_text_utf8(index_path))
        except Exception:
            pre_entries = []

    subsections_removed_by_page: Dict[str, int] = {}
    pages_deleted: List[str] = []
    pages_mutated: List[str] = []

    for page in collect_page_paths(codex_root):
        rel = page.relative_to(codex_root).as_posix()
        try:
            original_text = read_text_utf8(page)
        except Exception:
            continue

        sections = split_sections(original_text)
        main_before = sections.get("Main content", "").strip("\n")
        main_after, stale_refs, removed_count = _remove_stale_subsections(main_before, install_root=install_root)
        if removed_count <= 0:
            continue

        stale_ref_set = set(stale_refs)
        subsections_removed_by_page[rel] = removed_count
        remaining_refs = _remaining_summ_subsection_refs(main_after)
        if not remaining_refs:
            pages_deleted.append(rel)
            if not dry_run:
                delete_path(page)
            continue

        sections["Main content"] = main_after
        sections["Source refs"] = _strip_matching_bullet_refs(
            sections.get("Source refs", ""),
            stale_ref_set,
        )
        sections["__preamble__"] = _strip_matching_refs_block(
            sections.get("__preamble__", ""),
            stale_ref_set,
            header_pattern=r"^\s*\*\*Derived from\*\*:\s*$",
        )
        sections["__preamble__"] = update_last_updated_preamble(sections.get("__preamble__", ""))
        updated_text = compose_sections(sections)
        pages_mutated.append(rel)
        if not dry_run:
            write_text_utf8(page, updated_text)

    rebuild_result = command_rebuild(codex_root=codex_root, dry_run=dry_run)

    pre_keys = {f"{e.section}:{e.title}:{e.rel_path}" for e in pre_entries}
    post_keys = {str(x) for x in list(rebuild_result.get("entry_keys", []) or []) if str(x)}
    removed_entries = sorted(
        list(pre_keys - post_keys)
    )

    log_entry_text = ""
    if not dry_run:
        ensure_parent(log_path)
        if not log_path.exists():
            write_text_utf8(log_path, "# Codex Log\n\n")
        lines = [f"## {utc_now()}", "- Clean summary:"]
        if subsections_removed_by_page:
            total_removed = sum(subsections_removed_by_page.values())
            lines.append(f"  - subsections_removed={total_removed}")
            for rel in sorted(subsections_removed_by_page):
                lines.append(f"  - page={rel} removed={subsections_removed_by_page[rel]}")
        else:
            lines.append("  - subsections_removed=0")
        if pages_deleted:
            lines.append("  - pages_deleted:")
            for rel in pages_deleted:
                lines.append(f"    - {rel}")
        else:
            lines.append("  - pages_deleted=(none)")
        if removed_entries:
            lines.append("  - index_entries_removed:")
            for item in removed_entries:
                lines.append(f"    - {item}")
        else:
            lines.append("  - index_entries_removed=(none)")
        lines.append("")
        log_entry_text = "\n".join(lines)
        append_text_utf8(log_path, log_entry_text)
    else:
        lines = [f"## {utc_now()}", "- Clean summary:"]
        if subsections_removed_by_page:
            total_removed = sum(subsections_removed_by_page.values())
            lines.append(f"  - subsections_removed={total_removed}")
            for rel in sorted(subsections_removed_by_page):
                lines.append(f"  - page={rel} removed={subsections_removed_by_page[rel]}")
        else:
            lines.append("  - subsections_removed=0")
        if pages_deleted:
            lines.append("  - pages_deleted:")
            for rel in pages_deleted:
                lines.append(f"    - {rel}")
        else:
            lines.append("  - pages_deleted=(none)")
        if removed_entries:
            lines.append("  - index_entries_removed:")
            for item in removed_entries:
                lines.append(f"    - {item}")
        else:
            lines.append("  - index_entries_removed=(none)")
        lines.append("")
        log_entry_text = "\n".join(lines)

    return {
        "subsections_removed_by_page": subsections_removed_by_page,
        "pages_deleted": pages_deleted,
        "pages_mutated": sorted(pages_mutated),
        "index_entries_removed": removed_entries,
        "rebuilt_entries": int(rebuild_result.get("rebuilt_entries", 0) or 0),
        "rebuild_skipped": list(rebuild_result.get("skipped", []) or []),
        "log_entry_text": log_entry_text,
    }


def run_update(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    summ_path = Path(args.summ_path).resolve() if args.summ_path else None
    codex_root = Path(args.codex_root).resolve()
    docz_root = Path(args.docz_root).resolve() if args.docz_root else None
    dry_run = bool(args.dry_run)

    if args.rebuild:
        result = command_rebuild(codex_root=codex_root, dry_run=dry_run)
        print(f">>codex rebuild: rebuilt_entries={result['rebuilt_entries']} skipped={len(result['skipped'])}")
        for s in result["skipped"]:
            print(f"SKIPPED: {s}")
        return 0

    if args.doctor:
        result = command_doctor(codex_root=codex_root)
        print(f">>codex doctor: findings={len(result['findings'])}")
        for f in result["findings"]:
            print(f"FINDING: {f}")
        return 0

    if args.lint:
        result = command_lint(codex_root=codex_root)
        findings = list(result.get("findings", []) or [])
        print(f">>codex lint: findings={max(0, len(findings) - 1)}")
        for f in findings:
            print(f)
        return 0

    if args.clean:
        result = command_clean(codex_root=codex_root, dry_run=dry_run)
        removed_map = dict(result.get("subsections_removed_by_page", {}) or {})
        total_removed = sum(int(v or 0) for v in removed_map.values())
        pages_deleted = list(result.get("pages_deleted", []) or [])
        removed_entries = list(result.get("index_entries_removed", []) or [])
        print(
            f">>codex clean: subsections_removed={total_removed} "
            f"pages_deleted={len(pages_deleted)} index_entries_removed={len(removed_entries)}"
        )
        if removed_map:
            for rel in sorted(removed_map):
                print(f"REMOVED_SUBSECTIONS: {rel} count={removed_map[rel]}")
        if pages_deleted:
            for rel in pages_deleted:
                print(f"DELETED_PAGE: {rel}")
        if removed_entries:
            for item in removed_entries:
                print(f"REMOVED_INDEX: {item}")
        if not removed_map and not pages_deleted and not removed_entries:
            print("NOTHING_TO_CLEAN")
        return 0

    if not summ_path:
        print("ERROR: summ_path is required unless --rebuild or --doctor or --lint is used")
        return 2
    if not summ_path.exists():
        print(f"ERROR: SUMM path missing: {summ_path}")
        return 2
    if summ_path.name.startswith("SUMM_") is False:
        print(f"ERROR: invalid SUMM filename (must start with SUMM_): {summ_path}")
        return 2

    ensure_codex_bootstrap(codex_root, dry_run=dry_run)
    index_path = codex_root / "index.md"
    log_path = codex_root / "log.md"
    pages_created: List[str] = []
    pages_updated: List[str] = []
    pages_skipped: List[str] = []
    result_payload: Dict[str, object] = {
        "status": "ok",
        "summ_path": str(summ_path).replace("\\", "/"),
        "pages_created": pages_created,
        "pages_updated": pages_updated,
        "pages_skipped": pages_skipped,
        "collisions": [],
        "contradictions_added": 0,
        "index_entries_added": 0,
        "index_entries_updated": 0,
        "log_appended": False,
    }
    try:
        summ_text = sanitize_summ_ingest_text(read_text_utf8(summ_path))
        kb_name = args.kb_name
        if not kb_name and docz_root and summ_path.is_relative_to(docz_root):
            parts = list(summ_path.relative_to(docz_root).parts)
            if len(parts) >= 3 and parts[0] == "kb":
                kb_name = parts[1]
        entries = load_index_entries(index_path)
        targets = build_identity_targets(codex_root, entries)
        candidate_terms = extract_candidate_terms(summ_path, summ_text, kb_name)
        candidate_keys = [normalize_key(t) for t in candidate_terms if normalize_key(t)]
        confident_paths: Set[str] = set()
        for ck in candidate_keys:
            confident_paths.update(targets.get(ck, set()))
        touched_paths: List[str] = []
        notes: List[str] = []

        if confident_paths:
            vetted_confident: List[str] = []
            for rel in sorted(confident_paths):
                page_path = codex_root / rel
                if not page_path.exists():
                    pages_skipped.append(f"{rel}::missing_confident_target")
                    notes.append(f"skipped confident target: {rel} (missing)")
                    continue
                try:
                    read_text_utf8(page_path)
                except Exception:
                    pages_skipped.append(f"{rel}::unreadable_confident_target")
                    notes.append(f"skipped confident target: {rel} (unreadable)")
                    continue
                vetted_confident.append(rel)
            touched_paths = vetted_confident
        if not touched_paths:
            fallback_title = choose_fallback_title(summ_path, summ_text)
            rel, collision_occurred = page_rel_to_section_slug(fallback_title, codex_root)
            if collision_occurred:
                base = f"topics/{slugify(fallback_title)}"
                result_payload["collisions"] = [rel]
                notes.append(f"slug collision: {base}")
            touched_paths = [rel]
            notes.append("default-topic classification used")

        new_links_by_page: Dict[str, List[str]] = {}
        would_created: List[str] = []
        would_updated: List[str] = []
        created = pages_created
        updated = pages_updated
        contradictions_added = 0
        original_ref = extract_original_ref(summ_path)

        for rel in touched_paths:
            page_path = codex_root / rel
            if not page_path.exists():
                if dry_run:
                    if rel not in would_created:
                        would_created.append(rel)
                    if rel not in would_updated:
                        would_updated.append(rel)
                    continue
                created.append(rel)
                ensure_base_page(
                    page_path,
                    choose_fallback_title(summ_path, summ_text),
                    dry_run=dry_run,
                    summ_ref=summ_path.name,
                    original_ref=original_ref,
                )
            old_text = read_text_utf8(page_path)
            _, block_text, headings = append_subsection_and_refs(page_path, summ_path, summ_text, dry_run=dry_run)
            xlinks = compute_crosslinks(codex_root, rel, block_text, headings, entries)
            if xlinks:
                if not dry_run:
                    sec = split_sections(read_text_utf8(page_path))
                    add_or_update_related_pages(sec, xlinks)
                    write_text_utf8(page_path, compose_sections(sec))
                new_links_by_page[rel] = xlinks
            if dry_run:
                if rel not in would_updated:
                    would_updated.append(rel)
            elif rel not in updated:
                updated.append(rel)
            for line in summ_text.splitlines():
                if is_unresolved_contradiction(line):
                    contradictions_added += 1

        idx_added, idx_updated = update_index_for_pages(codex_root, entries, touched_paths, dry_run=dry_run)
        append_log_entry(log_path, summ_path, original_ref, updated, notes, dry_run=dry_run)

        result_payload["pages_created"] = created
        result_payload["pages_updated"] = updated
        result_payload["contradictions_added"] = contradictions_added
        result_payload["index_entries_added"] = idx_added
        result_payload["index_entries_updated"] = idx_updated
        result_payload["log_appended"] = not dry_run

        if any(page_conflict_stale(read_text_utf8(codex_root / p)) for p in updated if (codex_root / p).exists()):
            result_payload["conflict_warning"] = CONFLICT_WARNING

        if not dry_run:
            telem_path = write_telemetry(repo_root, summ_path.stem, result_payload)
            print(f"telemetry: {telem_path.as_posix()}")
        if dry_run:
            print(
                "codex_update summary: "
                f"would_create={len(would_created)} would_update={len(would_updated)} "
                f"index_added={idx_added} index_updated={idx_updated} "
                f"collisions={len(result_payload['collisions'])} dry_run={dry_run}"
            )
        else:
            print(
                "codex_update summary: "
                f"created={len(created)} updated={len(updated)} index_added={idx_added} "
                f"index_updated={idx_updated} collisions={len(result_payload['collisions'])} dry_run={dry_run}"
            )
        if args.json:
            print(json.dumps(result_payload, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        error_payload: Dict[str, object] = {
            "status": "error",
            "summ_path": str(summ_path).replace("\\", "/") if summ_path else "",
            "error_class": exc.__class__.__name__,
            "message": str(exc),
            "pages_created": pages_created,
            "pages_updated": pages_updated,
            "pages_skipped": pages_skipped,
        }
        if not dry_run and summ_path:
            write_telemetry(repo_root, summ_path.stem, error_payload)
        print(f"ERROR: SUMM={summ_path}")
        print(f"ERROR_CLASS: {exc.__class__.__name__}")
        print(f"MESSAGE: {exc}")
        print(f"PAGES_CREATED: {pages_created}")
        print(f"PAGES_UPDATED: {pages_updated}")
        print(f"PAGES_SKIPPED: {pages_skipped}")
        if args.verbose:
            traceback.print_exc()
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic Codex updater")
    parser.add_argument("summ_path", nargs="?", help="Path to one SUMM_*.md file")
    parser.add_argument("--codex-root", required=True, help="Path to codex root")
    parser.add_argument("--docz-root", default="", help="Path to docz root or kb root")
    parser.add_argument("--kb-name", default="", help="KB name override")
    parser.add_argument("--dry-run", action="store_true", help="No file mutations")
    parser.add_argument("--json", action="store_true", help="Also print JSON payload to stdout")
    parser.add_argument("--verbose", action="store_true", help="Verbose errors")
    parser.add_argument("--rebuild", action="store_true", help="Run codex rebuild")
    parser.add_argument("--doctor", action="store_true", help="Run codex doctor")
    parser.add_argument("--lint", action="store_true", help="Run codex lint (read-only quality diagnostics)")
    parser.add_argument("--clean", action="store_true", help="Run codex clean (remove stale SUMM-backed subsections)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_update(args)


if __name__ == "__main__":
    raise SystemExit(main())
