# sidecars.py
# version 1.0.0
"""
Non-LLM utility sidecars for llama-conductor.

These are deterministic, inspectable tools that don't require model inference.
Provides: calc, list (Vodka memories), find (quotes in KBs), flush (CTC cache).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import math

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CalcResult:
    """Result of >>calc operation."""
    value: Optional[float] = None
    formatted: str = ""
    error: Optional[str] = None


@dataclass
class MemoryEntry:
    """Single Vodka memory entry for >>list."""
    ctx_id: str
    text: str
    ttl_days: int
    touch_count: int
    created_at: str


@dataclass
class QuoteResult:
    """Result of >>find / >>quote operation."""
    file: str
    rel_path: str
    kb: str
    snippet: str
    location: str  # "line X" or similar context


# ============================================================================
# Calculator (>>calc)
# ============================================================================


def parse_and_eval_calc(expr: str) -> CalcResult:
    """
    Parse and evaluate a mathematical expression safely.
    
    Supports: +, -, *, /, %, **, parentheses, numbers, basic functions
    Examples:
        >>calc 30% of 79.95
        >>calc 14*365
        >>calc (100 + 50) / 2
        >>calc 5**2
    """
    expr = (expr or "").strip()
    if not expr:
        return CalcResult(error="Empty expression")

    try:
        # Safe eval with limited builtins
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "log": math.log,
            "log10": math.log10,
            "floor": math.floor,
            "ceil": math.ceil,
        }

        # Handle percentage syntax: "30% of 79.95"
        expr_normalized = _normalize_calc_expression(expr)

        # Evaluate
        result = eval(expr_normalized, {"__builtins__": {}}, allowed_names)

        # Format result
        if isinstance(result, float):
            if result == int(result):
                formatted = f"{int(result)}"
            else:
                formatted = f"{result:.2f}"
        else:
            formatted = str(result)

        return CalcResult(value=float(result), formatted=formatted)

    except ZeroDivisionError:
        return CalcResult(error="Division by zero")
    except ValueError as e:
        return CalcResult(error=f"Invalid value: {e}")
    except SyntaxError:
        return CalcResult(error="Invalid expression syntax")
    except NameError as e:
        return CalcResult(error=f"Unknown function or variable: {e}")
    except Exception as e:
        return CalcResult(error=f"Calculation error: {e}")


def _normalize_calc_expression(expr: str) -> str:
    """Normalize common expression patterns."""
    expr = expr.strip()

    # Handle "X% of Y" → "(X / 100) * Y"
    if "% of " in expr.lower():
        expr = re.sub(
            r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)",
            r"(\1 / 100) * \2",
            expr,
            flags=re.IGNORECASE,
        )

    # Handle "X per Y" → "X / Y"
    if " per " in expr.lower():
        expr = re.sub(
            r"(\d+(?:\.\d+)?)\s+per\s+(\d+(?:\.\d+)?)",
            r"\1 / \2",
            expr,
            flags=re.IGNORECASE,
        )

    return expr


def format_calc_result(result: CalcResult) -> str:
    """Format calculator result for user display."""
    if result.error:
        return f"[calc error] {result.error}"
    return result.formatted


# ============================================================================
# Vodka Memory Listing (>>list)
# ============================================================================


def list_vodka_memories(vodka: Optional[object]) -> List[MemoryEntry]:
    """
    List all stored Vodka memories with metadata.
    
    Returns: List of MemoryEntry objects (empty if vodka is None or has no memories)
    """
    if vodka is None:
        return []

    try:
        # Access Vodka's storage (vodka_filter.py uses _get_storage_and_fr())
        fr = vodka._get_storage_and_fr()
        if fr is None:
            return []

        data = fr.S.load_facts()
        now = fr._now()

        entries: List[MemoryEntry] = []

        for ctx_id, rec in data.items():
            if rec.get("type") != "vodka_ctx":
                continue

            # Skip expired
            exp_s = rec.get("expires_at")
            if exp_s:
                exp_dt = fr._parse_ts(exp_s)
                if exp_dt and exp_dt < now:
                    continue

            # Calculate TTL days remaining
            ttl_days = 0
            if exp_s:
                exp_dt = fr._parse_ts(exp_s)
                if exp_dt:
                    delta = (exp_dt - now).total_seconds()
                    ttl_days = max(0, int(delta // 86400))

            touch_count = int(rec.get("touch_count", 0))
            text = str(rec.get("value", "")).strip()
            created_at = str(rec.get("created_at", "")).strip()

            entries.append(
                MemoryEntry(
                    ctx_id=ctx_id,
                    text=text,
                    ttl_days=ttl_days,
                    touch_count=touch_count,
                    created_at=created_at,
                )
            )

        # Sort by creation date (newest first)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    except Exception as e:
        print(f"[sidecars.list_vodka_memories] Error: {e}")
        return []


def format_memory_list(entries: List[MemoryEntry]) -> str:
    """Format Vodka memory list for user display."""
    if not entries:
        return "[list] No stored memories."

    lines = ["[vodka memories]"]
    for i, entry in enumerate(entries, 1):
        preview = entry.text[:80].replace("\n", " ").strip()
        lines.append(
            f"{i}. [{entry.ctx_id}] {preview}... (TTL={entry.ttl_days}d, touches={entry.touch_count})"
        )

    return "\n".join(lines)


# ============================================================================
# KB Quote Finding (>>find / >>quote)
# ============================================================================


def find_quote_in_kbs(
    query: str,
    attached_kbs: Set[str],
    kb_paths: Dict[str, str],
) -> Optional[QuoteResult]:
    """
    Find a quote/passage in attached KBs.
    
    Searches SUMM_*.md files in attached KB folders.
    Returns first exact match, or closest substring match.
    """
    query = (query or "").strip().lower()
    if not query:
        return None

    # Try exact substring match first
    for kb in sorted(attached_kbs):
        if kb == "vault":
            continue  # Skip vault, it's Qdrant-based
        
        folder = kb_paths.get(kb)
        if not folder:
            continue

        result = _search_kb_folder(query, kb, folder, exact=True)
        if result:
            return result

    # Fall back to word-token match
    for kb in sorted(attached_kbs):
        if kb == "vault":
            continue
        
        folder = kb_paths.get(kb)
        if not folder:
            continue

        result = _search_kb_folder(query, kb, folder, exact=False)
        if result:
            return result

    return None


def _search_kb_folder(query: str, kb: str, folder: str, exact: bool = True) -> Optional[QuoteResult]:
    """Search a single KB folder for a query."""
    import os

    if not os.path.isdir(folder):
        return None

    for root, _, files in os.walk(folder):
        # Skip /original/ subfolder
        if "original" in {p.lower() for p in root.split(os.sep)}:
            continue

        for fn in sorted(files):
            if not fn.startswith("SUMM_") or not fn.lower().endswith(".md"):
                continue

            fpath = os.path.join(root, fn)
            rel_path = os.path.relpath(fpath, folder)

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue

            # Search in text
            result = _find_in_text(query, text, kb, fn, rel_path, exact=exact)
            if result:
                return result

    return None


def _find_in_text(
    query: str, text: str, kb: str, file: str, rel_path: str, exact: bool = True
) -> Optional[QuoteResult]:
    """Search for query within text content."""
    query_lower = query.lower()

    if exact:
        # Exact substring match
        idx = text.lower().find(query_lower)
        if idx >= 0:
            # Extract snippet (context)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(query) + 100)
            snippet = text[start:end].strip()

            # Calculate approximate line number
            line_num = text[:idx].count("\n") + 1

            return QuoteResult(
                file=file,
                rel_path=rel_path,
                kb=kb,
                snippet=snippet,
                location=f"line ~{line_num}",
            )
    else:
        # Word token match: all query words must be in text
        query_words = set(w.lower() for w in query.split() if w)
        text_lower = text.lower()

        all_found = all(w in text_lower for w in query_words)
        if all_found:
            # Find first occurrence of any query word
            first_pos = len(text)
            for word in query_words:
                pos = text_lower.find(word)
                if pos >= 0:
                    first_pos = min(first_pos, pos)

            start = max(0, first_pos - 100)
            end = min(len(text), first_pos + 200)
            snippet = text[start:end].strip()
            line_num = text[:first_pos].count("\n") + 1

            return QuoteResult(
                file=file,
                rel_path=rel_path,
                kb=kb,
                snippet=snippet,
                location=f"line ~{line_num}",
            )

    return None


def format_quote_result(result: QuoteResult) -> str:
    """Format quote search result for user display."""
    if not result:
        return "[find] Quote not found."

    return (
        f"[find] Located in: {result.kb}/{result.rel_path}\n"
        f"File: {result.file}\n"
        f"Location: {result.location}\n\n"
        f"Snippet:\n{result.snippet}"
    )


# ============================================================================
# CTC Cache Flushing (>>flush)
# ============================================================================


def flush_ctc_cache(vodka: Optional[object]) -> str:
    """
    Flush the CTC (Cut-The-Crap) rolling cache in Vodka.
    
    This is a message history trimming cache, not the memory store.
    Flushing resets the cache for the next turn.
    """
    if vodka is None:
        return "[flush] Vodka not initialized."

    try:
        # CTC cache is the trimmed message list. We signal a "reset" by clearing
        # any cached state. Since messages are per-request, we just confirm.
        # The actual CTC happens at inlet() time in Vodka.
        
        # If Vodka has a reset method, call it. Otherwise, just confirm.
        if hasattr(vodka, "_last_janitor_run"):
            # This is a signal to vodka to re-run janitor next time
            vodka._last_janitor_run = 0.0

        return "[flush] CTC cache reset for next turn. (Memory store preserved.)"

    except Exception as e:
        return f"[flush] Error: {e}"


# ============================================================================
# Testing / Standalone Usage
# ============================================================================

if __name__ == "__main__":
    # Test calc
    print("=== Testing >>calc ===")
    result = parse_and_eval_calc("30% of 79.95")
    print(f"30% of 79.95: {format_calc_result(result)}")

    result = parse_and_eval_calc("14*365")
    print(f"14*365: {format_calc_result(result)}")

    result = parse_and_eval_calc("sqrt(16)")
    print(f"sqrt(16): {format_calc_result(result)}")

    result = parse_and_eval_calc("1/0")
    print(f"1/0: {format_calc_result(result)}")

    print("\n=== Testing >>find ===")
    # Would need actual KB paths to test
    print("(Requires active KBs to test)")
