# trust_pipeline.py
# version 1.1.1
"""
>>trust mode - Tool recommendation pipeline (invariants-compliant)

This is a sidecar that analyzes queries and recommends appropriate tools.
It DOES NOT auto-execute anything - only suggests options to the user.

Design principles:
- Preserves user control and explicit routing
- Remains transparent and predictable
- No auto-execution, no implicit routing, no escalation
- Router stays dumb (just routes to this pipeline)

v1.1.1 changes (surgical, backwards-compatible):
- Recognize encyclopedia / wiki-shaped queries and recommend >>wiki
- Improve currency/exchange detection (e.g., "1 AUD to USD") and recommend >>exchange
- Keep deterministic regex/heuristic matching (no model calls, no external lookups)
- Improve >>wiki usability: suggest article-title commands for question-shaped queries
- Recognize single-token lowercase encyclopedia lookups (e.g., 'deathclaw')
"""

from typing import List, Dict, Any, Set
import re


# ============================================================================
# Query Classification
# ============================================================================

# Common ISO-4217 currency codes (not exhaustive, but covers the practical set).
# Deterministic: hard-coded allowlist to reduce false positives.
_CURRENCY_CODES = {
    "aud","usd","eur","gbp","jpy","cad","nzd","chf","cny","hkd","sgd","inr","krw","sek","nok","dkk",
    "mxn","brl","zar","rub","try","thb","idr","myr","php","vnd","pln","czk","huf","ils","sar","aed",
    "qar","kwd","bhd","omr","egp","pkr","bdt","lkr","ngn","kes","ghs","twd"
}

# Signals for encyclopedia-style lookups. These should tend toward >>wiki suggestions.
_ENCYCLOPEDIA_LEADS = (
    "what is", "who is", "who was", "where is", "when did", "when was", "define",
    "origin of", "history of", "meaning of", "what are", "who are"
)

# Words that usually indicate multi-step reasoning rather than encyclopedia lookup.
_REASONING_MARKERS = (
    "compare", "analyze", "evaluate", "assess", "pros and cons", "argue", "debate", "justify",
    "why", "how does", "how do", "explain"
)

# Single-token queries that should NOT trigger >>wiki suggestions.
# Deterministic denylist to avoid wiki-spam on chatter / acknowledgements.
_WIKI_SINGLE_TOKEN_DENYLIST = {
    "hi", "hello", "hey", "help", "thanks", "thank", "ok", "okay", "yes", "no",
    "lol", "lmao", "test", "testing"
}



def classify_query(query: str) -> Dict[str, Any]:
    """
    Classify query into categories using deterministic regex/heuristic matching.

    Returns:
        {
            'primary_type': str,  # math, current_data, factual, complex_reasoning, creative, general
            'patterns': List[str]  # detected patterns
        }
    """

    q = (query or "").strip()

    patterns: List[str] = []

    # --- Math patterns ---
    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", q):
        patterns.append("arithmetic_expression")
    if re.search(r"\d+%|percent|calculate|compute|solve|what.?s\s+\d+", q, re.I):
        patterns.append("math_question")

    # --- Current data patterns ---
    if re.search(r"\b(today|now|current|latest|right now)\b", q, re.I):
        patterns.append("current_data")
    if re.search(r"\b(weather|temperature|forecast)\b", q, re.I):
        patterns.append("weather")

    # Currency/exchange: broaden detection to include ISO codes and common formatting.
    if _is_exchange_query(q):
        patterns.append("exchange")

    # --- Factual / encyclopedia patterns ---
    if re.search(r"^(what is|who is|who was|when did|where is|how many|define|origin of|history of|meaning of)", q, re.I):
        patterns.append("factual_question")

    if re.search(r"\b(capital|population|born|died|invented|president|elected)\b", q, re.I):
        patterns.append("factual_lookup")

    if _is_encyclopedia_query(q):
        patterns.append("encyclopedia_lookup")

    # --- Reasoning patterns ---
    if re.search(r"\b(compare|analyze|evaluate|assess|pros and cons)\b", q, re.I):
        patterns.append("complex_reasoning")
    if q.count("?") > 1:
        patterns.append("multi_question")
    if _detect_multi_step(q):
        patterns.append("multi_step")

    # --- Creative patterns ---
    if re.search(r"\b(write|create|generate|make me a|compose)\b", q, re.I):
        patterns.append("creative_generation")

    # Determine primary type
    if "arithmetic_expression" in patterns or "math_question" in patterns:
        primary_type = "math"
    elif "weather" in patterns or "exchange" in patterns or "current_data" in patterns:
        primary_type = "current_data"
    elif "factual_question" in patterns or "factual_lookup" in patterns:
        primary_type = "factual"
    elif "complex_reasoning" in patterns or "multi_step" in patterns:
        primary_type = "complex_reasoning"
    elif "creative_generation" in patterns:
        primary_type = "creative"
    else:
        primary_type = "general"

    return {"primary_type": primary_type, "patterns": patterns}


def _detect_multi_step(text: str) -> bool:
    """Detect if query requires multi-step reasoning."""
    step_markers = ["then", "after", "next", "finally", "first", "second", "third"]
    step_count = sum(1 for marker in step_markers if marker in (text or "").lower())
    return step_count >= 2


def _extract_math_expression(query: str) -> str:
    """Extract clean math expression from natural language query."""
    clean = re.sub(r"^(what is|what's|whats|calculate|compute|solve|find|tell me)\s+", "", query, flags=re.I)
    clean = clean.strip()
    # Strip trailing punctuation that commonly appears in questions
    clean = re.sub(r"[\?\.!]+$", "", clean).strip()
    return clean

def _suggest_wiki_title(query: str) -> str:
    """Suggest an article-title style query for >>wiki.

    Deterministic, conservative transformation:
      - strips common encyclopedia lead phrases (e.g., "what is", "who is", "origin of")
      - strips leading articles ("a", "an", "the")
      - trims trailing context clauses (e.g., "in fallout", "from x") when they are clearly contextual
      - lightly normalizes casing for single-token all-lowercase terms (capitalize first letter)

    This is ONLY used to format a suggested command shown to the user.
    It does NOT auto-execute anything and does not modify the user's query itself.
    """
    q = (query or "").strip()
    if not q:
        return q

    ql = q.lower().strip()

    # Remove common lead phrases
    for lead in sorted(_ENCYCLOPEDIA_LEADS, key=len, reverse=True):
        if ql.startswith(lead):
            q = q[len(lead):].strip()
            break

    # Remove leading articles
    q = re.sub(r"^(a|an|the)\s+", "", q, flags=re.I).strip()

    # If query contains a clear context clause, keep the left-hand subject.
    # Examples: "deathclaw in fallout" -> "deathclaw"
    q = re.split(r"\s+(in|from|within|inside|universe|series)\s+", q, maxsplit=1, flags=re.I)[0].strip()

    # Trim common trailing franchise/context words when query is exactly 2 tokens.
    # Example: "deathclaw fallout" -> "deathclaw"
    toks = re.findall(r"[A-Za-z0-9'/\-]+", q)
    if len(toks) == 2:
        tail = toks[1].lower()
        if tail in {"fallout", "universe", "game", "series", "franchise", "lore"}:
            q = toks[0]

    # Light casing normalization for single-token all-lowercase alphabetic terms:
    # "deathclaw" -> "Deathclaw" (helps title lookup; still deterministic)
    toks2 = re.findall(r"[A-Za-z0-9'/\-]+", q)
    if len(toks2) == 1 and toks2[0].isalpha() and toks2[0].islower():
        q = toks2[0].capitalize()

    q = re.sub(r"[\?\.!]+$", "", q).strip()
    return q



def _is_exchange_query(query: str) -> bool:
    """
    Heuristic: detect currency conversion / exchange-rate queries.

    Examples:
      - "1 AUD to USD"
      - "AUD/USD"
      - "audusd"
      - "convert aud to usd"
      - "exchange rate aud usd"

    Deterministic: regex + allowlisted ISO codes.
    """
    q = (query or "").strip()
    ql = q.lower()

    # Direct keyword signals
    if re.search(r"\b(exchange|currency|fx|forex)\b", ql):
        return True
    if re.search(r"\b(convert|conversion|rate)\b", ql):
        # Might still be non-currency, so require an ISO code presence as well.
        if _contains_currency_code(ql):
            return True

    # ISO code pair patterns
    #  - "AUD to USD" / "AUD in USD" / "AUD->USD"
    if re.search(r"\b[a-z]{3}\b\s*(to|in|into|=|->|→)\s*\b[a-z]{3}\b", ql):
        if _contains_currency_code_pair(ql):
            return True

    # Slash format "AUD/USD"
    if re.search(r"\b[a-z]{3}\s*/\s*[a-z]{3}\b", ql):
        if _contains_currency_code_pair(ql):
            return True

    # Concatenated format "audusd"
    if re.search(r"\b[a-z]{6}\b", ql):
        tok = re.search(r"\b([a-z]{6})\b", ql)
        if tok:
            s = tok.group(1)
            a, b = s[:3], s[3:]
            if a in _CURRENCY_CODES and b in _CURRENCY_CODES:
                return True

    # With amount + codes, common query "1 AUD to USD"
    if re.search(r"\b\d+(?:\.\d+)?\s*\b[a-z]{3}\b\s*(to|in|into)\s*\b[a-z]{3}\b", ql):
        if _contains_currency_code_pair(ql):
            return True

    return False


def _contains_currency_code(text_lower: str) -> bool:
    return any(re.search(rf"\b{code}\b", text_lower) for code in _CURRENCY_CODES)


def _contains_currency_code_pair(text_lower: str) -> bool:
    codes = re.findall(r"\b[a-z]{3}\b", text_lower)
    if len(codes) < 2:
        return False
    # Ensure at least one adjacent pair are in allowlist
    for i in range(len(codes) - 1):
        if codes[i] in _CURRENCY_CODES and codes[i + 1] in _CURRENCY_CODES:
            return True
    return False


def _is_encyclopedia_query(query: str) -> bool:
    """
    Heuristic: detect encyclopedia/wiki-shaped lookups.

    This is used ONLY for recommendations (no auto-execution).
    Deterministic, conservative: aims to suggest >>wiki when likely helpful.

    Signals:
      - common lead phrases (who is/what is/origin/history/meaning/define)
      - short named-entity queries (e.g., "Albert Einstein", "New York City")
      - avoids triggering for explicit multi-step reasoning prompts
    """
    q = (query or "").strip()
    if not q:
        return False

    ql = q.lower()

    # Avoid suggesting wiki-first for obvious reasoning/comparison prompts.
    for m in _REASONING_MARKERS:
        if m in ql:
            return False

    # Lead phrases.
    for lead in _ENCYCLOPEDIA_LEADS:
        if ql.startswith(lead):
            return True

    # Very short noun phrase that looks like a named entity (title case / caps / contains proper nouns).
    # Conservative: require at least one capital letter in original text, and 1-6 tokens.
    tokens = re.findall(r"[A-Za-z0-9']+", q)
    if 1 <= len(tokens) <= 6:
        has_capital = any(any(c.isupper() for c in t) for t in tokens)
        # Also allow all-lower short entities ("new york city") by matching multiple tokens with common place-like structure.
        # For all-lower, require 2+ tokens and no obvious verbs.
        if has_capital:
            return True
        if len(tokens) >= 2 and not re.search(r"\b(is|are|was|were|be|do|does|did|make|create|write|generate)\b", ql):
            # Example: "new york city", "world war ii" (roman numerals allowed via tokens)
            return True


    # Single-token lowercase entity lookups should still offer >>wiki (as an option),
    # since users commonly type bare nouns (e.g., "deathclaw", "linux").
    # Deterministic: require alphabetic token, not in denylist, and not a reasoning prompt.
    if len(tokens) == 1:
        t0 = tokens[0]
        if t0.isalpha() and t0.islower() and len(t0) >= 3 and t0 not in _WIKI_SINGLE_TOKEN_DENYLIST:
            return True
    return False


# ============================================================================
# Resource Checking
# ============================================================================


def check_resources(attached_kbs: Set[str], kb_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Check what resources are currently available.

    Returns:
        {
            'kbs_attached': List[str],   # Currently attached filesystem KBs
            'kbs_available': List[str],  # All available filesystem KBs
            'has_kbs': bool              # Whether any KBs are attached
        }
    """
    return {
        "kbs_attached": sorted(list(attached_kbs)),
        "kbs_available": sorted(list(kb_paths.keys())),
        "has_kbs": len(attached_kbs) > 0
    }


# ============================================================================
# Recommendation Generation
# ============================================================================


def generate_recommendations(
    query: str,
    attached_kbs: Set[str],
    kb_paths: Dict[str, str],
    vault_kb_name: str
) -> List[Dict[str, str]]:
    """
    Generate ranked tool recommendations for a query.

    NOTE: Recommendations only. No auto-execution.

    Returns:
        List of recommendations, each with:
        - rank: str (A, B, C, D, etc.)
        - tool: str
        - confidence: str (HIGH, MEDIUM, LOW)
        - reason: str
        - command: str (exact command to run)
    """

    classification = classify_query(query)
    resources = check_resources(attached_kbs, kb_paths)

    recommendations: List[Dict[str, str]] = []
    primary_type = classification["primary_type"]
    pats = set(classification.get("patterns", []))

    # Helper: stable rank letters
    def _next_rank(rank_int: int) -> str:
        return chr(rank_int)

    # ===== MATH QUERIES =====
    if primary_type == "math":
        clean_expr = _extract_math_expression(query)
        recommendations.append({
            "rank": "A",
            "tool": ">>calc",
            "confidence": "HIGH",
            "reason": "Deterministic calculation - guaranteed accuracy",
            "command": f">>calc {clean_expr}"
        })
        recommendations.append({
            "rank": "B",
            "tool": "serious mode",
            "confidence": "LOW",
            "reason": "Model estimation - unreliable for math",
            "command": query
        })
        return recommendations

    # ===== CURRENT DATA QUERIES =====
    if primary_type == "current_data":
        r = ord("A")

        if "weather" in pats:
            recommendations.append({
                "rank": _next_rank(r),
                "tool": ">>weather",
                "confidence": "HIGH",
                "reason": "Live weather data from external API",
                "command": f">>weather {query}"
            })
            r += 1

        # Exchange should be strongly preferred for currency conversion/rates
        if "exchange" in pats:
            recommendations.append({
                "rank": _next_rank(r),
                "tool": ">>exchange",
                "confidence": "HIGH",
                "reason": "Live currency exchange data",
                "command": f">>exchange {query}"
            })
            r += 1

        # Fallback
        if not recommendations:
            recommendations.append({
                "rank": "A",
                "tool": "serious mode",
                "confidence": "LOW",
                "reason": "Model knowledge - may be outdated for current data",
                "command": query
            })
        return recommendations

    # ===== FACTUAL QUERIES =====
    if primary_type == "factual":
        r = ord("A")

        # If the query is encyclopedia-shaped, offer >>wiki prominently.
        if "encyclopedia_lookup" in pats:
            recommendations.append({
                "rank": _next_rank(r),
                "tool": ">>wiki",
                "confidence": "HIGH",
                "reason": "Encyclopedia-style lookup (stable public reference)",
                "command": f">>wiki {_suggest_wiki_title(query)}",
                "note": "Use the entity/article title (not the full question)."
            })
            r += 1

        # Mentats remains valuable when vault knowledge is relevant.
        recommendations.append({
            "rank": _next_rank(r),
            "tool": "##mentats",
            "confidence": "HIGH" if "encyclopedia_lookup" not in pats else "MEDIUM",
            "reason": f"Multi-pass verified reasoning using {vault_kb_name} (Qdrant)",
            "command": f"##mentats {query}"
        })
        r += 1

        # KB-based grounding options
        if resources["has_kbs"]:
            kb_list = ", ".join(resources["kbs_attached"])
            recommendations.append({
                "rank": _next_rank(r),
                "tool": "serious mode (with KBs)",
                "confidence": "MEDIUM",
                "reason": f"Search attached KBs: {kb_list}",
                "command": query
            })
            r += 1
        else:
            kb_available = ", ".join(resources["kbs_available"]) if resources["kbs_available"] else "none"
            recommendations.append({
                "rank": _next_rank(r),
                "tool": ">>attach all",
                "confidence": "MEDIUM",
                "reason": f"Attach all KBs ({kb_available}) and run query",
                "command": ">>attach all"
            })
            r += 1

        recommendations.append({
            "rank": _next_rank(r),
            "tool": "serious mode (no grounding)",
            "confidence": "LOW",
            "reason": "Model knowledge only - may hallucinate facts",
            "command": query
        })
        return recommendations

    # ===== COMPLEX REASONING QUERIES =====
    if primary_type == "complex_reasoning":
        recommendations.append({
            "rank": "A",
            "tool": "##mentats",
            "confidence": "HIGH",
            "reason": f"Multi-pass verified reasoning using {vault_kb_name} (Qdrant)",
            "command": f"##mentats {query}"
        })

        if resources["has_kbs"]:
            kb_list = ", ".join(resources["kbs_attached"])
            recommendations.append({
                "rank": "B",
                "tool": "serious mode (with KBs)",
                "confidence": "MEDIUM",
                "reason": f"Single-pass reasoning with KBs: {kb_list}",
                "command": query
            })
        else:
            recommendations.append({
                "rank": "B",
                "tool": ">>attach all",
                "confidence": "MEDIUM",
                "reason": "Attach all KBs and run query for grounded reasoning",
                "command": ">>attach all"
            })

        recommendations.append({
            "rank": "C",
            "tool": "serious mode (no grounding)",
            "confidence": "LOW",
            "reason": "Single-pass reasoning from model knowledge only",
            "command": query
        })
        return recommendations

    # ===== CREATIVE QUERIES =====
    if primary_type == "creative":
        recommendations.append({
            "rank": "A",
            "tool": ">>fun",
            "confidence": "HIGH",
            "reason": "Creative generation with style",
            "command": f">>fun\n{query}"
        })
        recommendations.append({
            "rank": "B",
            "tool": "serious mode",
            "confidence": "MEDIUM",
            "reason": "Plain creative generation",
            "command": query
        })
        return recommendations

    # ===== GENERAL QUERIES =====
    # For short named-entity lookups that weren't captured as 'factual' by leading verbs,
    # suggest >>wiki as an option (but do not force it).
    if "encyclopedia_lookup" in pats:
        recommendations.append({
            "rank": "A",
            "tool": ">>wiki",
            "confidence": "HIGH",
            "reason": "Encyclopedia-style lookup (stable public reference)",
            "command": f">>wiki {_suggest_wiki_title(query)}",
            "note": "Use the entity/article title (not the full question)."
        })
        recommendations.append({
            "rank": "B",
            "tool": "serious mode",
            "confidence": "MEDIUM",
            "reason": "Default reasoning pipeline",
            "command": query
        })
        # KB suggestion remains optional, but lower priority for encyclopedia lookups.
        if resources["kbs_available"] and not resources["has_kbs"]:
            kb_list = ", ".join(resources["kbs_available"])
            recommendations.append({
                "rank": "C",
                "tool": ">>attach all",
                "confidence": "LOW",
                "reason": f"Attach KBs ({kb_list}) and run query for grounded answers",
                "command": ">>attach all"
            })
        return recommendations

    # Default general behavior (unchanged)
    recommendations.append({
        "rank": "A",
        "tool": "serious mode",
        "confidence": "MEDIUM",
        "reason": "Default reasoning pipeline",
        "command": query
    })

    if resources["kbs_available"] and not resources["has_kbs"]:
        kb_list = ", ".join(resources["kbs_available"])
        recommendations.append({
            "rank": "B",
            "tool": ">>attach all",
            "confidence": "MEDIUM",
            "reason": f"Attach KBs ({kb_list}) and run query for grounded answers",
            "command": ">>attach all"
        })

    return recommendations


# ============================================================================
# Output Formatting
# ============================================================================


def format_recommendations(recommendations: List[Dict[str, str]], query: str = "") -> str:
    """Format recommendations for user display."""

    if not recommendations:
        return "No recommendations available."

    lines: List[str] = []

    if query:
        lines.append(f"Query: {query}")
        lines.append("")
    lines.append("**Recommended Tools:**")
    lines.append("")
    for rec in recommendations:
        rank = rec["rank"]
        tool = rec["tool"]
        confidence = rec["confidence"]
        reason = rec["reason"]
        command = rec["command"]
        lines.append(f"{rank}) **{tool}** (confidence: {confidence})")
        lines.append(f"   {reason}")
        label = "Command"
        if tool == ">>wiki":
            label = "Suggested command"
        lines.append(f"   {label}: `{command}`")
        note = rec.get("note", "")
        if note:
            lines.append(f"   Note: {note}")
        lines.append("")
    lines.append("**Choose an option (A, B, C...) or type your own command.**")
    return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================


def handle_trust_command(
    query: str,
    attached_kbs: Set[str],
    kb_paths: Dict[str, str],
    vault_kb_name: str = "vault"
) -> str:
    """Main entry point for >>trust command."""

    if not query or not query.strip():
        return "[trust] usage: >>trust <query>"

    recommendations = generate_recommendations(
        query=query.strip(),
        attached_kbs=attached_kbs,
        kb_paths=kb_paths,
        vault_kb_name=vault_kb_name
    )

    return format_recommendations(recommendations, query=query.strip())


# ============================================================================
# Testing / Standalone Usage
# ============================================================================


if __name__ == "__main__":
    print("=== Testing trust_pipeline.py v1.1.1 ===\n")

    mock_kbs = set()
    mock_kb_paths = {"amiga": "/path/to/amiga", "c64": "/path/to/c64", "dogs": "/path/to/dogs"}

    tests = [
        "What's 15% of 80?",
        "What's the weather in Perth?",
        "1 AUD to USD",
        "convert aud to usd",
        "Albert Einstein",
        "who is Marie Curie",
        "what is origin of deathclaw in Fallout",
        "Compare microservices vs monolithic architecture",
    ]

    for i, t in enumerate(tests, 1):
        print(f"TEST {i}: {t}")
        print(handle_trust_command(t, mock_kbs, mock_kb_paths))
        print("\n" + "=" * 80 + "\n")
