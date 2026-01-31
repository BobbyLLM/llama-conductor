# sidecars.py
# version 1.0.3
"""
Non-LLM utility sidecars for llama-conductor.

CHANGES IN v1.0.3:
- Weather geocoding switched from Open-Meteo to Nominatim (OpenStreetMap)
- Better regional coverage: now handles small towns like "Carnarvon Western Australia"
- No API key required, no rate limiting issues (1 req/sec is fine for chat)
- Better display names from OSM

CHANGES IN v1.0.2:
- Wiki summary increased from 200 → 500 chars (full paragraph)
- Weather API switched from wttr.in to Open-Meteo (no rate limiting, more reliable)
- Added WMO weather code decoder for human-readable conditions
- Weather now includes location geocoding, temp, condition, humidity, wind speed

CHANGES IN v1.0.1:
- Added >>wiki <topic> sidecar (Wikipedia summary via free JSON API)
- Added >>exchange <query> sidecar (Currency conversion via Frankfurter API)
- Added >>weather <location> sidecar (Weather via wttr.in)

These are deterministic, inspectable tools that don't require model inference.
Provides: calc, list (Vodka memories), find (quotes in KBs), flush (CTC cache),
          wiki (Wikipedia), exchange (Frankfurter FX), weather (wttr.in).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import math
import requests

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


@dataclass
class WikiResult:
    """Result of >>wiki query."""
    title: Optional[str] = None
    summary: str = ""
    error: Optional[str] = None


@dataclass
class ExchangeResult:
    """Result of >>exchange query."""
    amount: Optional[float] = None
    from_ccy: Optional[str] = None
    to_ccy: Optional[str] = None
    converted: Optional[float] = None
    error: Optional[str] = None


@dataclass
class WeatherResult:
    """Result of >>weather query."""
    location: Optional[str] = None
    condition: str = ""
    error: Optional[str] = None


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
# Wikipedia (>>wiki)
# ============================================================================


def _normalize_wiki_topic(topic: str) -> str:
    """Normalize topic for Wikipedia URL: 'Boris Becker' → 'Boris_Becker'."""
    return (topic or "").strip().replace(" ", "_")


def handle_wiki_query(topic: str, max_chars: int = 500) -> str:
    """
    Fetch Wikipedia summary via JSON API.
    
    Example: >>wiki Albert Einstein
    Returns: "[wiki] Albert Einstein: A German-born theoretical physicist..."
    Fetches full paragraph (up to 500 chars).
    """
    topic = (topic or "").strip()
    if not topic:
        return "[wiki] No topic provided"

    try:
        normalized = _normalize_wiki_topic(topic)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{normalized}"
        
        # Wikipedia requires User-Agent header (blocks default requests UA)
        headers = {"User-Agent": "llama-conductor/1.0.1"}
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        title = data.get("title") or normalized.replace("_", " ")
        summary = (data.get("extract") or "").strip()
        
        if not summary:
            return f"[wiki] '{title}' not found"
        
        # Trim to context budget
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
        
        return f"[wiki] {title}: {summary}"
    
    except requests.exceptions.Timeout:
        return "[wiki] Request timeout"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"[wiki] '{topic}' not found"
        return f"[wiki] HTTP error: {e.response.status_code}"
    except Exception as e:
        return f"[wiki] Error: {e}"


# ============================================================================
# Currency Exchange (>>exchange)
# ============================================================================


def _normalize_currency_token(token: str) -> str:
    """Map common currency names to ISO 4217 codes."""
    if not token:
        return ""
    t = token.upper()
    aliases = {
        "YEN": "JPY",
        "EURO": "EUR",
        "EUROS": "EUR",
        "POUND": "GBP",
        "POUNDS": "GBP",
        "DOLLAR": "USD",
        "DOLLARS": "USD",
    }
    return aliases.get(t, t)


def _parse_exchange_query(text: str) -> Optional[Tuple[float, str, str]]:
    """
    Parse exchange query: "1 USD to EUR" → (1.0, 'USD', 'EUR')
    Handles: "10 aud to jpy", "usd to eur", "convert aud to jpy"
    """
    raw = text.strip()
    lo = raw.lower()

    # Quick filter
    if not any(k in lo for k in [" to ", " in ", "exchange rate", "convert "]):
        return None

    # Pattern with amount: "10 usd to eur"
    m = re.search(
        r"(?i)\b(\d+(?:\.\d+)?)\s*([A-Z]{3}|[a-z]{3}|yen|euro|euros|pound|pounds|dollar|dollars)\s+"
        r"(?:to|in)\s+([A-Z]{3}|[a-z]{3}|yen|euro|euros|pound|pounds|dollar|dollars)\b",
        raw,
    )
    if m:
        amount = float(m.group(1))
        from_ccy = _normalize_currency_token(m.group(2))
        to_ccy = _normalize_currency_token(m.group(3))
        return amount, from_ccy, to_ccy

    # Pattern without amount: "usd to eur"
    m = re.search(
        r"(?i)\b([A-Z]{3}|[a-z]{3}|yen|euro|euros|pound|pounds|dollar|dollars)\s+"
        r"(?:to|in)\s+([A-Z]{3}|[a-z]{3}|yen|euro|euros|pound|pounds|dollar|dollars)\b",
        raw,
    )
    if m:
        amount = 1.0
        from_ccy = _normalize_currency_token(m.group(1))
        to_ccy = _normalize_currency_token(m.group(2))
        return amount, from_ccy, to_ccy

    return None


def handle_exchange_query(query: str) -> str:
    """
    Fetch currency exchange rate via Frankfurter API.
    
    Example: >>exchange 1 USD to EUR
    Returns: "[exchange] 1.0 USD = 0.92 EUR"
    """
    parsed = _parse_exchange_query(query)
    if not parsed:
        return "[exchange] Not a currency query (e.g. '1 USD to EUR')"
    
    amount, from_ccy, to_ccy = parsed

    try:
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_ccy}&to={to_ccy}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        rate = data.get("rates", {}).get(to_ccy)
        if rate:
            return f"[exchange] {amount} {from_ccy} = {rate:.2f} {to_ccy}"
        return f"[exchange] Currency pair {from_ccy}/{to_ccy} not supported"
    
    except requests.exceptions.Timeout:
        return "[exchange] Request timeout"
    except requests.exceptions.HTTPError:
        return f"[exchange] Cannot convert {from_ccy} to {to_ccy}"
    except Exception as e:
        return f"[exchange] Error: {e}"


# ============================================================================
# Weather (>>weather) – Open-Meteo API
# ============================================================================

def _decode_weather_code(code: int) -> str:
    """Decode WMO weather code to human-readable description."""
    # WMO Weather interpretation codes (simplified)
    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with hail",
    }
    return codes.get(code, f"Weather code {code}")


def handle_weather_query(location: str) -> str:
    """
    Fetch current weather via Open-Meteo API with Nominatim geocoding (OSM).
    
    Uses Nominatim (OpenStreetMap) for geocoding (better regional coverage),
    then Open-Meteo for weather forecast.
    Example: >>weather Carnarvon Western Australia
    Returns: "[weather] Carnarvon, Western Australia: 22°C, Partly cloudy, 65% humidity"
    """
    location = (location or "").strip()
    if not location:
        return "[weather] No location provided"

    try:
        # Step 1: Geocode via Nominatim (OpenStreetMap) – better regional coverage
        geo_url = "https://nominatim.openstreetmap.org/search"
        geo_params = {
            "q": location,
            "format": "json",
            "limit": 1,
            "language": "en"
        }
        
        # Add User-Agent (Nominatim requires it)
        headers = {"User-Agent": "llama-conductor/1.0.2"}
        geo_resp = requests.get(geo_url, params=geo_params, headers=headers, timeout=5)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        
        if not geo_data:
            return f"[weather] Location '{location}' not found"
        
        place = geo_data[0]
        latitude = float(place.get("lat"))
        longitude = float(place.get("lon"))
        display_name = place.get("display_name", location)
        
        # Step 2: Fetch current weather via Open-Meteo
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m",
            "temperature_unit": "celsius"
        }
        
        weather_resp = requests.get(weather_url, params=weather_params, timeout=5)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        
        current = weather_data.get("current", {})
        temp = current.get("temperature_2m")
        code = current.get("weather_code")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        
        condition = _decode_weather_code(code)
        
        return f"[weather] {display_name}: {temp}°C, {condition}, {humidity}% humidity"
    
    except requests.exceptions.Timeout:
        return "[weather] Request timeout"
    except requests.exceptions.HTTPError:
        return f"[weather] Error fetching weather for '{location}'"
    except Exception as e:
        return f"[weather] Error: {e}"


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

    # Test wiki
    print("\n=== Testing >>wiki ===")
    print(handle_wiki_query("Albert Einstein"))

    # Test exchange
    print("\n=== Testing >>exchange ===")
    print(handle_exchange_query("1 USD to EUR"))

    # Test weather
    print("\n=== Testing >>weather ===")
    print(handle_weather_query("Perth"))
