# trust_pipeline.py
# version 1.0.0
"""
>>trust mode - Tool recommendation pipeline (invariants-compliant)

This is a sidecar that analyzes queries and recommends appropriate tools.
It DOES NOT auto-execute anything - only suggests options to the user.

Design principles:
- Preserves user control and explicit routing
- Remains transparent and predictable
- No auto-escalation, no implicit routing
- Router stays dumb (just routes to this pipeline)
"""

from typing import List, Dict, Any, Set
import re


# ============================================================================
# Query Classification
# ============================================================================


def classify_query(query: str) -> Dict[str, Any]:
    """
    Classify query into categories using regex pattern matching.
    
    Returns:
        {
            'primary_type': str,  # math, current_data, factual, complex_reasoning, creative, general
            'patterns': List[str]  # detected patterns
        }
    """
    
    patterns = []
    
    # Math patterns
    if re.search(r'\d+\s*[\+\-\*/]\s*\d+', query):
        patterns.append('arithmetic_expression')
    if re.search(r'\d+%|percent|calculate|compute|solve|what.?s \d+', query, re.I):
        patterns.append('math_question')
    
    # Current data patterns
    if re.search(r'\b(today|now|current|latest|right now)\b', query, re.I):
        patterns.append('current_data')
    if re.search(r'\b(weather|temperature)\b', query, re.I):
        patterns.append('weather')
    if re.search(r'\b(exchange|currency|convert.*to|rate.*to)\b', query, re.I):
        patterns.append('price_data')
    
    # Factual patterns
    if re.search(r'^(what is|who is|who was|when did|where is|how many|define)', query, re.I):
        patterns.append('factual_question')
    if re.search(r'\b(capital|population|born|died|invented|president|elected)\b', query, re.I):
        patterns.append('factual_lookup')
    
    # Reasoning patterns
    if re.search(r'\b(compare|analyze|evaluate|assess|pros and cons)\b', query, re.I):
        patterns.append('complex_reasoning')
    if query.count('?') > 1:
        patterns.append('multi_question')
    if _detect_multi_step(query):
        patterns.append('multi_step')
    
    # Creative patterns
    if re.search(r'\b(write|create|generate|make me a|compose)\b', query, re.I):
        patterns.append('creative_generation')
    
    # Determine primary type
    if 'arithmetic_expression' in patterns or 'math_question' in patterns:
        primary_type = 'math'
    elif 'current_data' in patterns or 'weather' in patterns or 'price_data' in patterns:
        primary_type = 'current_data'
    elif 'factual_question' in patterns or 'factual_lookup' in patterns:
        primary_type = 'factual'
    elif 'complex_reasoning' in patterns or 'multi_step' in patterns:
        primary_type = 'complex_reasoning'
    elif 'creative_generation' in patterns:
        primary_type = 'creative'
    else:
        primary_type = 'general'
    
    return {
        'primary_type': primary_type,
        'patterns': patterns
    }


def _detect_multi_step(text: str) -> bool:
    """Detect if query requires multi-step reasoning."""
    step_markers = ['then', 'after', 'next', 'finally', 'first', 'second', 'third']
    step_count = sum(1 for marker in step_markers if marker in text.lower())
    return step_count >= 2


def _extract_math_expression(query: str) -> str:
    """Extract clean math expression from natural language query."""
    # Remove common question words
    clean = re.sub(r"^(what is|what's|whats|calculate|compute|solve|find|tell me)\s+", "", query, flags=re.I)
    # Remove "of" constructs but keep percentage syntax
    # "15% of 80" stays as "15% of 80" (calc understands this)
    return clean.strip()


# ============================================================================
# Resource Checking
# ============================================================================


def check_resources(attached_kbs: Set[str], kb_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Check what resources are currently available.
    
    Returns:
        {
            'kbs_attached': List[str],  # Currently attached filesystem KBs
            'kbs_available': List[str],  # All available filesystem KBs
            'has_kbs': bool  # Whether any KBs are attached
        }
    """
    
    return {
        'kbs_attached': sorted(list(attached_kbs)),
        'kbs_available': sorted(list(kb_paths.keys())),
        'has_kbs': len(attached_kbs) > 0
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
    
    Args:
        query: User's query
        attached_kbs: Set of currently attached filesystem KBs
        kb_paths: Dict of available KB names to paths
        vault_kb_name: Name of Qdrant vault (for reference, not attachment)
    
    Returns:
        List of recommendations, each with:
        - rank: str (A, B, C, D, etc.)
        - tool: str (tool/command name)
        - confidence: str (HIGH, MEDIUM, LOW)
        - reason: str (why this is recommended)
        - command: str (exact command to run)
    """
    
    classification = classify_query(query)
    resources = check_resources(attached_kbs, kb_paths)
    
    recommendations = []
    primary_type = classification['primary_type']
    
    # ===== MATH QUERIES =====
    if primary_type == 'math':
        clean_expr = _extract_math_expression(query)
        recommendations.append({
            'rank': 'A',
            'tool': '>>calc',
            'confidence': 'HIGH',
            'reason': 'Deterministic calculation - guaranteed accuracy',
            'command': f'>>calc {clean_expr}'
        })
        recommendations.append({
            'rank': 'B',
            'tool': 'serious mode',
            'confidence': 'LOW',
            'reason': 'Model estimation - unreliable for math',
            'command': query
        })
    
    # ===== CURRENT DATA QUERIES =====
    elif primary_type == 'current_data':
        rank_counter = ord('A')
        
        if 'weather' in classification['patterns']:
            recommendations.append({
                'rank': chr(rank_counter),
                'tool': '>>weather',
                'confidence': 'HIGH',
                'reason': 'Live weather data from external API',
                'command': f'>>weather {query}'
            })
            rank_counter += 1
        
        if 'price_data' in classification['patterns']:
            recommendations.append({
                'rank': chr(rank_counter),
                'tool': '>>exchange',
                'confidence': 'HIGH',
                'reason': 'Live currency exchange data',
                'command': f'>>exchange {query}'
            })
            rank_counter += 1
        
        # Fallback to model if no specific sidecar matched
        if not recommendations:
            recommendations.append({
                'rank': 'A',
                'tool': 'serious mode',
                'confidence': 'LOW',
                'reason': 'Model knowledge - may be outdated for current data',
                'command': query
            })
    
    # ===== FACTUAL QUERIES =====
    elif primary_type == 'factual':
        # Option A: Always recommend Mentats (searches Qdrant vault)
        recommendations.append({
            'rank': 'A',
            'tool': '##mentats',
            'confidence': 'HIGH',
            'reason': f'Multi-pass verified reasoning using {vault_kb_name} (Qdrant)',
            'command': f'##mentats {query}'
        })
        
        # Option B: If KBs attached, use them
        if resources['has_kbs']:
            kb_list = ', '.join(resources['kbs_attached'])
            recommendations.append({
                'rank': 'B',
                'tool': 'serious mode (with KBs)',
                'confidence': 'MEDIUM',
                'reason': f'Search attached KBs: {kb_list}',
                'command': query
            })
        else:
            # No KBs attached - suggest attaching them
            kb_available = ', '.join(resources['kbs_available']) if resources['kbs_available'] else 'none'
            recommendations.append({
                'rank': 'B',
                'tool': '>>attach all',
                'confidence': 'MEDIUM',
                'reason': f'Attach all KBs ({kb_available}) and run query',
                'command': f'>>attach all'
            })
        
        # Option C: Model knowledge only (least reliable)
        recommendations.append({
            'rank': 'C',
            'tool': 'serious mode (no grounding)',
            'confidence': 'LOW',
            'reason': 'Model knowledge only - may hallucinate facts',
            'command': query
        })
    
    # ===== COMPLEX REASONING QUERIES =====
    elif primary_type == 'complex_reasoning':
        # Option A: Mentats for verified reasoning
        recommendations.append({
            'rank': 'A',
            'tool': '##mentats',
            'confidence': 'HIGH',
            'reason': f'Multi-pass verified reasoning using {vault_kb_name} (Qdrant)',
            'command': f'##mentats {query}'
        })
        
        # Option B: If KBs attached, use serious mode with them
        if resources['has_kbs']:
            kb_list = ', '.join(resources['kbs_attached'])
            recommendations.append({
                'rank': 'B',
                'tool': 'serious mode (with KBs)',
                'confidence': 'MEDIUM',
                'reason': f'Single-pass reasoning with KBs: {kb_list}',
                'command': query
            })
        else:
            # No KBs - suggest attaching
            recommendations.append({
                'rank': 'B',
                'tool': '>>attach all',
                'confidence': 'MEDIUM',
                'reason': 'Attach all KBs and run query for grounded reasoning',
                'command': '>>attach all'
            })
        
        # Option C: Model knowledge only
        recommendations.append({
            'rank': 'C',
            'tool': 'serious mode (no grounding)',
            'confidence': 'LOW',
            'reason': 'Single-pass reasoning from model knowledge only',
            'command': query
        })
    
    # ===== CREATIVE QUERIES =====
    elif primary_type == 'creative':
        recommendations.append({
            'rank': 'A',
            'tool': '>>fun',
            'confidence': 'HIGH',
            'reason': 'Creative generation with style',
            'command': f'>>fun\n{query}'
        })
        recommendations.append({
            'rank': 'B',
            'tool': 'serious mode',
            'confidence': 'MEDIUM',
            'reason': 'Plain creative generation',
            'command': query
        })
    
    # ===== GENERAL QUERIES =====
    else:
        recommendations.append({
            'rank': 'A',
            'tool': 'serious mode',
            'confidence': 'MEDIUM',
            'reason': 'Default reasoning pipeline',
            'command': query
        })
        
        # If KBs available, suggest attaching
        if resources['kbs_available'] and not resources['has_kbs']:
            kb_list = ', '.join(resources['kbs_available'])
            recommendations.append({
                'rank': 'B',
                'tool': '>>attach all',
                'confidence': 'MEDIUM',
                'reason': f'Attach KBs ({kb_list}) and run query for grounded answers',
                'command': '>>attach all'
            })
    
    return recommendations


# ============================================================================
# Output Formatting
# ============================================================================


def format_recommendations(recommendations: List[Dict[str, str]], query: str = "") -> str:
    """
    Format recommendations for user display.
    
    Args:
        recommendations: List of recommendation dicts
        query: Original query (optional, for context)
    
    Returns:
        Formatted string for user
    """
    
    if not recommendations:
        return "No recommendations available."
    
    lines = []
    
    # Header
    if query:
        lines.append(f"Query: {query}")
        lines.append("")
    
    lines.append("**Recommended Tools:**")
    lines.append("")
    
    # Each recommendation
    for rec in recommendations:
        rank = rec['rank']
        tool = rec['tool']
        confidence = rec['confidence']
        reason = rec['reason']
        command = rec['command']
        
        lines.append(f"{rank}) **{tool}** (confidence: {confidence})")
        lines.append(f"   {reason}")
        lines.append(f"   Command: `{command}`")
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
    """
    Main entry point for >>trust command.
    
    Args:
        query: User's query
        attached_kbs: Set of currently attached KBs
        kb_paths: Dict of available KBs
        vault_kb_name: Name of Qdrant vault
    
    Returns:
        Formatted recommendation string
    """
    
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
    # Test cases
    print("=== Testing trust_pipeline.py ===\n")
    
    # Mock data
    mock_kbs = set()
    mock_kb_paths = {'amiga': '/path/to/amiga', 'c64': '/path/to/c64', 'dogs': '/path/to/dogs'}
    
    # Test 1: Math query
    print("TEST 1: Math query")
    print(handle_trust_command("What's 15% of 80?", mock_kbs, mock_kb_paths))
    print("\n" + "="*80 + "\n")
    
    # Test 2: Factual query (no KBs attached)
    print("TEST 2: Factual query (no KBs)")
    print(handle_trust_command("What's the capital of France?", mock_kbs, mock_kb_paths))
    print("\n" + "="*80 + "\n")
    
    # Test 3: Factual query (with KBs attached)
    print("TEST 3: Factual query (with KBs)")
    mock_kbs_attached = {'amiga', 'c64'}
    print(handle_trust_command("What software was pivotal for Amiga?", mock_kbs_attached, mock_kb_paths))
    print("\n" + "="*80 + "\n")
    
    # Test 4: Complex reasoning
    print("TEST 4: Complex reasoning")
    print(handle_trust_command("Compare microservices vs monolithic architecture", mock_kbs, mock_kb_paths))
    print("\n" + "="*80 + "\n")
    
    # Test 5: Weather query
    print("TEST 5: Weather query")
    print(handle_trust_command("What's the weather in Perth?", mock_kbs, mock_kb_paths))
