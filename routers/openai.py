"""
routers/openai.py

GPT-4o router for citation lookup.

GPT-4o is used as an intermediate AI tier between free engines and Claude Opus.
Cost: ~$2.50/$10 per 1M tokens (input/output) vs Claude Opus at $15/$75.

This provides ~7x cost savings while still achieving high accuracy for 
well-known academic citations.
"""

import os
import json
from typing import Optional, Dict, Any

from models import CitationMetadata, CitationType
from config import OPENAI_API_KEY


# API Configuration
GPT_MODEL = "gpt-4o"


def lookup_citation(
    author: str,
    year: str,
    second_author: Optional[str] = None,
    third_author: Optional[str] = None,
    context: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Use GPT-4o to identify a citation from author(s) and year.
    
    Args:
        author: Primary author surname
        year: Publication year
        second_author: Optional second author
        third_author: Optional third author
        context: Optional field/topic context
        
    Returns:
        Dict with citation data, or None if not found/low confidence
    """
    if not OPENAI_API_KEY:
        print("[OpenAI Router] No OPENAI_API_KEY set")
        return None
    
    try:
        import requests
        
        # Build author string
        authors_str = author
        if second_author:
            authors_str += f", {second_author}"
        if third_author:
            authors_str += f", & {third_author}"
        citation_ref = f"{authors_str} ({year})"
        
        context_hint = ""
        if context:
            context_hint = f"\n\nThis citation appears in a document about {context}."
        
        prompt = f"""Identify this academic citation reference: {citation_ref}{context_hint}

Return a JSON object with these fields:
- title: full title of the work
- authors: array of author names (full names, not just surnames)
- year: publication year
- type: "journal", "book", or "newspaper"
- journal: journal name (if journal article)
- volume: volume number (if applicable)
- issue: issue number (if applicable)
- pages: page range (if applicable)
- publisher: publisher name (if book)
- doi: DOI if you know it
- confidence: your confidence 0.0 to 1.0

Only respond with valid JSON, no other text."""

        print(f"[OpenAI Router] Looking up: {authors_str} ({year})")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GPT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a scholarly citation expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            },
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"[OpenAI Router] API error: {response.status_code}")
            return None
        
        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Parse JSON from response
        content = content.strip()
        if content.startswith('```'):
            content = content.split('\n', 1)[1] if '\n' in content else content
        if content.endswith('```'):
            content = content.rsplit('```', 1)[0]
        content = content.strip()
        if content.startswith('json'):
            content = content[4:].strip()
        
        result = json.loads(content)
        
        if result.get('confidence', 0) < 0.5:
            print(f"[OpenAI Router] Low confidence: {result.get('confidence', 0)}")
            return None
        
        print(f"[OpenAI Router] Found: {result.get('title', '')[:50]}...")
        return result
        
    except Exception as e:
        print(f"[OpenAI Router] Error: {e}")
        return None


def to_metadata(data: Dict[str, Any]) -> Optional[CitationMetadata]:
    """Convert GPT-4o response to CitationMetadata."""
    if not data or not data.get('title'):
        return None
    
    type_map = {
        'journal': CitationType.JOURNAL,
        'book': CitationType.BOOK,
        'newspaper': CitationType.NEWSPAPER,
        'medical': CitationType.MEDICAL,
    }
    
    return CitationMetadata(
        citation_type=type_map.get(data.get('type', 'journal'), CitationType.JOURNAL),
        title=data.get('title', ''),
        authors=data.get('authors', []),
        year=data.get('year', ''),
        journal=data.get('journal', ''),
        volume=data.get('volume', ''),
        issue=data.get('issue', ''),
        pages=data.get('pages', ''),
        publisher=data.get('publisher', ''),
        doi=data.get('doi', ''),
        source_engine="GPT-4o",
        confidence=data.get('confidence', 0.7)
    )


def search_citation(
    author: str,
    year: str,
    second_author: Optional[str] = None,
    third_author: Optional[str] = None,
    context: Optional[str] = None
) -> Optional[CitationMetadata]:
    """
    Search for a citation using GPT-4o and return CitationMetadata.
    
    Convenience function that combines lookup_citation() and to_metadata().
    """
    data = lookup_citation(author, year, second_author, third_author, context)
    return to_metadata(data) if data else None
