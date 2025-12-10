"""
routers/ - Decision logic for routing citations to appropriate engines.

Modules:
    unified.py  - Main routing: detect type → search → format
    url.py      - URL-specific handling: news, gov, academic sites
    claude.py   - Claude API for ambiguous citations
    gemini.py   - Gemini API for metadata extraction
    openai.py   - GPT-4o API (cheaper AI fallback tier)
"""

from routers.unified import get_citation, get_multiple_citations

__all__ = [
    'get_citation',
    'get_multiple_citations',
]
