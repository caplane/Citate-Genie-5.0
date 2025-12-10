"""
processors/ - Document-level processing pipelines.

Modules:
    word_document.py        - Read/write Word footnotes and endnotes
    author_date.py          - Full pipeline for author-year citation documents
    author_year_extractor.py - Parse "(Smith, 2020)" patterns from text
"""

from processors.word_document import WordDocumentProcessor, process_document
from processors.author_date import process_author_date_document

__all__ = [
    'WordDocumentProcessor',
    'process_document',
    'process_author_date_document',
]
