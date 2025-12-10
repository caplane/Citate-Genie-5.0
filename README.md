# CitateGenie 2.0

**Transform hand-typed citations into properly formatted references.**

CitateGenie processes Word documents containing citations, looks up metadata from authoritative sources (Crossref, PubMed, Google Books, CourtListener), and reformats citations in your chosen style.

## Features

- **Notes-Bibliography Pipeline**: Process footnotes/endnotes (Chicago, Bluebook)
- **Author-Date Pipeline**: Process in-text citations like "(Smith, 2020)" (APA, Harvard)
- **Multiple Styles**: Chicago, APA, MLA, Bluebook, OSCOLA
- **Smart Detection**: Automatically identifies citation types (legal, book, journal, newspaper)
- **Tiered AI Fallback**: Free APIs → GPT-4o → Claude for difficult citations

## Project Structure

```
citeflex/
├── app.py                 # Flask application entry point
├── config.py              # API keys, timeouts, domain mappings
├── models.py              # CitationMetadata dataclass, CitationType enum
│
├── engines/               # Search engines for metadata retrieval
│   ├── academic.py        # Crossref, OpenAlex, Semantic Scholar, PubMed
│   ├── books.py           # Google Books, Open Library
│   ├── legal.py           # CourtListener, Famous Cases cache
│   ├── author_year_search.py  # Multi-engine author+year search
│   └── ...
│
├── routers/               # Citation routing logic
│   ├── unified.py         # Main routing: detect → search → format
│   ├── url.py             # URL-specific handling
│   ├── claude.py          # Claude API fallback
│   ├── gemini.py          # Gemini API fallback
│   └── openai.py          # GPT-4o API (cheaper fallback tier)
│
├── processors/            # Document processing pipelines
│   ├── word_document.py   # Read/write Word footnotes and endnotes
│   ├── author_date.py     # Full pipeline for APA/Harvard documents
│   └── author_year_extractor.py  # Parse "(Author, Year)" from text
│
├── formatters/            # Output formatters for citation styles
│   ├── base.py            # BaseFormatter, get_formatter()
│   ├── chicago.py         # Chicago 17th ed.
│   ├── apa.py             # APA 7th ed.
│   ├── mla.py             # MLA 9th ed.
│   └── legal.py           # Bluebook, OSCOLA
│
├── utils/                 # Shared helper functions
│   ├── type_detection.py  # Detect citation types
│   └── metadata_extraction.py  # Extract metadata from API responses
│
├── templates/
│   └── index.html         # Frontend UI
│
└── tests/
    ├── stress_test_notes.py       # Notes-bibliography tests
    └── stress_test_author_date.py # Author-date tests
```

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

```bash
# Required for full functionality
ANTHROPIC_API_KEY=       # Claude API
OPENAI_API_KEY=          # GPT-4o API (cheaper fallback)
GOOGLE_API_KEY=          # Gemini API

# Optional (for enhanced search)
COURTLISTENER_API_KEY=   # Legal citations
SERPAPI_KEY=             # Google Scholar
PUBMED_API_KEY=          # Medical citations
```

## Running

```bash
# Development
python app.py

# Production (Railway)
gunicorn app:app --workers 2 --threads 4 --timeout 120
```

## Usage

1. Upload a Word document with citations
2. Select citation style (Chicago, APA, MLA, Bluebook, OSCOLA)
3. Choose document type (Notes or Author-Date)
4. Process and download the formatted document

## API Endpoints

- `POST /api/cite` - Single citation lookup
- `POST /api/process` - Process Word document (notes-bibliography)
- `POST /api/process-author-date` - Process Word document (author-date)
- `GET /api/download/<session_id>` - Download processed document

## License

MIT
