# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Google Gemini Context Caching Optimization** - 85% reduction in LLM fallback latency
  - Implemented Google Gemini explicit context caching for metadata.json
  - Pre-caches document metadata to eliminate 500KB-2MB payload on every LLM fallback
  - Reduces LLM fallback latency from ~20s to ~2-3s (85% improvement)
  - Automatic cache management with 1-hour TTL and metadata change detection
  - Graceful fallback to traditional approach if caching fails
  - New API endpoints:
    - `GET /cache/status` - Monitor cache status and expiration
    - `POST /cache/refresh` - Force refresh cache with updated metadata
  - Comprehensive test suite for cache optimization functionality

- Multi-company support for embedding-based document selection
  - `semantic_document_selection` now extracts companies from queries using `get_companies_from_query`
  - Runs individual searches for each detected company with company-specific filters
  - Combines and deduplicates results from multiple companies
  - Falls back to broader search (n_results=15) when no companies are detected
  - Comprehensive test coverage for multi-company scenarios

### Fixed
- Fixed single-clause `$and` filter crash in ChromaDB queries ([#4](https://github.com/Aeontsolutions/jse-datasphere-chatbot/issues/4))
  - ChromaDB rejects `$and` filters with fewer than two expressions
  - Updated `query_collection` function to handle filter clauses correctly:
    - Zero clauses: `where_filter = None`
    - One clause: pass the single filter directly (no `$and` wrapper)  
    - Two+ clauses: wrap in `{"$and": [...]}`
  - Added comprehensive unit tests covering all three scenarios
  - Endpoints now return 200 status for queries with single metadata filters
- Fixed embedding-based document selection to return documents for all companies mentioned in multi-company queries
  - Previously only returned documents for the highest-scoring company
  - Now ensures `/fast_chat` endpoint can compare multiple companies in a single query

### Added
- Unit test suite for `chroma_utils.query_collection` function
- pytest testing framework support
- Test coverage for ChromaDB filter logic edge cases
- Unit tests for multi-company query scenarios
- Test coverage for document deduplication logic

---

## [Previous Releases]

_This changelog was introduced with the single-clause filter fix. Previous changes were not documented._