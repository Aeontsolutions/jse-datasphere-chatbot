# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed single-clause `$and` filter crash in ChromaDB queries ([#4](https://github.com/Aeontsolutions/jse-datasphere-chatbot/issues/4))
  - ChromaDB rejects `$and` filters with fewer than two expressions
  - Updated `query_collection` function to handle filter clauses correctly:
    - Zero clauses: `where_filter = None`
    - One clause: pass the single filter directly (no `$and` wrapper)  
    - Two+ clauses: wrap in `{"$and": [...]}`
  - Added comprehensive unit tests covering all three scenarios
  - Endpoints now return 200 status for queries with single metadata filters

### Added
- Unit test suite for `chroma_utils.query_collection` function
- pytest testing framework support
- Test coverage for ChromaDB filter logic edge cases

---

## [Previous Releases]

_This changelog was introduced with the single-clause filter fix. Previous changes were not documented._