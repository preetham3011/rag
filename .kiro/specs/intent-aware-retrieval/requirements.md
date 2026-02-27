# Intent-Aware Retrieval Requirements

## Feature Overview
Implement an intent-aware retrieval system that enhances baseline FAISS similarity search by applying heuristic scoring based on detected query intent. This module will re-rank initial retrieval results to better match user information needs.

## User Stories

### US-1: Intent-Based Result Re-ranking
As a RAG system user, I want retrieval results to be re-ranked based on my query intent, so that the most relevant chunks appear first even if they have slightly lower embedding similarity.

### US-2: Heuristic Scoring for Different Intents
As a researcher, I want the system to boost results containing specific patterns (numbers, code, comparisons) based on my query type, so that I get contextually appropriate answers.

### US-3: Explainable Scoring
As a developer, I want the scoring mechanism to be simple and transparent, so that I can understand and debug why certain results rank higher.

## Acceptance Criteria

### 1. Core Retrieval Functionality

#### 1.1 Initial FAISS Retrieval
- The module retrieves top_k * 2 initial candidates from FAISS index
- Uses query_embedding as input vector
- Returns results with similarity scores (negative L2 distance)

#### 1.2 Intent-Aware Re-ranking
- Applies intent-specific heuristic bonuses to initial results
- Computes final_score = similarity_score + intent_bonus
- Re-sorts results by final_score in descending order
- Returns top_k results after re-ranking

#### 1.3 Output Format
- Returns list of dictionaries with structure:
  - text: str (chunk content)
  - metadata: dict (section, page, source, etc.)
  - score: float (final re-ranked score)
  - similarity_score: float (original FAISS score)
  - intent_bonus: float (applied bonus)

### 2. Intent Heuristics

#### 2.1 RESULT Intent
- Bonus applied if section metadata contains "Results"
- Bonus applied if text contains numbers (digits)
- Bonus applied if text contains percentage symbols "%"
- Bonus range: +0.1 to +0.3

#### 2.2 METHOD Intent
- Bonus applied if section metadata contains "Method"
- Bonus applied if text contains keywords: "algorithm", "pipeline", "step"
- Bonus range: +0.1 to +0.3

#### 2.3 API_USAGE Intent
- Bonus applied if text contains code-like symbols: "(", ")", "=", ":"
- Higher bonus for multiple code symbols present
- Bonus range: +0.1 to +0.3

#### 2.4 DEFINITION Intent
- Bonus applied if section is "Abstract" or "Introduction"
- Bonus applied for early sections in document
- Bonus range: +0.1 to +0.3

#### 2.5 COMPARISON Intent
- Bonus applied if text contains comparison keywords: "compare", "difference", "versus", "vs"
- Case-insensitive matching
- Bonus range: +0.1 to +0.3

#### 2.6 Unknown Intent Handling
- If intent is not recognized or confidence is low, apply no bonus
- Fall back to pure similarity-based ranking

### 3. Technical Constraints

#### 3.1 Implementation Style
- No classes (functional programming style)
- No machine learning models
- No external libraries beyond standard Python
- Modular functions with clear responsibilities

#### 3.2 Code Quality
- All functions have docstrings
- Inline comments for complex logic
- Type hints where appropriate
- Readable variable names

#### 3.3 Testing
- Include test code under `if __name__ == "__main__":`
- Test with mock FAISS index and metadata
- Demonstrate each intent type
- Verify re-ranking behavior

### 4. Integration Requirements

#### 4.1 Input Compatibility
- Accepts intent_info dict from intent_detector.py
- Works with FAISS index format used in vector_store.py
- Compatible with metadata structure from chunker.py

#### 4.2 Performance
- Re-ranking should be fast (< 100ms for 100 candidates)
- No redundant computations
- Efficient string matching

## Non-Functional Requirements

### NFR-1: Maintainability
- Code should be easy to modify when adding new intent types
- Heuristic rules should be clearly separated and documented

### NFR-2: Debuggability
- Scoring breakdown should be traceable
- Each bonus component should be identifiable in output

### NFR-3: Extensibility
- Easy to add new intent types
- Easy to adjust bonus weights
- Easy to add new heuristic rules

## Out of Scope
- Machine learning-based re-ranking
- Cross-encoder models
- Query expansion
- Feedback loops
- A/B testing framework
- Configuration file management (hardcoded bonuses acceptable)

## Dependencies
- src/retrieval/intent_detector.py (provides intent_info)
- src/indexing/vector_store.py (provides FAISS index)
- src/indexing/chunker.py (defines metadata structure)
- numpy (for FAISS operations)
- faiss-cpu (for vector search)

## Success Metrics
- Module successfully re-ranks results based on intent
- All intent types have measurable impact on ranking
- Code passes basic functional tests
- Integration with existing RAG pipeline works without errors
