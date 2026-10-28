-- Paper Feed SQLite Schema (Phase 1.2 / docs/plan.md §1.6)

PRAGMA foreign_keys = ON;

-- 1. Papers table
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,  -- JSON list of {id, display_name, position}
    year INTEGER NOT NULL,
    publication_date TEXT,
    venue TEXT,
    doi TEXT,
    url TEXT,
    citation_count_T INTEGER DEFAULT 0,
    primary_topic_id TEXT,
    subfield_id TEXT,
    field_id TEXT,
    in_candidates BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_primary_topic ON papers(primary_topic_id);
CREATE INDEX IF NOT EXISTS idx_papers_in_candidates ON papers(in_candidates);

-- 2. Time-bounded citation edges
CREATE TABLE IF NOT EXISTS paper_citations (
    source_paper_id TEXT NOT NULL,
    target_paper_id TEXT NOT NULL,
    source_year INTEGER NOT NULL,
    PRIMARY KEY (source_paper_id, target_paper_id)
);

CREATE INDEX IF NOT EXISTS idx_citations_source ON paper_citations(source_paper_id);
CREATE INDEX IF NOT EXISTS idx_citations_target ON paper_citations(target_paper_id);
CREATE INDEX IF NOT EXISTS idx_citations_year ON paper_citations(source_year);

-- 3. Extracted scientific facets
CREATE TABLE IF NOT EXISTS paper_facets (
    facet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    facet_type TEXT NOT NULL, -- task, method, dataset, metric, topic
    facet_value TEXT NOT NULL,
    canonical_id TEXT,
    evidence_sentence TEXT,
    start_char INTEGER,
    end_char INTEGER,
    confidence REAL,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_facets_paper ON paper_facets(paper_id);
CREATE INDEX IF NOT EXISTS idx_facets_type ON paper_facets(facet_type);
CREATE INDEX IF NOT EXISTS idx_facets_canon ON paper_facets(canonical_id);

-- 4. Facet canonicalization cluster mapping
CREATE TABLE IF NOT EXISTS facet_canon (
    canonical_id TEXT PRIMARY KEY,
    facet_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_canon_type ON facet_canon(facet_type);

-- 5. User management
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    persona_vector BLOB,
    preferred_facets TEXT, -- JSON
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_vector BLOB,
    last_active_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    action_type TEXT NOT NULL, -- click, save, skip
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_interactions_user ON user_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_session ON user_interactions(session_id);

