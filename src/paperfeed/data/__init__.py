"""Paper Feed data subpackage."""

from paperfeed.data.cohort import (
    CandidateAuthor,
    CohortMiner,
    classify_history_slice,
    clean_id,
    extract_first_author,
    filter_author_plausibility,
    load_candidate_authors,
    save_candidate_authors,
    select_target_paper_b,
)
from paperfeed.data.crawl import CorpusCrawler
from paperfeed.data.db import DatabaseManager

__all__ = [
    "DatabaseManager",
    "CandidateAuthor",
    "CohortMiner",
    "CorpusCrawler",
    "classify_history_slice",
    "clean_id",
    "extract_first_author",
    "filter_author_plausibility",
    "load_candidate_authors",
    "save_candidate_authors",
    "select_target_paper_b",
]
