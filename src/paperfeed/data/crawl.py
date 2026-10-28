"""Corpus expansion and time-bounded crawler (Phase 1, Step 1.4).

Governed by docs/prob.md §4 and docs/design.md §4.1.
Implements:
1. Multi-tier crawling:
   - Tier 1: Author-First Ego Networks (Hop 1 references + citations <= T, Hop 2 references <= per_author_cap).
   - Tier 2: Foundational CS/AI/ML/IR Anchors (top-cited classic papers <= T).
   - Tier 3: Thematic CS/AI/ML/IR Fill (up to target corpus size).
2. Strict Temporal & Leakage Integrity:
   - Only works with publication_year <= cutoff_year (2024) are ingested as candidates.
   - Target papers B (2025) are strictly prohibited from being crawled or added.
3. Resumable Checkpointing:
   - SQLite batch insertion via DatabaseManager and disk progress tracking.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from paperfeed.common.config import find_project_root, load_config
from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import CandidateAuthor, clean_id, load_candidate_authors
from paperfeed.data.db import DatabaseManager
from paperfeed.openalex.client import OpenAlexClient

logger = get_logger("data.crawl")

ALLOWED_TYPES = {
    "article",
    "conference-paper",
    "proceedings-article",
    "proceedings",
    "preprint",
    "book-chapter",
    "review",
    "book",
    "dissertation",
}


ALL_STAGES = ["authored", "hop1", "hop2", "anchors", "fill"]


class CorpusCrawler:
    """Manages the author-first, time-bounded corpus expansion pipeline."""

    def __init__(
        self,
        client: OpenAlexClient,
        db: DatabaseManager,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[str | Path] = None,
    ) -> None:
        self.client = client
        self.db = db
        self.cfg = config or load_config()

        project_cfg = self.cfg.get("project", {})
        self.cutoff_year: int = project_cfg.get("cutoff_year", 2024)
        self.target_year: int = project_cfg.get("target_year", 2025)

        corpus_cfg = self.cfg.get("corpus", {})
        self.target_size: int = corpus_cfg.get("target_size", 120000)
        self.hop2_per_author_cap: int = corpus_cfg.get("hop2_per_author_cap", 150)
        self.foundational_anchor_count: int = corpus_cfg.get("foundational_anchor_count", 15000)
        # Hop-1 (inbound citations, feeds co-citation CC): fair per-author budget
        self.hop1_seeds_per_author: int = corpus_cfg.get("hop1_seeds_per_author", 5)
        self.hop1_max_citing_per_paper: int = corpus_cfg.get("hop1_max_citing_per_paper", 40)
        self.hop1_min_citations: int = corpus_cfg.get("hop1_min_citations", 1)
        self.hop1_max_citations: int = corpus_cfg.get("hop1_max_citations", 300)
        self.hop1_total_new_cap: int = corpus_cfg.get("hop1_total_new_cap", 40000)
        self.focus_subfields: List[str] = corpus_cfg.get(
            "focus_subfields", ["subfields/1702", "subfields/1710"]
        )
        self.focus_topics: List[str] = corpus_cfg.get(
            "focus_topics",
            [
                "T10286",  # IR & Search
                "T10203",  # RecSys
                "T14435",  # IR & Data Mining
                "T12072",  # ML & Algorithms
                "T12535",  # ML & Data Classification
                "T10320",  # Neural Networks
                "T10028",  # NLP & Topic Modeling
                "T12026",  # Explainable AI (XAI)
                "T13904",  # AI Applications
            ],
        )

        root = find_project_root()
        ckpt_path = checkpoint_dir or "data/interim/crawl_checkpoints"
        self.checkpoint_dir = Path(ckpt_path)
        if not self.checkpoint_dir.is_absolute():
            self.checkpoint_dir = root / self.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set of forbidden target paper B IDs to prevent accidental leakage
        self.forbidden_paper_b_ids: Set[str] = set()

    def set_forbidden_paper_b_ids(self, ids: Set[str]) -> None:
        """Register future target paper B IDs to strictly block them from the corpus."""
        self.forbidden_paper_b_ids = {clean_id(i) for i in ids if i}
        logger.info(f"Registered {len(self.forbidden_paper_b_ids)} forbidden target paper B IDs.")

    def is_valid_work(self, work: Dict[str, Any], enforce_type_lang: bool = True) -> bool:
        """Check temporal bounds, publication type, language, and target paper B exclusion.

        ``enforce_type_lang=False`` is used only for the evaluation authors' own papers: their
        reference lists must always feed Citations(u, <=T), whatever the publication type.
        """
        if not work:
            return False

        work_id = clean_id(work.get("id", ""))
        if not work_id:
            return False

        # Leakage guard 1: Never ingest target paper B
        if work_id in self.forbidden_paper_b_ids:
            logger.warning(f"Blocked forbidden target paper B ({work_id}) from candidate corpus.")
            return False

        # Leakage guard 2: Strictly enforce year <= cutoff_year (2024)
        year = work.get("publication_year")
        if year is not None and year > self.cutoff_year:
            return False

        pub_date = work.get("publication_date") or ""
        if pub_date and pub_date > f"{self.cutoff_year}-12-31":
            return False

        if not enforce_type_lang:
            return True

        # Type filter
        work_type = work.get("type", "")
        if work_type and work_type not in ALLOWED_TYPES:
            return False

        # Language filter (allow English or unstated)
        lang = work.get("language")
        if lang and lang != "en":
            return False

        return True

    def ingest_works_batch(
        self,
        works: List[Dict[str, Any]],
        authored_ids: Optional[Set[str]] = None,
    ) -> Tuple[int, int]:
        """Parse and insert a batch of OpenAlex works into SQLite database.

        Works whose ID is in ``authored_ids`` bypass the type / language filter. If they would have
        failed it, they are stored as graph nodes only (``in_candidates = 0``): their references
        still count towards Citations(u) and BC/CC, but they are never retrievable. Authored papers
        are always in the exclusion set, so this loses nothing for recommendation.
        """
        paper_rows: List[Dict[str, Any]] = []
        citation_edges: List[Tuple[str, str, int]] = []

        for w in works:
            is_authored = bool(authored_ids) and clean_id(w.get("id", "")) in authored_ids
            if not self.is_valid_work(w, enforce_type_lang=not is_authored):
                continue
            p_row, c_list = DatabaseManager.parse_openalex_work(w, cutoff_year=self.cutoff_year)
            if is_authored and not self.is_valid_work(w):
                p_row["in_candidates"] = 0
            paper_rows.append(p_row)
            citation_edges.extend(c_list)

        inserted_papers = self.db.insert_papers_batch(paper_rows)
        inserted_citations = self.db.insert_citations_batch(citation_edges)
        return inserted_papers, inserted_citations

    def crawl_authored_papers(self, candidate_authors: List[CandidateAuthor]) -> Set[str]:
        """
        Crawl and store all historical papers authored by the candidate authors <= T.
        First ingests from already cached raw author history files on local disk (instant),
        then fetches any remaining missing papers using batched OpenAlex requests.
        """
        logger.info(f"Crawling authored papers for {len(candidate_authors)} candidate authors.")
        all_authored_ids: Set[str] = set()
        for author in candidate_authors:
            for pid in author.authored_paper_ids_T:
                all_authored_ids.add(clean_id(pid))

        # 1. Fast local cache ingestion: load from existing author_history_*.json files
        cached_files = list(self.client.raw_cache_dir.glob("author_history_*.json"))
        logger.info(f"Scanning {len(cached_files)} local raw author history cache files...")
        batch: List[Dict[str, Any]] = []
        local_ingested = 0

        for fpath in cached_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                works = data.get("results", [])
                for w in works:
                    batch.append(w)
                    if len(batch) >= 200:
                        p_count, _ = self.ingest_works_batch(batch, authored_ids=all_authored_ids)
                        local_ingested += p_count
                        batch = []
            except Exception as e:
                logger.warning(f"Error reading cache file {fpath}: {e}")

        if batch:
            p_count, _ = self.ingest_works_batch(batch, authored_ids=all_authored_ids)
            local_ingested += p_count

        logger.info(f"Ingested {local_ingested} papers from local cache into database.")

        # 2. Check remaining missing authored papers in DB
        to_fetch = [pid for pid in all_authored_ids if not self.db.get_paper(pid)]
        logger.info(f"{len(to_fetch)} authored papers still need to be fetched via network.")

        if to_fetch:
            batch = []
            network_ingested = 0
            for work in self.client.fetch_works_batch(to_fetch, chunk_size=50):
                batch.append(work)
                if len(batch) >= 100:
                    p_count, _ = self.ingest_works_batch(batch, authored_ids=all_authored_ids)
                    network_ingested += p_count
                    batch = []
            if batch:
                p_count, _ = self.ingest_works_batch(batch, authored_ids=all_authored_ids)
                network_ingested += p_count
            logger.info(f"Fetched and ingested {network_ingested} papers via batched network calls.")

        total_in_db = self.db.count_papers()
        logger.info(f"Authored papers stage complete. Database now contains {total_in_db} papers.")
        return all_authored_ids

    def _select_hop1_seeds(self, candidate_authors: List[CandidateAuthor]) -> List[List[str]]:
        """Per author, the authored papers (<= T) used as inbound-citation seeds, best first.

        Papers outside [hop1_min_citations, hop1_max_citations] are skipped (no citers to find, or
        mega-papers that trigger OpenAlex 504 timeouts). Among the rest, the most-cited come first
        because they have the most citing papers, hence the most co-citation evidence.
        """
        seeds_by_author: List[List[str]] = []
        for author in candidate_authors:
            ids = [clean_id(pid) for pid in author.authored_paper_ids_T]
            rows: List[Dict[str, Any]] = []
            for i in range(0, len(ids), 500):
                rows.extend(self.db.get_papers_batch([f"https://openalex.org/{x}" for x in ids[i : i + 500]]))
            eligible = [
                r
                for r in rows
                if self.hop1_min_citations <= (r["citation_count_T"] or 0) <= self.hop1_max_citations
            ]
            eligible.sort(key=lambda r: (-(r["citation_count_T"] or 0), -(r["year"] or 0), r["paper_id"]))
            seeds_by_author.append([clean_id(r["paper_id"]) for r in eligible[: self.hop1_seeds_per_author]])
        return seeds_by_author

    def crawl_hop1_neighborhood(self, candidate_authors: List[CandidateAuthor]) -> Set[str]:
        """
        Hop 1 Expansion (inbound citations, needed for co-citation CC):
        - Outgoing: references of authored papers (stored when the papers are ingested).
        - Incoming: papers published <= T that cite an authored paper, up to
          ``hop1_max_citing_per_paper`` per seed paper and ``hop1_seeds_per_author`` seeds per author.
        Seeds are processed rank by rank (every author's best seed first, then every author's
        second seed, ...) so the global cap ``hop1_total_new_cap`` on newly added papers never starves
        the authors at the end of the list.
        """
        seeds_by_author = self._select_hop1_seeds(candidate_authors)
        n_seeds = sum(len(x) for x in seeds_by_author)
        logger.info(
            f"Hop-1 citation expansion: {n_seeds} seed papers across {len(candidate_authors)} authors "
            f"({sum(1 for x in seeds_by_author if not x)} authors have no eligible seed)."
        )

        hop1_ids: Set[str] = set()
        new_papers = 0
        for rank in range(self.hop1_seeds_per_author):
            for seeds in seeds_by_author:
                if rank >= len(seeds):
                    continue
                if new_papers >= self.hop1_total_new_cap:
                    logger.info(f"Hop-1 new-paper cap ({self.hop1_total_new_cap}) reached; stopping.")
                    return hop1_ids
                pid = seeds[rank]
                filter_str = f"cites:{pid},to_publication_date:{self.cutoff_year}-12-31"
                citing_works: List[Dict[str, Any]] = []
                try:
                    for work in self.client.paginate_works(
                        filter_str=filter_str,
                        cache_prefix=f"cites_{pid}",
                        max_results=self.hop1_max_citing_per_paper,
                    ):
                        citing_works.append(work)
                        hop1_ids.add(clean_id(work.get("id", "")))
                except Exception as e:
                    logger.warning(f"Skipping slow paper {pid}: {e}")

                if citing_works:
                    before = self.db.count_papers()
                    self.ingest_works_batch(citing_works)
                    new_papers += self.db.count_papers() - before

        logger.info(f"Hop-1 discovered {len(hop1_ids)} citing papers ({new_papers} new to the corpus).")
        return hop1_ids

    def crawl_hop2_neighborhood(
        self,
        authored_paper_ids: Set[str],
        per_author_cap: Optional[int] = None,
    ) -> Set[str]:
        """
        Hop 2 Expansion:
        - References of Hop 1 papers, fetched in high-speed batches (chunk_size=50).
        """
        cap = per_author_cap or self.hop2_per_author_cap
        total_cap = min(cap * 100, 15000)
        logger.info(f"Starting Hop-2 expansion (fetching up to {total_cap} referenced papers).")
        hop2_ids: Set[str] = set()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT target_paper_id FROM paper_citations LIMIT ?",
                (total_cap,),
            )
            rows = cursor.fetchall()

        target_refs = [clean_id(r[0]) for r in rows if not self.db.get_paper(clean_id(r[0]))]
        logger.info(f"Found {len(target_refs)} Hop-2 target references to fetch via batched network calls.")

        batch: List[Dict[str, Any]] = []
        for work in self.client.fetch_works_batch(target_refs, chunk_size=50):
            batch.append(work)
            hop2_ids.add(clean_id(work.get("id", "")))
            if len(batch) >= 100:
                self.ingest_works_batch(batch)
                batch = []

        if batch:
            self.ingest_works_batch(batch)

        logger.info(f"Hop-2 ingested {len(hop2_ids)} papers.")
        return hop2_ids

    def crawl_foundational_anchors(
        self,
        target_count: Optional[int] = None,
    ) -> Set[str]:
        """
        Crawl classic, highly-cited foundational CS/AI/ML/IR papers published <= cutoff_year.
        """
        limit = target_count or self.foundational_anchor_count
        logger.info(f"Crawling up to {limit} foundational classic papers <= {self.cutoff_year}.")
        anchor_ids: Set[str] = set()

        # Query top-cited papers across focus subfields and broad CS
        queries = [
            f"primary_topic.subfield.id:subfields/1702,to_publication_date:{self.cutoff_year}-12-31",  # AI
            f"primary_topic.subfield.id:subfields/1710,to_publication_date:{self.cutoff_year}-12-31",  # Info Systems
            f"primary_topic.field.id:fields/17,to_publication_date:{self.cutoff_year}-12-31",           # CS
        ]

        per_query_limit = limit // len(queries)
        for filter_str in queries:
            batch: List[Dict[str, Any]] = []
            try:
                # Sorted by citations so these really are the classics. The cache key hashes the whole
                # filter: a truncated prefix made the AI and Information-Systems queries share one cache
                # file, so the second query silently re-read the first one's results.
                key = hashlib.md5(filter_str.encode()).hexdigest()[:10]
                for work in self.client.paginate_works(
                    filter_str=filter_str,
                    cache_prefix=f"foundational_{key}",
                    max_results=per_query_limit,
                    sort="cited_by_count:desc",
                ):
                    batch.append(work)
                    anchor_ids.add(clean_id(work.get("id", "")))
                    if len(batch) >= 200:
                        self.ingest_works_batch(batch)
                        batch = []
            except Exception as e:
                logger.warning(f"Error querying foundational papers for filter '{filter_str}': {e}")

            if batch:
                self.ingest_works_batch(batch)

        logger.info(f"Ingested {len(anchor_ids)} foundational anchor papers.")
        return anchor_ids

    def crawl_thematic_fill(self, target_total: Optional[int] = None) -> int:
        """
        Fill the candidate corpus to target_size using topic-stratified sampling
        across our focus CS, AI, ML, and IR topics.
        """
        target = target_total or self.target_size
        current_count = self.db.count_papers()
        needed = target - current_count
        if needed <= 0:
            logger.info(f"Corpus already has {current_count} papers (target {target}). No fill needed.")
            return current_count

        logger.info(f"Thematic fill: currently {current_count} papers, crawling {needed} more to reach {target}.")

        per_topic_quota = max(needed // (len(self.focus_topics) + 1), 200)

        for topic_id in self.focus_topics:
            if self.db.count_papers() >= target:
                break

            filter_str = f"primary_topic.id:{topic_id},to_publication_date:{self.cutoff_year}-12-31"
            batch: List[Dict[str, Any]] = []
            try:
                for work in self.client.paginate_works(
                    filter_str=filter_str,
                    cache_prefix=f"fill_{topic_id}",
                    max_results=per_topic_quota,
                ):
                    batch.append(work)
                    if len(batch) >= 200:
                        self.ingest_works_batch(batch)
                        batch = []
                        if self.db.count_papers() >= target:
                            break
            except Exception as e:
                logger.warning(f"Error during thematic fill for topic {topic_id}: {e}")

            if batch:
                self.ingest_works_batch(batch)

        final_count = self.db.count_papers()
        logger.info(f"Thematic fill complete. Corpus now stands at {final_count} papers.")
        return final_count

    def run_pipeline(
        self,
        candidate_authors_path: str | Path,
        target_size: Optional[int] = None,
        stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the corpus expansion pipeline in sequence (``stages`` selects a subset of
        authored, hop1, hop2, anchors, fill; every stage is idempotent and resumable):
        1. Load candidate authors and register forbidden target paper B IDs.
        2. Ingest authored papers.
        3. Hop 1 expansion (references + citing works).
        4. Hop 2 expansion (capped references).
        5. Foundational anchors crawl.
        6. Thematic fill up to target_size.
        """
        target = target_size or self.target_size
        logger.info(f"Starting full corpus crawl pipeline to reach {target} papers.")

        candidates = load_candidate_authors(candidate_authors_path)
        paper_b_ids = {c.target_paper_b_id for c in candidates}
        self.set_forbidden_paper_b_ids(paper_b_ids)

        # Tier 1: Authored papers
        run = set(stages or ALL_STAGES)
        unknown = run - set(ALL_STAGES)
        if unknown:
            raise ValueError(f"Unknown crawl stages {sorted(unknown)}; choose from {ALL_STAGES}")

        # authored_paper_ids_T is read from the cohort file, so this set is needed even if skipped
        authored_ids = {clean_id(pid) for c in candidates for pid in c.authored_paper_ids_T}
        if "authored" in run:
            authored_ids = self.crawl_authored_papers(candidates)

        # Tier 1: Hop-1 expansion
        hop1_ids = self.crawl_hop1_neighborhood(candidates) if "hop1" in run else set()

        # Tier 1: Hop-2 expansion
        hop2_ids = self.crawl_hop2_neighborhood(authored_ids) if "hop2" in run else set()

        # Tier 2: Foundational classic anchors
        anchor_ids = self.crawl_foundational_anchors() if "anchors" in run else set()

        # Tier 3: Thematic fill
        final_papers = self.crawl_thematic_fill(target_total=target) if "fill" in run else self.db.count_papers()
        final_citations = self.db.count_citations()

        summary = {
            "target_size": target,
            "final_paper_count": final_papers,
            "final_citation_count": final_citations,
            "authored_papers": len(authored_ids),
            "hop1_papers": len(hop1_ids),
            "hop2_papers": len(hop2_ids),
            "foundational_anchors": len(anchor_ids),
            "forbidden_target_paper_b_count": len(self.forbidden_paper_b_ids),
        }

        # Write crawl manifest
        summary_path = self.checkpoint_dir / "crawl_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Corpus crawl finished successfully: {summary}")
        return summary

