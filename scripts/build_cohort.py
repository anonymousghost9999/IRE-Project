"""Freeze the evaluation cohort, ground truth, split, leakage check and data report (Phase 1.8-1.11).

Run after the corpus crawl:
    python scripts/build_cohort.py [--authors-file data/candidate_authors.jsonl] [--out-dir data/processed]

Outputs (in --out-dir): cohort.jsonl, ground_truth.jsonl, split.json, data_report.json/.md.
Exits with status 1 if the leakage checker finds a violation.
"""

import argparse
import json
from pathlib import Path
import sys

from paperfeed.common.config import find_project_root, load_config
from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import (
    clean_id,
    load_candidate_authors,
    load_topic_names,
    work_cited_before_publication,
    seed_topics_for_author,
    select_cohort,
    split_cohort,
)
from paperfeed.data.db import DatabaseManager
from paperfeed.data.ground_truth import compute_ground_truth, save_ground_truth, to_uri
from paperfeed.data.leakage import check_all, find_targets_cited_before_publication, titles_match
from paperfeed.data.report import build_data_report, write_report
from paperfeed.openalex.client import OpenAlexClient

logger = get_logger("scripts.build_cohort")


def validate_targets_with_api(pool, cutoff: int, cache_path: Path) -> set:
    """Target papers B that OpenAlex reports as cited in a year <= T (cached; one small request per 50 papers)."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [c.target_paper_b_id for c in pool if c.target_paper_b_id not in cache]
    if todo:
        logger.info(f"Validating {len(todo)} target papers against OpenAlex counts_by_year...")
        client = OpenAlexClient()
        try:
            for work in client.fetch_works_batch(todo, chunk_size=50, select_fields=["id", "counts_by_year"]):
                cache[work["id"].replace("https://openalex.org/", "")] = work_cited_before_publication(work, cutoff)
        finally:
            client.close()
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {b for b, predates in cache.items() if predates}


def main() -> int:
    parser = argparse.ArgumentParser(description="Select cohort, compute ground truth, split, check leakage.")
    parser.add_argument("--authors-file", default="data/candidate_authors.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--offline", action="store_true", help="Skip the OpenAlex validation of target papers (cached results are still used)")
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_config()
    cutoff = cfg["project"]["cutoff_year"]
    seed = cfg["project"].get("random_seed", 42)
    cohort_cfg = cfg.get("cohort", {})
    n_total = cohort_cfg.get("total_authors", 400)
    min_disc = cohort_cfg.get("min_discovery_references", 5)
    tune_frac = cohort_cfg.get("tuning_authors", 80) / n_total

    authors_path = root / args.authors_file
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    db = DatabaseManager(cutoff_year=cutoff)
    all_candidates = load_candidate_authors(authors_path)

    # A genuine T+1 paper cannot already be cited by papers from <= T. Drop authors whose "B" is an old work
    # that OpenAlex dated to T+1 (reissues, merged records): they would not be a leave-future-out test.
    predating = find_targets_cited_before_publication(db, [c.target_paper_b_id for c in all_candidates], cutoff)
    cache_path = root / cfg["paths"]["interim_dir"] / "target_validation.json"
    if args.offline:
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        predating |= {b for b, flag in cached.items() if flag}
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        predating |= validate_targets_with_api(all_candidates, cutoff, cache_path)
    pool = [c for c in all_candidates if c.target_paper_b_id not in predating]
    logger.info(f"Dropped {len(all_candidates) - len(pool)} authors whose target paper predates T+1; {len(pool)} remain.")

    # Drop authors whose target paper B leaked into their history <= T as a preprint
    clean_pool = []
    dropped_preprint = 0
    with db.get_connection() as conn:
        for c in pool:
            b_title = getattr(c, "target_paper_b_title", "")
            authored = [clean_id(a) for a in c.authored_paper_ids_T]
            leaked = False
            if b_title and authored:
                chunk = [to_uri(a) for a in authored]
                for i in range(0, len(chunk), 500):
                    sub = chunk[i : i + 500]
                    marks = ",".join("?" for _ in sub)
                    for row in conn.execute(
                        f"SELECT paper_id, title FROM papers WHERE paper_id IN ({marks})", sub
                    ):
                        if row[1] and titles_match(row[1], b_title):
                            leaked = True
                            break
                    if leaked:
                        break
            if leaked:
                dropped_preprint += 1
            else:
                clean_pool.append(c)

    pool = clean_pool
    logger.info(f"Dropped {dropped_preprint} authors whose target paper B leaked as a preprint; {len(pool)} remain.")
    logger.info(f"Computing ground truth for {len(pool)} candidate authors...")
    pool_truths = [compute_ground_truth(db, c, cutoff) for c in pool]
    truth_by_author = {t.author_id: t for t in pool_truths}

    cohort = select_cohort(
        pool,
        {t.author_id: len(t.discovery_in_corpus) for t in pool_truths},
        n_total=n_total,
        min_discovery=min_disc,
        seed=seed,
    )
    cohort_truths = [truth_by_author[c.author_id] for c in cohort]
    split = split_cohort(cohort, tuning_fraction=tune_frac, seed=seed)
    logger.info(f"Cohort: {len(cohort)} authors ({len(split['tuning'])} tuning / {len(split['test'])} test)")

    # Zero-publication slice: 3 most frequent primary Topics of each author's papers <= T.
    topic_names = load_topic_names(root / cfg["paths"]["raw_dir"])
    split_of = {a: part for part, ids in split.items() for a in ids}
    with db.get_connection() as conn, open(out_dir / "cohort.jsonl", "w", encoding="utf-8") as f:
        for c in cohort:
            uris = [to_uri(a) for a in c.authored_paper_ids_T]
            topics = []
            for i in range(0, len(uris), 500):
                chunk = uris[i : i + 500]
                marks = ",".join("?" for _ in chunk)
                topics += [r[0] for r in conn.execute(f"SELECT primary_topic_id FROM papers WHERE paper_id IN ({marks})", chunk)]
            row = {
                "author_id": c.author_id,
                "author_name": c.author_name,
                "paper_b_id": c.target_paper_b_id,
                "history_size_slice": c.history_size_slice,
                "split": split_of[c.author_id],
                "n_authored": len(c.authored_paper_ids_T),
                "authored_paper_ids_T": c.authored_paper_ids_T,
                "seed_topics": seed_topics_for_author(topics, topic_names, k=3),
                "selection_rule": c.selection_rule,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    save_ground_truth(cohort_truths, out_dir / "ground_truth.jsonl")
    (out_dir / "split.json").write_text(json.dumps({"seed": seed, **split}, indent=2), encoding="utf-8")

    forbidden = {c.target_paper_b_id for c in all_candidates}  # every candidate's B must be absent from the corpus
    leakage = check_all(db, cohort, cohort_truths, cutoff, forbidden, cited_forbidden_ids={c.target_paper_b_id for c in pool})
    report = build_data_report(db, cutoff, pool_truths, cohort, cohort_truths, split, leakage, min_disc)
    report["candidate_pool"]["authors_dropped_target_predates_T"] = len(all_candidates) - (len(pool) + dropped_preprint)
    report["candidate_pool"]["authors_dropped_preprint_leakage"] = dropped_preprint
    write_report(report, out_dir)

    logger.info(f"Wrote cohort, ground truth, split and data report to {out_dir}")
    if not leakage.clean:
        logger.error(f"LEAKAGE CHECK FAILED: {leakage.counts}")
        return 1
    logger.info("Leakage check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
