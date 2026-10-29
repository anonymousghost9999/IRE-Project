"""Data report: corpus statistics, coverage ceiling, cohort and slice sizes (Phase 1, Step 1.11).

Governed by docs/plan.md §1.11 and docs/design.md §4.1. Pure reporting: reads the frozen database,
cohort and ground truth, and writes a JSON file plus a Markdown summary.
"""

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

from paperfeed.data.cohort import CandidateAuthor, clean_id
from paperfeed.data.db import DatabaseManager
from paperfeed.data.ground_truth import GroundTruth, to_uri
from paperfeed.data.leakage import LeakageReport


def _ceiling(truths: List[GroundTruth]) -> Dict[str, Any]:
    """Coverage ceiling: share of Discovery references that are in the corpus with an abstract."""
    ratios = [len(t.discovery_in_corpus) / len(t.discovery) for t in truths if t.discovery]
    in_corpus = [len(t.discovery_in_corpus) for t in truths]
    return {
        "n_authors": len(truths),
        "coverage_ceiling_mean": round(mean(ratios), 4) if ratios else None,
        "coverage_ceiling_median": round(median(ratios), 4) if ratios else None,
        "discovery_refs_mean": round(mean(len(t.discovery) for t in truths), 2) if truths else None,
        "discovery_in_corpus_mean": round(mean(in_corpus), 2) if in_corpus else None,
        "discovery_in_corpus_median": median(in_corpus) if in_corpus else None,
    }


def build_data_report(
    db: DatabaseManager,
    cutoff_year: int,
    pool_truths: List[GroundTruth],
    cohort: List[CandidateAuthor],
    cohort_truths: List[GroundTruth],
    split: Dict[str, List[str]],
    leakage: LeakageReport,
    min_discovery: int,
) -> Dict[str, Any]:
    with db.get_connection() as conn:
        one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
        n_papers = one("SELECT COUNT(*) FROM papers")
        n_cand = one("SELECT COUNT(*) FROM papers WHERE in_candidates = 1")
        corpus: Dict[str, Any] = {
            "papers": n_papers,
            "candidate_papers_with_abstract": n_cand,
            "abstract_coverage": round(n_cand / n_papers, 4) if n_papers else None,
            "citation_edges": one("SELECT COUNT(*) FROM paper_citations"),
            "papers_with_outgoing_edges": one("SELECT COUNT(DISTINCT source_paper_id) FROM paper_citations"),
            "unknown_topic_share": round(one("SELECT COUNT(*) FROM papers WHERE primary_topic_id = 'UNKNOWN'") / max(n_papers, 1), 5),
            "papers_before_1990": one("SELECT COUNT(*) FROM papers WHERE year < 1990"),
            "year_histogram_5y": {
                str(y): c for y, c in conn.execute("SELECT (year/5)*5, COUNT(*) FROM papers WHERE year >= 1990 GROUP BY 1")
            },
            "top_fields": {
                f: c
                for f, c in conn.execute(
                    "SELECT field_id, COUNT(*) FROM papers GROUP BY 1 ORDER BY 2 DESC LIMIT 8"
                )
            },
            "max_year": one("SELECT MAX(year) FROM papers"),
        }

        # Co-citation availability: in-corpus papers citing the author's papers.
        citing_counts: List[int] = []
        for c in cohort:
            uris = [to_uri(a) for a in c.authored_paper_ids_T]
            total = 0
            for i in range(0, len(uris), 500):
                chunk = uris[i : i + 500]
                marks = ",".join("?" for _ in chunk)
                total += conn.execute(
                    f"SELECT COUNT(DISTINCT source_paper_id) FROM paper_citations WHERE target_paper_id IN ({marks})",
                    chunk,
                ).fetchone()[0]
            citing_counts.append(total)

        # Item cold-start: popularity at T of the in-corpus Discovery targets of the test split.
        test_ids = set(split["test"])
        target_ids = sorted({t for g in cohort_truths if g.author_id in test_ids for t in g.discovery_in_corpus})
        cites: List[int] = []
        for i in range(0, len(target_ids), 500):
            chunk = [to_uri(x) for x in target_ids[i : i + 500]]
            marks = ",".join("?" for _ in chunk)
            cites += [
                (r[0], r[1])
                for r in conn.execute(f"SELECT citation_count_T, year FROM papers WHERE paper_id IN ({marks})", chunk)
            ]

    slice_by_author = {c.author_id: c.history_size_slice for c in cohort}
    slice_counts = {
        part: dict(Counter(slice_by_author[a] for a in ids)) for part, ids in (("tuning", split["tuning"]), ("test", split["test"]))
    }

    return {
        "cutoff_year": cutoff_year,
        "corpus": corpus,
        "coverage_before_cohort_filter": _ceiling(pool_truths),
        "coverage_after_cohort_filter": _ceiling(cohort_truths),
        "candidate_pool": {
            "authors": len(pool_truths),
            "authors_with_zero_references": sum(1 for t in pool_truths if not t.refs_all),
            "authors_passing_min_discovery": sum(1 for t in pool_truths if len(t.discovery_in_corpus) >= min_discovery),
            "min_discovery_in_corpus": min_discovery,
        },
        "cohort": {
            "realized_N": len(cohort),
            "slices": dict(Counter(c.history_size_slice for c in cohort)),
            "split_sizes": {k: len(v) for k, v in split.items()},
            "split_slice_sizes": slice_counts,
        },
        "co_citation_availability": {
            "authors_with_no_in_corpus_citing_paper": sum(1 for x in citing_counts if x == 0),
            "median_citing_papers_per_author": median(citing_counts) if citing_counts else None,
        },
        "item_cold_start_test_targets": {
            "n_targets": len(cites),
            # OpenAlex counts_by_year starts in 2012, so citation_count_T is complete only for papers
            # published from 2012 on. "Fresh" additionally requires a recent paper, otherwise an old paper
            # whose citations all predate 2012 would look uncited.
            "fresh_le_1_citation_and_recent": sum(1 for c, y in cites if c <= 1 and y >= cutoff_year - 1),
            "established_ge_20_citations_since_2012": sum(1 for c, y in cites if c >= 20 and y >= 2012),
        },
        "assumption": "References of B that are not in the database have unknown year and are assumed <= T.",
        "leakage_check": {"clean": leakage.clean, "violations": leakage.counts},
    }


def render_markdown(r: Dict[str, Any]) -> str:
    c, co, b, a = r["corpus"], r["cohort"], r["coverage_before_cohort_filter"], r["coverage_after_cohort_filter"]
    cc, ics, pool = r["co_citation_availability"], r["item_cold_start_test_targets"], r["candidate_pool"]
    lines = [
        f"# Data report (T = {r['cutoff_year']})",
        "",
        "## Corpus",
        f"- Papers: {c['papers']:,} ({c['candidate_papers_with_abstract']:,} candidates with an abstract, {c['abstract_coverage']:.1%})",
        f"- Citation edges: {c['citation_edges']:,} (from {c['papers_with_outgoing_edges']:,} citing papers)",
        f"- Latest year: {c['max_year']}; papers before 1990: {c['papers_before_1990']:,}; unknown-topic share: {c['unknown_topic_share']:.2%}",
        f"- Top fields: {c['top_fields']}",
        "",
        "## Coverage ceiling (share of Discovery references in the corpus with an abstract)",
        f"- Candidate pool ({b['n_authors']} authors): mean {b['coverage_ceiling_mean']}, median {b['coverage_ceiling_median']}",
        f"- Cohort after filter ({a['n_authors']} authors): mean {a['coverage_ceiling_mean']}, median {a['coverage_ceiling_median']}; "
        f"in-corpus Discovery refs per author: mean {a['discovery_in_corpus_mean']}, median {a['discovery_in_corpus_median']}",
        "",
        "## Cohort",
        f"- Pool: {pool['authors']} authors after dropping {pool.get('authors_dropped_target_predates_T', 0)} whose target paper predates T+1 "
        f"(cited by papers from <= T); {pool['authors_with_zero_references']} with no references, "
        f"{pool['authors_passing_min_discovery']} with >= {pool['min_discovery_in_corpus']} in-corpus Discovery references",
        f"- Realized N = {co['realized_N']}; slices {co['slices']}",
        f"- Split {co['split_sizes']}; per slice {co['split_slice_sizes']}",
        "",
        "## Signals",
        f"- Co-citation availability: {cc['authors_with_no_in_corpus_citing_paper']} cohort authors have no in-corpus citing paper "
        f"(median {cc['median_citing_papers_per_author']} per author)",
        f"- Item cold start (test targets): {ics['n_targets']} targets, {ics['fresh_le_1_citation_and_recent']} fresh (<= 1 citation, published in the last 2 years), "
        f"{ics['established_ge_20_citations_since_2012']} established (>= 20, published since 2012)",
        "",
        "## Leakage check",
        f"- Clean: {r['leakage_check']['clean']} {r['leakage_check']['violations'] or ''}",
        "",
        f"_{r['assumption']}_",
        "",
    ]
    return "\n".join(lines)


def write_report(report: Dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "data_report.md").write_text(render_markdown(report), encoding="utf-8")
