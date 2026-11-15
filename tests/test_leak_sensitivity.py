"""Leak-free sensitivity analysis (design §18a): leak finder and re-scoring of saved results."""

import pytest

from paperfeed.data.db import DatabaseManager
from paperfeed.data.leakage import preprint_leaked_authors
from paperfeed.eval.leak_sensitivity import format_report, run_sensitivity


def _paper(pid, title):
    return {"paper_id": f"https://openalex.org/{pid}", "title": title, "abstract": "x", "authors": "[]", "year": 2023,
            "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": 0,
            "primary_topic_id": None, "subfield_id": None, "field_id": None, "in_candidates": 1}


def test_preprint_leak_finder(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "l.sqlite3", cutoff_year=2024)
    db.insert_papers_batch([_paper("P1", "Deep Graph Anomaly Detection: A Survey and New Perspectives"),
                            _paper("P2", "Something else entirely"), _paper("P3", "Unrelated work on galaxies")])
    cohort = [{"author_id": "https://openalex.org/A1", "split": "test", "authored_paper_ids_T": ["P1", "P2"]},
              {"author_id": "A2", "split": "tuning", "authored_paper_ids_T": ["P3"]}]
    titles = {"A1": "Deep graph anomaly detection - a survey and new perspectives", "A2": "Galaxy rotation curves"}
    leaked = preprint_leaked_authors(db, cohort, titles)
    assert set(leaked) == {"A1"} and leaked["A1"]["history_paper"] == "P1" and leaked["A1"]["split"] == "test"


def _rows(n, full, base, sota, extra=None):
    rows = {}
    for i in range(n):
        m = lambda v: {"ndcg_at_10": v, "recall_at_10": v, "topic_entropy": 1.0}  # noqa: E731
        sm = {"Paper Feed (Full)": m(full(i)), "Ablation: No-MMR": m(full(i)), "SOTA: X": m(sota(i))}
        for b in ("popularity", "bm25_author", "dense_centroid", "citation_only", "hybrid_persona"):
            sm[f"Baseline: {b}"] = m(base(i))
        sm.update(extra(i) if extra else {})
        rows[f"A{i}"] = {"system_metrics": sm}
    return rows


def test_sensitivity_flags_a_conclusion_that_depends_on_leaked_authors():
    n = 40
    # Paper Feed beats SOTA: X only because of the 3 leaked authors A0-A2 (huge scores there, a tie elsewhere)
    core = _rows(n, full=lambda i: 0.9 if i < 3 else 0.2, base=lambda i: 0.1 + 0.01 * (i % 2), sota=lambda i: 0.2)
    variants = {"v1": _rows(n, full=lambda i: 0.2, base=lambda i: 0.1, sota=lambda i: 0.2)}
    s = run_sensitivity(core, variants, leaked=["A0", "A1", "A2", "ZZ"], n_resamples=300)
    assert (s["n_test_all"], s["n_test_leaked"], s["n_test_clean"]) == (40, 3, 37)  # ZZ is not a test author
    sota = {r["comparison"]: r for r in s["sota"]}
    x = sota["SOTA: X | ndcg_at_10"]
    assert x["all"]["diff"] > 0  # only the leaked authors make Paper Feed look better
    assert x["clean"]["diff"] == pytest.approx(0.0) and x["clean"]["significant"] is False
    assert x["conclusion_changed"] == x["all"]["significant"]  # flagged exactly when the full-set result was significant
    h1 = {r["comparison"]: r for r in s["h1"]}
    assert h1["Baseline: popularity | ndcg_at_10"]["clean"]["significant"] is True  # 0.2 vs ~0.1 everywhere
    md = format_report(s, {"A0": {"split": "test", "history_paper": "P", "history_title": "t", "b_title": "t"}})
    assert "Leak-Free Sensitivity" in md and "| A0 | test |" in md


def test_variant_runs_must_cover_the_same_authors():
    core = _rows(5, lambda i: 0.2, lambda i: 0.1, lambda i: 0.2)
    with pytest.raises(ValueError, match="different test authors"):
        run_sensitivity(core, {"v1": _rows(4, lambda i: 0.2, lambda i: 0.1, lambda i: 0.2)}, leaked=[], n_resamples=50)
