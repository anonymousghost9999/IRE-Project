"""Tests for ground truth, cohort selection / split, leakage checker and data report (Phase 1.8-1.11)."""

from pathlib import Path
from typing import List

import pytest

from paperfeed.data.cohort import CandidateAuthor, seed_topics_for_author, select_cohort, split_cohort, work_cited_before_publication
from paperfeed.data.db import DatabaseManager
from paperfeed.data.ground_truth import compute_ground_truth, load_ground_truth, save_ground_truth
from paperfeed.data.leakage import LeakageError, check_all, check_corpus, check_paper_ids, find_targets_cited_before_publication
from paperfeed.data.report import build_data_report, render_markdown

U = lambda x: f"https://openalex.org/{x}"  # noqa: E731


def _paper(pid, year=2020, inc=1, topic="T1", cites=0):
    return {"paper_id": U(pid), "title": pid, "abstract": "x" if inc else "", "authors": "[]", "year": year,
            "publication_date": f"{year}-06-01", "venue": None, "doi": None, "url": None,
            "citation_count_T": cites, "primary_topic_id": U(topic), "subfield_id": None, "field_id": None,
            "in_candidates": inc}


def _author(aid="A1", slice_name="warm", refs=None, authored=None) -> CandidateAuthor:
    return CandidateAuthor(aid, aid, "WB_" + aid, "B", "2025-01-10",
                           refs or [], ["Topic"], authored or ["WA1", "WA2"], "rule", slice_name)


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    d = DatabaseManager(db_path=tmp_path / "t.sqlite3", cutoff_year=2024)
    d.insert_papers_batch([
        _paper("WA1"), _paper("WA2", year=2023),
        _paper("WC1"),               # cited by the author before -> re-citation
        _paper("WD1", cites=0),      # new, in corpus -> discovery
        _paper("WD2", cites=40),     # new, in corpus -> discovery
        _paper("WNOABS", inc=0),     # new but no abstract -> discovery, not in corpus
    ])
    d.insert_citations_batch([(U("WA1"), U("WC1"), 2020), (U("WA2"), U("WC1"), 2023)])
    return d


def test_ground_truth_sets(db):
    a = _author(refs=[U("WA1"), U("WC1"), U("WD1"), U("WD2"), U("WNOABS"), U("WOUT")])
    gt = compute_ground_truth(db, a, cutoff_year=2024)
    assert gt.n_citations_u == 1
    assert set(gt.recitation) == {"WC1"}
    assert "WA1" not in gt.zero_pub                      # self-citation removed everywhere
    assert set(gt.discovery) == {"WD1", "WD2", "WNOABS", "WOUT"}
    assert set(gt.discovery_in_corpus) == {"WD1", "WD2"}  # no abstract / not stored -> not in corpus
    assert set(gt.zero_pub) == {"WC1", "WD1", "WD2", "WNOABS", "WOUT"}  # no exclusion of past citations


def test_ground_truth_drops_refs_after_cutoff(db):
    db.insert_papers_batch([_paper("WLATE", year=2025)])
    gt = compute_ground_truth(db, _author(refs=[U("WLATE"), U("WD1")]), cutoff_year=2024)
    assert gt.discovery == ["WD1"]


def test_ground_truth_roundtrip(db, tmp_path):
    gt = compute_ground_truth(db, _author(refs=[U("WD1")]), 2024)
    save_ground_truth([gt], tmp_path / "gt.jsonl")
    assert load_ground_truth(tmp_path / "gt.jsonl")[0] == gt


def _pool() -> List[CandidateAuthor]:
    pool = [_author(f"W{i:03d}", "warm") for i in range(50)]
    pool += [_author(f"M{i:03d}", "moderate") for i in range(6)] + [_author(f"S{i:03d}", "sparse") for i in range(4)]
    return pool


def test_select_cohort_keeps_rare_slices_first_and_is_deterministic():
    pool = _pool()
    disc = {c.author_id: 5 for c in pool}
    disc["W000"] = 4        # below the minimum -> never selected
    disc["M000"] = 0
    first = select_cohort(pool, disc, n_total=20, min_discovery=5, seed=1)
    assert len(first) == 20
    ids = {c.author_id for c in first}
    assert {f"M{i:03d}" for i in range(1, 6)} <= ids and {f"S{i:03d}" for i in range(4)} <= ids
    assert "W000" not in ids and "M000" not in ids
    assert first == select_cohort(pool, disc, n_total=20, min_discovery=5, seed=1)
    assert first != select_cohort(pool, disc, n_total=20, min_discovery=5, seed=2)


def test_split_is_stratified_disjoint_and_complete():
    pool = _pool()
    split = split_cohort(pool, tuning_fraction=0.2, seed=7)
    assert set(split["tuning"]).isdisjoint(split["test"])
    assert set(split["tuning"]) | set(split["test"]) == {c.author_id for c in pool}
    tune_slices = {c.history_size_slice for c in pool if c.author_id in set(split["tuning"])}
    assert tune_slices == {"warm", "moderate", "sparse"}  # every slice appears in tuning
    assert len(split["tuning"]) == 10 + 1 + 1  # round(0.2*50), and one each for moderate / sparse
    assert split == split_cohort(pool, tuning_fraction=0.2, seed=7)


def test_seed_topics_most_frequent_first():
    names = {"T1": "Retrieval", "T2": "Ranking", "T3": "Vision"}
    assert seed_topics_for_author(["T2", "T1", "T1", "UNKNOWN", "T3", "T2", "T1"], names, k=2) == ["Retrieval", "Ranking"]
    assert seed_topics_for_author(["TX"], names) == ["TX"]  # unknown name falls back to the ID


def test_leakage_checker_clean_and_dirty(db):
    forbidden = {"WB_A1"}
    assert check_corpus(db, 2024, forbidden).clean

    db.insert_papers_batch([_paper("WB_A1", year=2025), _paper("WFUT", year=2026)])
    with db.get_connection() as conn:
        conn.execute("INSERT INTO paper_citations VALUES (?,?,?)", (U("WFUT"), U("WD1"), 2026))
    report = check_corpus(db, 2024, forbidden)
    assert not report.clean
    assert {"paper_year_after_T", "target_paper_B_in_papers", "citation_edge_source_year_after_T"} <= set(report.violations)
    with pytest.raises(LeakageError):
        report.assert_clean()


def test_user_leakage_checks(db):
    a = _author(refs=[U("WD1")], authored=["WA1", "WB_A1"])  # B listed in the author's own history
    db.insert_papers_batch([_paper("WB_A1", year=2025)])
    gt = compute_ground_truth(db, a, 2024)
    report = check_all(db, [a], [gt], 2024, ["WB_A1"])
    assert "target_paper_B_in_authored_history" in report.violations
    assert "authored_paper_year_after_T" in report.violations


def test_check_paper_ids_hook(db):
    db.insert_papers_batch([_paper("WLATE", year=2026)])
    r = check_paper_ids(db, ["WD1", "WLATE", "WB_A1"], 2024, ["WB_A1"])
    assert r.violations["uses_target_paper_B"] == ["WB_A1"]
    assert r.violations["uses_paper_after_T"] == ["WLATE"]
    assert check_paper_ids(db, ["WD1"], 2024, ["WB_A1"]).clean


def test_data_report(db):
    a = _author(refs=[U("WD1"), U("WD2"), U("WNOABS")])
    gt = compute_ground_truth(db, a, 2024)
    split = {"tuning": [], "test": [a.author_id]}
    leak = check_all(db, [a], [gt], 2024, ["WB_A1"])
    rep = build_data_report(db, 2024, [gt], [a], [gt], split, leak, min_discovery=2)
    assert rep["corpus"]["papers"] == 6
    assert rep["coverage_after_cohort_filter"]["coverage_ceiling_mean"] == round(2 / 3, 4)
    assert rep["cohort"]["realized_N"] == 1 and rep["leakage_check"]["clean"]
    assert rep["item_cold_start_test_targets"] == {"n_targets": 2, "fresh_le_1_citation_and_recent": 0, "established_ge_20_citations_since_2012": 1}
    assert "Data report" in render_markdown(rep)


def test_b_is_never_its_own_target(db):
    a = _author(refs=[U("WB_A1"), U("WD1")])
    gt = compute_ground_truth(db, a, 2024)
    assert "WB_A1" not in gt.refs_all and gt.discovery == ["WD1"]


def test_target_cited_before_publication_is_detected(db):
    # a 2022 paper cites "B": B cannot be a genuine 2025 paper
    db.insert_citations_batch([(U("WA1"), U("WOLD_B"), 2022)])
    assert find_targets_cited_before_publication(db, ["WOLD_B", "WGOOD_B"], 2024) == {"WOLD_B"}
    # the corpus check ignores dropped targets when told which targets are still in use
    assert check_corpus(db, 2024, ["WOLD_B"], cited_forbidden_ids=["WGOOD_B"]).clean
    assert "target_paper_B_is_cited" in check_corpus(db, 2024, ["WOLD_B"]).violations


def test_work_cited_before_publication():
    assert work_cited_before_publication({"counts_by_year": [{"year": 2019, "cited_by_count": 3}]}, 2024)
    assert not work_cited_before_publication({"counts_by_year": [{"year": 2025, "cited_by_count": 3}]}, 2024)
    assert not work_cited_before_publication({"counts_by_year": [{"year": 2020, "cited_by_count": 0}]}, 2024)
    assert not work_cited_before_publication({}, 2024)
