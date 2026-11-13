"""Explanation faithfulness audit (Phase 8 / docs/plan.md §8 / docs/design.md §13.6).

Audits deterministic recommendations against two criteria (prob.md §2.4, §6E.5):
1. Extractive Grounding Rate: Displayed facet spans and evidence sentences exist verbatim at their character offsets
   in the candidate paper's abstract text (target: 100%); citation edges are verified in SQLite under year <= T.
2. Counterfactual Necessity (Drop-out Rate): When the explained feature contribution is set to 0 (DeltaScore = 0),
   the paper leaves the Top-10 or falls >= 3 ranks under MMR re-diversification (verifying the explanation highlights
   a necessary ranking signal). The counterfactual re-ranks the exact scored pool and weights that produced the feed
   (``RecommendationResult.scored_pool``), and the audit first checks that re-ranking it unchanged reproduces the feed.
3. Explanation Coverage: Percentage of Top-10 recommendations successfully annotated with a grounded explanation (target >= 90%).
4. Template Distribution: Proportions of explanation templates across the evaluated recommendations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.index.store import EmbeddingStore
from paperfeed.rec.explain import EXPLAINABLE_FEATURES, Explainer, Explanation, ExplanationResult
from paperfeed.rec.features import edges
from paperfeed.rec.mmr import select_top_k
from paperfeed.rec.recommender import (
    RecommendConfig,
    RecommenderResources,
    UserProfile,
    build_user_profile,
    recommend,
)
from paperfeed.rec.scoring import ScoredCandidate

logger = get_logger("eval.faithfulness")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ExplanationSample:
    """An explanation produced for an author's recommended feed item."""

    author_id: str
    paper_id: str
    rank: int  # 1-based original rank in feed
    dominant_feature: str
    explanation: Explanation
    scored_candidates: List[ScoredCandidate]  # candidate pool for counterfactual re-scoring


@dataclass
class AuditedExplanation:
    """Audit verification record for a single explanation."""

    author_id: str
    paper_id: str
    original_rank: int
    template: str
    text: str
    is_grounded: bool
    grounding_details: Dict[str, Any]
    counterfactual_rank: Optional[int]  # None if fell out of top-K
    rank_drop: int  # original_rank to counterfactual_rank difference (>= 0)
    is_counterfactually_necessary: bool  # leaves top-K OR drops >= 3 ranks
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FaithfulnessAuditResult:
    """Aggregated results of the explanation faithfulness audit."""

    n_sampled: int
    n_users_in_sample: int
    total_evaluated_recommendations: int
    total_covered_recommendations: int
    explanation_coverage: float  # share of top-10 items with a grounded explanation
    extractive_grounding_rate: float  # share of audited items with 100% verified grounding
    counterfactual_dropout_rate: float  # share of audited items that drop >= 3 ranks or leave top-10
    left_top_k_count: int  # items that completely fell out of top-10
    dropped_gte_3_count: int  # items that remained in top-10 but dropped >= 3 ranks
    retained_near_rank_count: int  # items that remained in top-10 with < 3 rank drop
    mean_rank_drop: float
    median_rank_drop: float
    template_distribution: Dict[str, int]
    template_proportions: Dict[str, float]
    audited_explanations: List[AuditedExplanation] = field(default_factory=list)
    # template -> {"n", "grounded", "necessary", "left_top_k", "dropped_gte_3"} over the sample
    per_template: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # population template counts over every covered Top-10 item (before sampling)
    population_template_distribution: Dict[str, int] = field(default_factory=dict)
    # V2 (design §13.8 point 6): Top-10 items whose reranker contribution exceeds every explainable contribution, i.e.
    # where the shown explanation is not the main reason for the rank. None when the reranker is off.
    n_reranker_dominant: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_sampled": self.n_sampled,
            "n_users_in_sample": self.n_users_in_sample,
            "total_evaluated_recommendations": self.total_evaluated_recommendations,
            "total_covered_recommendations": self.total_covered_recommendations,
            "explanation_coverage": self.explanation_coverage,
            "extractive_grounding_rate": self.extractive_grounding_rate,
            "counterfactual_dropout_rate": self.counterfactual_dropout_rate,
            "left_top_k_count": self.left_top_k_count,
            "dropped_gte_3_count": self.dropped_gte_3_count,
            "retained_near_rank_count": self.retained_near_rank_count,
            "mean_rank_drop": self.mean_rank_drop,
            "median_rank_drop": self.median_rank_drop,
            "template_distribution": self.template_distribution,
            "template_proportions": self.template_proportions,
            "per_template": self.per_template,
            "population_template_distribution": self.population_template_distribution,
            "n_span_explanations_audited": self.n_span_explanations,
            "n_reranker_dominant": self.n_reranker_dominant,
            "audited_explanations": [a.to_dict() for a in self.audited_explanations],
        }

    @property
    def n_span_explanations(self) -> int:
        """Audited explanations that quote spans from the abstract (facet, new_method): the plan's grounding check."""
        return sum(self.per_template.get(t, {}).get("n", 0) for t in ("facet", "new_method"))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# Paper Feed — Phase 8: Explanation Faithfulness Audit Report",
            "",
            "> [!NOTE]",
            f"> **Audit Configuration:**",
            f"> - Sample Size: {self.n_sampled} explanations across {self.n_users_in_sample} distinct researchers",
            f"> - Total Evaluated Top-10 Recommendations: {self.total_evaluated_recommendations:,}",
            f"> - Extractive Grounding Target: 100.0% | Explanation Coverage Target: ≥ 90.0%",
            "",
            "## 1. Core Verification Metrics",
            "",
            "| Metric | Target | Realized | Status |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Extractive Grounding Rate** | 100.0% | **{self.extractive_grounding_rate * 100.0:.2f}%** | {'✅ PASS' if self.extractive_grounding_rate >= 0.999 else '⚠️ WARN'} |",
            f"| **Counterfactual Necessity (Drop-Out Rate)** | — | **{self.counterfactual_dropout_rate * 100.0:.2f}%** | {'✅ HIGH' if self.counterfactual_dropout_rate >= 0.70 else 'INFO'} |",
            f"| **Explanation Coverage (Top-10)** | ≥ 90.0% | **{self.explanation_coverage * 100.0:.2f}%** | {'✅ PASS' if self.explanation_coverage >= 0.90 else '⚠️ WARN'} |",
            f"| **Mean Rank Drop (ΔRank)** | — | **+{self.mean_rank_drop:.2f}** | INFO |",
            f"| **Median Rank Drop** | — | **+{self.median_rank_drop:.1f}** | INFO |",
            "",
            "## 2. Counterfactual Impact Breakdown",
            "",
            "When the dominant explained feature is zeroed out (ΔScore = 0) and MMR is re-diversified:",
            "",
            "| Counterfactual Outcome | Count | Percentage | Description |",
            "| :--- | :---: | :---: | :--- |",
            f"| **Dropped Out of Top-10** | {self.left_top_k_count} | {self.left_top_k_count / max(self.n_sampled, 1) * 100.0:.1f}% | Item completely exited the recommendation feed |",
            f"| **Fell ≥ 3 Ranks** | {self.dropped_gte_3_count} | {self.dropped_gte_3_count / max(self.n_sampled, 1) * 100.0:.1f}% | Item stayed in Top-10 but suffered major rank demotion |",
            f"| **Retained Rank (Drop < 3)** | {self.retained_near_rank_count} | {self.retained_near_rank_count / max(self.n_sampled, 1) * 100.0:.1f}% | Item was strongly supported by secondary score components |",
            f"| **Total Counterfactually Necessary** | **{self.left_top_k_count + self.dropped_gte_3_count}** | **{self.counterfactual_dropout_rate * 100.0:.1f}%** | **(Fell out of Top-10 OR dropped ≥ 3 ranks)** |",
            "",
            "## 3. Explanation Template Distribution",
            "",
            "| Template / Dominant Feature | Sample Count | Sample Share (%) |",
            "| :--- | :---: | :---: |",
        ]
        for template, count in sorted(self.template_distribution.items(), key=lambda x: -x[1]):
            share = self.template_proportions.get(template, 0.0) * 100.0
            lines.append(f"| `{template}` | {count} | {share:.1f}% |")

        lines.extend([
            "",
            "Per template in the audited sample (grounding / counterfactual):",
            "",
            "| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for template in sorted(set(self.per_template) | set(self.population_template_distribution)):
            t = self.per_template.get(template, {})
            lines.append(
                f"| `{template}` | {t.get('n', 0)} | {t.get('grounded', 0)} | {t.get('necessary', 0)} | {t.get('left_top_k', 0)} "
                f"| {t.get('dropped_gte_3', 0)} | {self.population_template_distribution.get(template, 0)} |"
            )
        notes = [
            "",
            "**Reading these numbers:**",
            f"- The plan's grounding check is about quoted spans (facet and new-method templates); "
            f"{self.n_span_explanations} of the {self.n_sampled} audited explanations quote spans.",
        ]
        if self.n_span_explanations == 0:
            n_span_population = sum(self.population_template_distribution.get(t, 0) for t in ("facet", "new_method"))
            why = (
                f"span templates were only {n_span_population} of the covered Top-10 items, too rare to be sampled"
                if n_span_population
                else "the facet and new-method terms never dominated a Top-10 score, so no span explanation was shown"
            )
            notes.append(
                "- **No span-quoting explanation was audited**, so the Extractive Grounding Rate above covers only citation "
                f"edges (re-verified in the database) and persona/session cosines, not verbatim spans. With the audited weights {why}."
            )
        if self.n_reranker_dominant is not None:
            share = self.n_reranker_dominant / max(self.total_evaluated_recommendations, 1)
            notes.append(
                f"- **Reranker (V2):** in {self.n_reranker_dominant} of {self.total_evaluated_recommendations} Top-10 items "
                f"({share:.1%}) the cross-encoder term outweighs every explainable term, so the shown explanation is not "
                "the main reason for the rank (design §13.8, V2 point 6)."
            )
        notes.append(
            "- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected "
            "by construction; it shows the explained signal is necessary, not that the other signals are irrelevant."
        )
        lines.extend(notes)
        lines.extend([
            "",
            "## 4. Sample Audited Explanations",
            "",
            "| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ])
        for a in self.audited_explanations[:15]:
            cf_str = f"+{a.rank_drop} (Rank {a.counterfactual_rank})" if a.counterfactual_rank else f"+{a.rank_drop} (Fell Out)"
            g_str = "✅ Yes" if a.is_grounded else "❌ No"
            # escape pipes in explanation text
            safe_text = a.text.replace("|", "\\|")
            lines.append(f"| {a.author_id} | {a.paper_id} | {a.original_rank} | `{a.template}` | {cf_str} | {g_str} | {safe_text} |")

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Grounding Verification
# ---------------------------------------------------------------------------

def verify_grounding(
    paper_id: str,
    explanation: Explanation,
    db: Any,
    cutoff_year: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Strictly verify that the explanation text and evidence are verifiably grounded.

    - facet / new_method: Exact character offsets in abstract text and sentence boundaries.
    - citation: Real graph edges in SQLite with year <= T.
    - persona: Positive cosine similarity.
    - session: Active session item and positive cosine similarity.
    """
    feature = explanation.feature
    evidence = explanation.evidence
    details: Dict[str, Any] = {"feature": feature}

    paper = db.get_paper(paper_id)
    abstract = (paper or {}).get("abstract") or ""

    if feature == "facet":
        span = evidence.get("span")
        start = evidence.get("start_char")
        end = evidence.get("end_char")
        sentence = (evidence.get("evidence_sentence") or "").strip()

        if not abstract or not span or start is None or end is None or not sentence:
            details["error"] = "Missing abstract, span, offsets, or evidence sentence."
            return False, details

        exact_span_match = (abstract[start:end] == span)
        span_in_sentence = (span in sentence)
        sentence_in_abstract = (sentence in abstract)

        details["exact_span_match"] = exact_span_match
        details["span_in_sentence"] = span_in_sentence
        details["sentence_in_abstract"] = sentence_in_abstract

        is_valid = exact_span_match and span_in_sentence and sentence_in_abstract
        return is_valid, details

    if feature == "new_method":
        method_ev = evidence.get("method") or {}
        task_ev = evidence.get("task") or {}

        def _check_span(ev: Dict[str, Any]) -> bool:
            s, st, en = ev.get("span"), ev.get("start_char"), ev.get("end_char")
            sen = (ev.get("evidence_sentence") or "").strip()
            if not abstract or not s or st is None or en is None or not sen:
                return False
            return abstract[st:en] == s and s in sen and sen in abstract

        m_ok = _check_span(method_ev)
        t_ok = _check_span(task_ev)
        details["method_grounded"] = m_ok
        details["task_grounded"] = t_ok
        return (m_ok and t_ok), details

    if feature == "citation":
        kind = evidence.get("kind")
        authored_pid = clean_id(evidence.get("authored_paper_id", ""))
        cand_pid = clean_id(paper_id)

        if not authored_pid:
            details["error"] = "No authored paper ID."
            return False, details

        if kind == "bibliographic_coupling":
            claimed_refs = evidence.get("shared_references", 0)
            refs: Dict[str, set] = {cand_pid: set(), authored_pid: set()}
            for src, tgt in edges(db, "source_paper_id", [cand_pid, authored_pid], cutoff_year):
                refs[src].add(tgt)
            actual_refs = len(refs[cand_pid] & refs[authored_pid])
            details["claimed_refs"] = claimed_refs
            details["actual_refs"] = actual_refs
            is_valid = (actual_refs == claimed_refs and actual_refs > 0)
            return is_valid, details

        if kind == "co_citation":
            claimed_citers = evidence.get("co_citing_papers", 0)
            citers: Dict[str, set] = {cand_pid: set(), authored_pid: set()}
            for src, tgt in edges(db, "target_paper_id", [cand_pid, authored_pid], cutoff_year):
                citers[tgt].add(src)
            common_citers = (citers[cand_pid] & citers[authored_pid]) - {cand_pid, authored_pid}
            actual_citers = len(common_citers)
            details["claimed_citers"] = claimed_citers
            details["actual_citers"] = actual_citers
            is_valid = (actual_citers == claimed_citers and actual_citers > 0)
            return is_valid, details

        details["error"] = f"Unknown citation kind: {kind}"
        return False, details

    if feature == "persona":
        cos = evidence.get("persona_cosine", 0.0)
        details["persona_cosine"] = cos
        return cos > 0.0, details

    if feature == "session":
        cos = evidence.get("cosine", 0.0)
        details["session_cosine"] = cos
        return cos > 0.0, details

    details["error"] = f"Unknown feature: {feature}"
    return False, details


# ---------------------------------------------------------------------------
# Counterfactual Ranking
# ---------------------------------------------------------------------------

def compute_counterfactual_impact(
    sample: ExplanationSample,
    store: EmbeddingStore,
    k: int = 10,
    lambda_div: float = 0.2,
) -> Tuple[Optional[int], int, bool]:
    """Zero out the explained feature contribution, re-score, re-run MMR, and evaluate rank drop.

    Returns:
        (counterfactual_rank, rank_drop, is_counterfactually_necessary)
        where is_counterfactually_necessary is True if the item leaves the Top-10 or drops >= 3 ranks.
    """
    target_pid = sample.paper_id
    dom_feature = sample.dominant_feature
    pool_size = len(sample.scored_candidates)

    cf_scored: List[ScoredCandidate] = []
    for sc in sample.scored_candidates:
        if sc.paper_id == target_pid:
            cf_contributions = dict(sc.contributions)
            cf_contributions[dom_feature] = 0.0
            cf_score = float(sum(cf_contributions.values()))
            cf_sc = ScoredCandidate(
                paper_id=sc.paper_id,
                features=sc.features,
                scaled=sc.scaled,
                contributions=cf_contributions,
                score=cf_score,
                session_active=sc.session_active,
                extra=sc.extra,
            )
            cf_scored.append(cf_sc)
        else:
            cf_scored.append(sc)

    # Re-run MMR diversification
    cf_picks = select_top_k(cf_scored, store, k=k, lambda_div=lambda_div)

    cf_rank: Optional[int] = None
    for pick in cf_picks:
        if pick.paper_id == target_pid:
            cf_rank = pick.rank
            break

    orig_rank = sample.rank
    if cf_rank is None:
        # Fell out of top-K completely
        rank_drop = max(k + 1 - orig_rank, 1)
        is_necessary = True
    else:
        rank_drop = max(cf_rank - orig_rank, 0)
        is_necessary = (rank_drop >= 3)

    return cf_rank, rank_drop, is_necessary


# ---------------------------------------------------------------------------
# Sampling & Audit Runner
# ---------------------------------------------------------------------------

def sample_stratified_explanations(
    candidates: Sequence[ExplanationSample],
    target_n: int = 100,
    seed: int = 42,
) -> List[ExplanationSample]:
    """Sample explanations stratified across templates and distinct authors."""
    if not candidates:
        return []

    if len(candidates) <= target_n:
        return list(candidates)

    rng = random.Random(seed)

    # Group by template
    by_template: Dict[str, List[ExplanationSample]] = {}
    for c in candidates:
        by_template.setdefault(c.dominant_feature, []).append(c)

    templates = sorted(by_template.keys())
    sampled: List[ExplanationSample] = []
    sampled_user_item_keys = set()

    # Determine quotas per template
    # Allocate at least min(len(pool), 5) to rare templates, remaining proportionally
    quotas: Dict[str, int] = {}
    total_avail = len(candidates)

    remaining_slots = target_n
    for t in templates:
        pool = by_template[t]
        # Guarantee presence of rare templates up to available items or 5
        base = min(len(pool), 5)
        quotas[t] = base
        remaining_slots -= base

    # Proportional allocation for remainder
    if remaining_slots > 0:
        shares = {t: len(by_template[t]) / total_avail for t in templates}
        for t in templates:
            extra = int(round(shares[t] * remaining_slots))
            extra = min(extra, len(by_template[t]) - quotas[t])
            quotas[t] += max(extra, 0)

    # Adjust sum to exactly target_n if rounding caused deviation
    curr_sum = sum(quotas.values())
    while curr_sum < target_n:
        eligible = [t for t in templates if quotas[t] < len(by_template[t])]
        if not eligible:
            break
        best_t = max(eligible, key=lambda t: len(by_template[t]) - quotas[t])
        quotas[best_t] += 1
        curr_sum += 1

    while curr_sum > target_n:
        eligible = [t for t in templates if quotas[t] > 1]
        if not eligible:
            break
        worst_t = max(eligible, key=lambda t: quotas[t])
        quotas[worst_t] -= 1
        curr_sum -= 1

    # Now sample from each template pool, prioritizing user diversity
    for t in templates:
        pool = list(by_template[t])
        rng.shuffle(pool)

        # Sort pool to prioritize users not yet heavily represented
        user_counts: Dict[str, int] = {}
        for item in pool:
            user_counts[item.author_id] = user_counts.get(item.author_id, 0)

        # Pick items prioritizing lower-frequency users
        picked_for_t: List[ExplanationSample] = []
        user_picks: Dict[str, int] = {}

        # First pass: at most 1 item per user
        for item in pool:
            if len(picked_for_t) >= quotas[t]:
                break
            if user_picks.get(item.author_id, 0) == 0:
                picked_for_t.append(item)
                user_picks[item.author_id] = 1

        # Second pass: if quota not met, allow additional items per user
        if len(picked_for_t) < quotas[t]:
            for item in pool:
                if len(picked_for_t) >= quotas[t]:
                    break
                if item not in picked_for_t:
                    picked_for_t.append(item)

        sampled.extend(picked_for_t)

    # Final shuffle
    rng.shuffle(sampled)
    return sampled[:target_n]


def run_faithfulness_audit(
    resources: RecommenderResources,
    test_authors: Sequence[Mapping[str, Any]],
    config: Optional[RecommendConfig] = None,
    sample_size: int = 100,
    seed: int = 42,
    max_authors: Optional[int] = None,
) -> FaithfulnessAuditResult:
    """Run the complete Phase 8 Explanation Faithfulness Audit across test authors."""
    cfg = config or RecommendConfig()
    authors_to_eval = list(test_authors)
    if max_authors is not None:
        authors_to_eval = authors_to_eval[:max_authors]

    logger.info("Generating recommendations across %d test authors for explanation audit...", len(authors_to_eval))

    candidate_samples: List[ExplanationSample] = []
    population_templates: Dict[str, int] = {}
    total_recs = 0
    covered_recs = 0
    reranker_dominant = 0

    for idx, author_info in enumerate(authors_to_eval):
        aid = clean_id(author_info["author_id"])
        authored_ids = [clean_id(p) for p in author_info.get("authored_paper_ids_T", [])]
        seed_topics = author_info.get("seed_topics", [])

        profile = build_user_profile(resources, aid, authored_ids, seed_topics, cfg)

        # Generate recommendation with candidates and features
        res = recommend(resources, profile, session=None, mode="feed", config=cfg)

        # The counterfactual must re-rank exactly what produced this feed: the scored pool and the effective weights
        # (adaptive_graph changes w_cite for sparse authors). Re-ranking it unchanged must give back the feed.
        scored_candidates = res.scored_pool
        replay = [p.paper_id for p in select_top_k(scored_candidates, resources.store, k=cfg.k, lambda_div=cfg.lambda_div)]
        if replay != res.paper_ids:
            raise RuntimeError(
                f"Faithfulness audit: re-ranking the scored pool of author {aid} does not reproduce the feed; "
                "the counterfactual would measure the wrong ranking."
            )

        for item in res.items:
            total_recs += 1
            if cfg.cross_encoder:
                explainable = max((item.contributions.get(f, 0.0) for f in EXPLAINABLE_FEATURES), default=0.0)
                if item.contributions.get("cross_encoder", 0.0) > explainable + 1e-12:
                    reranker_dominant += 1
            exp_res = item.explanation
            if exp_res and exp_res.covered and exp_res.dominant_feature:
                population_templates[exp_res.dominant_feature] = population_templates.get(exp_res.dominant_feature, 0) + 1
            if exp_res and exp_res.covered and exp_res.explanation and exp_res.dominant_feature:
                covered_recs += 1
                candidate_samples.append(
                    ExplanationSample(
                        author_id=aid,
                        paper_id=item.paper_id,
                        rank=item.rank,
                        dominant_feature=exp_res.dominant_feature,
                        explanation=exp_res.explanation,
                        scored_candidates=scored_candidates,
                    )
                )

    coverage_rate = covered_recs / max(total_recs, 1)
    logger.info("Explanation coverage across %d recommendations: %.2f%% (%d/%d)", total_recs, coverage_rate * 100.0, covered_recs, total_recs)

    # Stratified sampling of target_n explanations
    sampled = sample_stratified_explanations(candidate_samples, target_n=sample_size, seed=seed)
    logger.info("Sampled %d explanations stratified across templates and authors.", len(sampled))

    # Audit each sampled explanation
    audited_list: List[AuditedExplanation] = []
    grounded_count = 0
    dropout_count = 0
    left_top_k = 0
    dropped_gte_3 = 0
    retained_near = 0
    rank_drops: List[int] = []
    template_counts: Dict[str, int] = {}
    per_template: Dict[str, Dict[str, int]] = {}

    for sample in sampled:
        dom = sample.dominant_feature
        template_counts[dom] = template_counts.get(dom, 0) + 1
        stats = per_template.setdefault(dom, {"n": 0, "grounded": 0, "necessary": 0, "left_top_k": 0, "dropped_gte_3": 0})
        stats["n"] += 1

        # 1. Grounding check
        is_grounded, grounding_details = verify_grounding(
            sample.paper_id, sample.explanation, resources.db, resources.cutoff_year
        )
        if is_grounded:
            grounded_count += 1
            stats["grounded"] += 1

        # 2. Counterfactual impact
        cf_rank, r_drop, is_necessary = compute_counterfactual_impact(
            sample, resources.store, k=cfg.k, lambda_div=cfg.lambda_div
        )
        rank_drops.append(r_drop)

        if is_necessary:
            dropout_count += 1
            stats["necessary"] += 1
            if cf_rank is None or cf_rank > cfg.k:
                left_top_k += 1
                stats["left_top_k"] += 1
            else:
                dropped_gte_3 += 1
                stats["dropped_gte_3"] += 1
        else:
            retained_near += 1

        audited_list.append(
            AuditedExplanation(
                author_id=sample.author_id,
                paper_id=sample.paper_id,
                original_rank=sample.rank,
                template=dom,
                text=sample.explanation.text,
                is_grounded=is_grounded,
                grounding_details=grounding_details,
                counterfactual_rank=cf_rank,
                rank_drop=r_drop,
                is_counterfactually_necessary=is_necessary,
                evidence=sample.explanation.evidence,
            )
        )

    n_samp = len(sampled)
    grounding_rate = grounded_count / max(n_samp, 1)
    dropout_rate = dropout_count / max(n_samp, 1)
    mean_drop = float(np.mean(rank_drops)) if rank_drops else 0.0
    median_drop = float(np.median(rank_drops)) if rank_drops else 0.0

    proportions = {t: c / max(n_samp, 1) for t, c in template_counts.items()}
    unique_users = len(set(s.author_id for s in sampled))

    return FaithfulnessAuditResult(
        n_sampled=n_samp,
        n_users_in_sample=unique_users,
        total_evaluated_recommendations=total_recs,
        total_covered_recommendations=covered_recs,
        explanation_coverage=coverage_rate,
        extractive_grounding_rate=grounding_rate,
        counterfactual_dropout_rate=dropout_rate,
        left_top_k_count=left_top_k,
        dropped_gte_3_count=dropped_gte_3,
        retained_near_rank_count=retained_near,
        mean_rank_drop=mean_drop,
        median_rank_drop=median_drop,
        template_distribution=template_counts,
        template_proportions=proportions,
        audited_explanations=audited_list,
        per_template=per_template,
        population_template_distribution=population_templates,
        n_reranker_dominant=reranker_dominant if cfg.cross_encoder else None,
    )

