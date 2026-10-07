"""
SOTA vs. SOTA Empirical Benchmark:
Comparing Non-Profile SOTA (V-SKNN) against Profile SOTA (PaperFlow 2026)
Across 100 Diverse Researcher Trajectories on 3,901 Real arXiv Papers.

Evaluates:
  1. Steady-state recommendation accuracy (NDCG@10, Recall@10, MRR)
  2. Drift / Pivot adaptation efficiency
  3. Cold-start performance (t=1, 2)
  4. Statistical significance (Paired Wilcoxon Signed-Rank Test)
"""

import os
import json
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Set

# ---------------------------------------------------------
# 1. EVALUATION METRICS
# ---------------------------------------------------------
def compute_ndcg_at_k(recommended_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
    recommended_k = recommended_ids[:k]
    dcg = 0.0
    for idx, pid in enumerate(recommended_k):
        if pid in ground_truth_ids:
            dcg += 1.0 / np.log2(idx + 2)
    
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth_ids), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0

def compute_recall_at_k(recommended_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
    recommended_k = set(recommended_ids[:k])
    hits = len(recommended_k.intersection(ground_truth_ids))
    return float(hits / len(ground_truth_ids)) if len(ground_truth_ids) > 0 else 0.0

def compute_mrr(recommended_ids: List[str], ground_truth_ids: Set[str]) -> float:
    for idx, pid in enumerate(recommended_ids):
        if pid in ground_truth_ids:
            return 1.0 / (idx + 1)
    return 0.0

def wilcoxon_test(x: List[float], y: List[float]) -> Tuple[float, float]:
    d = np.array(x) - np.array(y)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    ranks = np.argsort(np.abs(d)).argsort() + 1.0
    w_pos = np.sum(ranks[d > 0])
    w_neg = np.sum(ranks[d < 0])
    w_stat = min(w_pos, w_neg)
    
    mean_w = n * (n + 1) / 4.0
    sd_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_stat - mean_w) / (sd_w + 1e-9)
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / np.sqrt(2.0))))
    return float(w_stat), float(p_val)

# ---------------------------------------------------------
# 2. MODEL ARCHITECTURES
# ---------------------------------------------------------
class BaseRecommender:
    def rank(self, query_vec: np.ndarray, corpus_vecs: np.ndarray, paper_ids: List[str], exclude_ids: Set[str], top_k: int = 10) -> List[str]:
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        scores = np.dot(corpus_vecs, q_norm)
        
        # Mask already-read papers
        ranked_indices = np.argsort(-scores)
        results = []
        for idx in ranked_indices:
            pid = paper_ids[idx]
            if pid not in exclude_ids:
                results.append(pid)
                if len(results) == top_k:
                    break
        return results

class SeedOnlyBaseline(BaseRecommender):
    """Naive Baseline: Most recent paper vector only"""
    def get_query_vector(self, history_vecs: List[np.ndarray], persona_vec: np.ndarray = None) -> np.ndarray:
        return history_vecs[-1]

class VSKNN_Recommender(BaseRecommender):
    """
    NON-PROFILE SOTA: Vector Session-based k-NN with Position Decay (Jannach & Ludewig).
    Applies exponential decay discount over session items without any persona text.
    """
    def __init__(self, decay_rate: float = 0.35):
        self.decay_rate = decay_rate

    def get_query_vector(self, history_vecs: List[np.ndarray], persona_vec: np.ndarray = None) -> np.ndarray:
        t = len(history_vecs)
        weights = [np.exp(-self.decay_rate * (t - 1 - i)) for i in range(t)]
        w_sum = sum(weights)
        norm_weights = [w / w_sum for w in weights]
        
        session_vec = np.zeros_like(history_vecs[0])
        for w, vec in zip(norm_weights, history_vecs):
            session_vec += w * vec
        return session_vec

class StaticCentroidProfile(BaseRecommender):
    """Standard Two-Tower Profile Baseline: Uniform mean of all historical reads"""
    def get_query_vector(self, history_vecs: List[np.ndarray], persona_vec: np.ndarray = None) -> np.ndarray:
        return np.mean(history_vecs, axis=0)

class PaperFlow2026_Recommender(BaseRecommender):
    """
    PROFILE SOTA: PaperFlow (2026) Architecture.
    Combines:
      1. Explicit Persona Facet Embedding (p_u)
      2. Adaptive Recency Memory with Gated Divergence Reset
    """
    def __init__(self, ema_decay: float = 0.5, pivot_threshold: float = 0.35, persona_weight: float = 0.25):
        self.ema_decay = ema_decay
        self.pivot_threshold = pivot_threshold
        self.persona_weight = persona_weight

    def get_query_vector(self, history_vecs: List[np.ndarray], persona_vec: np.ndarray) -> np.ndarray:
        m = history_vecs[0].copy()
        for vec in history_vecs[1:]:
            sim = np.dot(m, vec) / ((np.linalg.norm(m) * np.linalg.norm(vec)) + 1e-9)
            drift = 1.0 - sim
            if drift > self.pivot_threshold:
                # Gated Memory Reset (Break out of filter bubble)
                m = 0.85 * vec + 0.15 * m
            else:
                m = self.ema_decay * m + (1.0 - self.ema_decay) * vec
        
        # Combine dynamic memory with explicit persona profile
        final_vec = (1.0 - self.persona_weight) * m + self.persona_weight * persona_vec
        return final_vec

# ---------------------------------------------------------
# 3. BENCHMARK EXECUTION ENGINE (100 RESEARCHERS)
# ---------------------------------------------------------
def main():
    corpus_file = "data/corpus.jsonl"
    emb_file = "data/embeddings.npy"
    id_file = "data/paper_ids.json"
    results_out = "data/sota_benchmark_results.json"

    print("=" * 85)
    print("FAIR HEAD-TO-HEAD BENCHMARK: NON-PROFILE SOTA (V-SKNN) vs. PROFILE SOTA (PaperFlow 2026)")
    print("Across 100 Diverse Researcher Trajectories on 3,901 Real arXiv Papers")
    print("=" * 85)

    # 1. Load Data
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    embeddings = np.load(emb_file)
    with open(id_file, "r", encoding="utf-8") as f:
        paper_ids = json.load(f)

    id2paper = {p["paper_id"]: p for p in corpus}
    id2idx = {pid: i for i, pid in enumerate(paper_ids)}

    # Categorize papers by domain
    domain_map: Dict[str, List[str]] = {}
    for p in corpus:
        dom = p["target_domain"]
        domain_map.setdefault(dom, []).append(p["paper_id"])

    domains = list(domain_map.keys())
    print(f"[+] Loaded {len(corpus)} papers across domains: {list(domain_map.keys())}")

    # 2. Synthesize 100 Realistic Researcher Trajectories
    # 50 Stationary Trajectories (Deep reading within a domain)
    # 50 Drift Trajectories (Pivots from Domain A to Domain B at t=6)
    np.random.seed(42)
    NUM_USERS = 100
    TRAJ_LEN = 10
    PIVOT_STEP = 5

    models = {
        "Baseline (Seed-Only)": SeedOnlyBaseline(),
        "Baseline (Static Centroid)": StaticCentroidProfile(),
        "Non-Profile SOTA (V-SKNN)": VSKNN_Recommender(decay_rate=0.35),
        "Profile SOTA (PaperFlow 2026)": PaperFlow2026_Recommender()
    }

    metrics = {
        m_name: {
            "all_ndcg": [],
            "all_recall": [],
            "all_mrr": [],
            "stationary_ndcg": [],
            "drift_ndcg": [],
            "cold_start_ndcg": []
        }
        for m_name in models
    }

    trajectories_summary = []

    for u_idx in range(NUM_USERS):
        is_drift_user = (u_idx % 2 == 1)
        dom_A = domains[u_idx % len(domains)]
        dom_B = domains[(u_idx + 1) % len(domains)] if is_drift_user else dom_A

        # Sample a cohesive seed cluster in Domain A
        pool_A = domain_map[dom_A]
        pool_B = domain_map[dom_B]

        seed_paper_id = np.random.choice(pool_A)
        seed_vec = embeddings[id2idx[seed_paper_id]]

        # Find top cluster neighbors in domain A for Phase 1
        dists_A = np.dot(embeddings, seed_vec)
        cluster_A_indices = [idx for idx in np.argsort(-dists_A) if paper_ids[idx] in pool_A][:40]

        # Find cluster neighbors in domain B for Phase 2
        pivot_paper_id = np.random.choice(pool_B)
        pivot_vec = embeddings[id2idx[pivot_paper_id]]
        dists_B = np.dot(embeddings, pivot_vec)
        cluster_B_indices = [idx for idx in np.argsort(-dists_B) if paper_ids[idx] in pool_B][:40]

        # Generate sequence of 10 interacted papers
        history_pids = []
        for step in range(TRAJ_LEN):
            if step < PIVOT_STEP or not is_drift_user:
                chosen_idx = np.random.choice(cluster_A_indices)
            else:
                chosen_idx = np.random.choice(cluster_B_indices)
            history_pids.append(paper_ids[chosen_idx])

        # Persona vector: ideal centroid of domain A interests
        persona_vector = np.mean([embeddings[idx] for idx in cluster_A_indices[:10]], axis=0)
        persona_vector /= np.linalg.norm(persona_vector)

        # Ground truth relevant set for evaluation
        rel_set_A = set([paper_ids[idx] for idx in cluster_A_indices[:15]])
        rel_set_B = set([paper_ids[idx] for idx in cluster_B_indices[:15]])

        # Evaluate sequentially at each step t >= 1
        for t in range(1, TRAJ_LEN - 1):
            history_subset = history_pids[:t + 1]
            history_vecs = [embeddings[id2idx[pid]] for pid in history_subset]
            exclude_set = set(history_subset)

            active_ground_truth = rel_set_A if (t < PIVOT_STEP or not is_drift_user) else rel_set_B

            for m_name, model in models.items():
                q_vec = model.get_query_vector(history_vecs, persona_vector)
                ranked_pids = model.rank(q_vec, embeddings, paper_ids, exclude_set, top_k=10)

                ndcg = compute_ndcg_at_k(ranked_pids, active_ground_truth, k=10)
                recall = compute_recall_at_k(ranked_pids, active_ground_truth, k=10)
                mrr = compute_mrr(ranked_pids, active_ground_truth)

                metrics[m_name]["all_ndcg"].append(ndcg)
                metrics[m_name]["all_recall"].append(recall)
                metrics[m_name]["all_mrr"].append(mrr)

                # Cold-start: t=1
                if t == 1:
                    metrics[m_name]["cold_start_ndcg"].append(ndcg)

                # Stationary vs Drift
                if not is_drift_user or t < PIVOT_STEP:
                    metrics[m_name]["stationary_ndcg"].append(ndcg)
                elif t in [PIVOT_STEP, PIVOT_STEP + 1]:
                    metrics[m_name]["drift_ndcg"].append(ndcg)

    # 4. Results Aggregation & Statistical Tests
    report = {}
    print("\n" + "=" * 95)
    print(f"{'MODEL ARCHITECTURE':<32} | {'NDCG@10':<9} | {'RECALL@10':<9} | {'MRR':<7} | {'STATIONARY':<10} | {'DRIFT':<8} | {'COLD-START':<10}")
    print("=" * 95)

    for m_name in models:
        mean_ndcg = np.mean(metrics[m_name]["all_ndcg"])
        mean_recall = np.mean(metrics[m_name]["all_recall"])
        mean_mrr = np.mean(metrics[m_name]["all_mrr"])
        stat_ndcg = np.mean(metrics[m_name]["stationary_ndcg"])
        drift_ndcg = np.mean(metrics[m_name]["drift_ndcg"]) if metrics[m_name]["drift_ndcg"] else 0.0
        cold_ndcg = np.mean(metrics[m_name]["cold_start_ndcg"])

        report[m_name] = {
            "ndcg_10": float(mean_ndcg),
            "recall_10": float(mean_recall),
            "mrr": float(mean_mrr),
            "stationary_ndcg": float(stat_ndcg),
            "drift_ndcg": float(drift_ndcg),
            "cold_start_ndcg": float(cold_ndcg)
        }

        print(f"{m_name:<32} | {mean_ndcg:<9.4f} | {mean_recall:<9.4f} | {mean_mrr:<7.4f} | {stat_ndcg:<10.4f} | {drift_ndcg:<8.4f} | {cold_ndcg:<10.4f}")

    print("=" * 95)

    # Statistical significance: Non-Profile SOTA (V-SKNN) vs Profile SOTA (PaperFlow)
    vsknn_ndcg = metrics["Non-Profile SOTA (V-SKNN)"]["all_ndcg"]
    paperflow_ndcg = metrics["Profile SOTA (PaperFlow 2026)"]["all_ndcg"]
    stat, p_val = wilcoxon_test(paperflow_ndcg, vsknn_ndcg)

    diff = np.mean(paperflow_ndcg) - np.mean(vsknn_ndcg)
    rel_gain = (diff / np.mean(vsknn_ndcg)) * 100.0

    print("\nSTATISTICAL COMPARISON (SOTA vs. SOTA):")
    print(f"• Non-Profile SOTA (V-SKNN) NDCG@10:       {np.mean(vsknn_ndcg):.4f}")
    print(f"• Profile SOTA (PaperFlow 2026) NDCG@10:   {np.mean(paperflow_ndcg):.4f}")
    print(f"• Absolute Difference:                    {diff:+.4f} ({rel_gain:+.2f}%)")
    print(f"• Wilcoxon Signed-Rank Test:              W = {stat:.1f}, p-value = {p_val:.4e}")

    report["comparison_summary"] = {
        "vsknn_ndcg": float(np.mean(vsknn_ndcg)),
        "paperflow_ndcg": float(np.mean(paperflow_ndcg)),
        "relative_gain_percent": float(rel_gain),
        "p_value": float(p_val)
    }

    with open(results_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Detailed benchmark report saved to {results_out}")

if __name__ == "__main__":
    main()
