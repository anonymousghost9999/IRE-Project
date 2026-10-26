# Benchmarking Datasets and User Interaction Modeling for Paper Feed

> **Purpose:** Comprehensive reference guide on acquiring, simulating, and evaluating user interaction data (clicks, reads, bookmarks, and skips) for personalized academic paper recommendation in **Paper Feed**.

---

## 1. Executive Summary & The Scientific IR Reality

In domains like e-commerce (Amazon, Taobao) or news recommendation (Microsoft MIND), user clickstream datasets containing impressions, clicks, dwell times, and skips are widely published.

However, in **academic paper recommendation**, platforms like Google Scholar, Semantic Scholar, and arXiv **never release raw session clickstreams** due to:
1. Strict researcher privacy and institutional IP considerations.
2. Proprietary search-ranking data protection.
3. Lack of unified public web telemetry across decentralized publisher paywalls.

To evaluate personalized paper recommenders like **PaperFlow (2026)** and **V-SKNN** scientifically, the Information Retrieval (IR) and RecSys communities rely on **four complementary methodologies**:
1. **Bibliographic Citation Trajectories (Author Timelines):** The strongest ground truth of actual papers read and endorsed.
2. **Personal Reference Manager Libraries:** Real researchers saving papers into tools like CiteULike and Mendeley.
3. **Simulated Longitudinal Researcher Trajectories:** The SOTA evaluation methodology introduced by PaperFlow (2026).
4. **Interactive Application Telemetry:** Live feedback captured directly in the system UI.

---

## 2. Public Datasets for Academic Paper Recommendation

The following table summarizes the primary public datasets available for scientific paper recommendation:

| Dataset | Primary Source | Scale | Signal Type | Negatives / Skips Available? | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAlex Citation Graph** | OpenAlex API | 250M+ works, 100M+ authors | Real Citations & Author Timelines | Derived (Uncited in Top-K) | **Primary Corpus & Ground Truth** |
| **CiteULike (`CiteULike-a` / `CiteULike-t`)** | CiteULike Service | 5,551 users, 16,980 papers, 204k events | Implicit Bookmarks ($1=\text{Saved}$) | Implicit (Unsaved candidates) | Standard RecSys Benchmark |
| **Mendeley User Collections** | Mendeley Reference Manager | 100k+ users, millions of records | Reference Manager Libraries | Implicit (Unsaved candidates) | Large-scale Collaborative Filtering |
| **Scholar Inbox (ACL 2025 Demo)** | Scholar Inbox Web Service | Real researcher feedback | Explicit Ratings (👍 Up / 👎 Down) | Explicit (Downvotes / Dismissals) | Explicit Preference Modeling |
| **PaperFlow 2026 Benchmark** | arXiv:2606.07454 | 24 personas, 1,200 episodes | Sequential Reading & Topic Shifts | Simulated Skips & Reads | **Drift & Cold-Start Adaptation** |

---

### Detailed Breakdown of Public Datasets

#### 1. CiteULike (`CiteULike-a` and `CiteULike-t`)
* **Overview:** The classic academic recommendation benchmark established by Wang & Blei (Collaborative Topic Modeling). It consists of real researchers bookmarking papers into their CiteULike academic libraries.
* **Format:**
  * `users.dat`: User ID mapping.
  * `raw-text.dat`: Paper titles, abstracts, and metadata.
  * `mult.dat`: Bag-of-words / text tokens per paper.
  * `users-papers.dat`: User-paper interaction matrix ($u, i$).
* **How to use it:**
  * Treat each user's bookmarked papers as positive reads.
  * Order interactions chronologically if timestamps are available, or evaluate as Top-$K$ retrieval against held-out test libraries.
* **Access:** Publicly hosted in RecSys repositories and on GitHub (e.g., [CiteULike Data Repository](https://github.com/weiguang/citeulike)).

#### 2. Mendeley Reference Manager Public Datasets
* **Overview:** Mendeley periodically released anonymized datasets of academic papers saved in public researcher groups.
* **Contents:** User IDs, discipline tags, group memberships, and lists of saved document IDs.
* **How to use it:** Excellent for testing collaborative filtering and community-based topic matching, though text abstracts often need to be matched back to DOI/OpenAlex.

#### 3. Scholar Inbox (ACL 2025 Demo)
* **Overview:** Built by researchers to recommend daily arXiv papers. Authors interact with recommendations by upvoting, bookmarking, or dismissing papers.
* **Key Advantage:** Contains explicit negative feedback signals (papers explicitly disliked or hidden by real scientists).

#### 4. PaperFlow (2026) Longitudinal Benchmark
* **Overview:** The official benchmark dataset accompanying the PaperFlow paper (arXiv:2606.07454).
* **Contents:**
  * 24 structured researcher personas (defined by task, methodology, and domain preferences).
  * 1,200 sequential user-day reading episodes.
  * Explicit topic drift events (where a researcher abruptly shifts domains).
* **Key Advantage:** The exact benchmark designed to test gated streaming memory against filter-bubble inertia.

---

## 3. Four Actionable Implementation Options for Paper Feed

Here is how to implement each option in Paper Feed:

---

### Option 1: OpenAlex Author Citation Timelines *(Recommended Primary Strategy)*

#### Why It Is Best:
You already crawl OpenAlex for your paper corpus. Using OpenAlex author publication histories requires **zero third-party downloads**, guarantees 100% ID compatibility, and is the most scientifically defensible methodology in academic IR.

#### The Core Insight:
When an author publishes a paper at year $t$, its bibliography (`referenced_works`) represents the exact papers the author **actually read, studied, and cited** to produce that work.

#### Step-by-Step Pipeline:
```text
1. Select 100-500 Active Authors from OpenAlex
   (Filtered by Concept: Computer Science / Machine Learning / IR)
                            │
                            ▼
2. Fetch Chronological Publication History for Each Author
   • Paper P_1 (2021) ──► Cited: [C_1, C_2, C_3]
   • Paper P_2 (2022) ──► Cited: [C_4, C_5]
   • Paper P_3 (2024) ──► Cited: [C_6, C_7, C_8]  <-- Target Publication
                            │
                            ▼
3. Partition by Strict Temporal Cutoff (Section 25 Policy):
   • History / Session Input (t <= 2023):  [C_1, C_2, C_3, C_4, C_5]
     --> Used to build Persona (p_u) and initialize Session Memory (m_t)
   • Target Positive Ground Truth (t = 2024): [C_6, C_7, C_8]
     --> What the recommender must retrieve in Top-10!
                            │
                            ▼
4. Formulate Negative / Skip Feedback:
   • Papers recommended in Top-10 that the author NEVER cited
     and belong to distant topic clusters = IMPLICIT SKIPS.
```

#### OpenAlex Python Acquisition Script Example:
```python
import requests
import json

# Fetch an author's publication timeline and cited works
def fetch_author_timeline(author_id: str):
    url = f"https://api.openalex.org/works?filter=author.id:{author_id}&sort=publication_year:asc"
    resp = requests.get(url).json()
    
    timeline = []
    for work in resp.get("results", []):
        timeline.append({
            "paper_id": work["id"],
            "title": work["title"],
            "year": work["publication_year"],
            "cited_references": work.get("referenced_works", []) # Papers they read/cited!
        })
    return timeline
```

---

### Option 2: Simulated Longitudinal Trajectories *(What Currently Runs in `plan.md`)*

#### Why It Works:
This is the protocol already implemented in [`scripts/benchmark_sota_vs_sota.py`](file:///d:/IRE%20Project/IRE-Project/scripts/benchmark_sota_vs_sota.py). It models realistic researcher behavior across 100 reading trajectories ($T=10$ sequential steps each).

#### Experimental Design:
1. **50 Stationary Trajectories:** Researchers specializing consistently within one sub-domain (e.g., Dense Retrieval in `cs.IR`).
2. **50 Topic Drift Trajectories:** Researchers who abruptly pivot their focus at step $t=5$ (e.g., from Information Retrieval to Computer Vision).

#### Simulation Mechanics:
* At step $t$, the recommender proposes Top-10 candidates.
* **Positive Interaction ("Read"):** Paper in the candidate list matching the active cluster is selected ($\mathbf{d}_{(t)}$) and updates streaming memory $\mathbf{m}_t$.
* **Negative Interaction ("Skip"):** Off-target or irrelevant candidates in the Top-10 are counted as skips.
* **Drift Adaptation:** Measures how quickly PaperFlow resets memory ($\mathbf{m}_t = 0.85\mathbf{d}_t + 0.15\mathbf{m}_{t-1}$) compared to static baselines.

---

### Option 3: Third-Party Benchmark using `CiteULike-a`

If an external validation benchmark is requested by evaluators:

1. **Download:** Download `CiteULike-a` dataset (`users-papers.dat` and `raw-text.dat`).
2. **Encode:** Encode CiteULike paper abstracts using the same paper encoder (`bge-small` or `SPECTER2`).
3. **Partition:** For each user with $\ge 10$ saved papers:
   * First 70% of saved papers $\rightarrow$ Input reading sequence $\mathcal{H}_t$.
   * Last 30% of saved papers $\rightarrow$ Ground-truth positive test set.
4. **Evaluate:**
   * Compute Recall@10, NDCG@10, and MRR.
   * Compare PaperFlow vs. V-SKNN vs. Static Centroid.

---

### Option 4: Live Telemetry in the Streamlit Interactive UI (`ui/app.py`)

For the live project demonstration, human users generate real interaction telemetry:

#### 1. UI Interaction Controls:
Each recommended paper card in the feed displays action buttons:
* 📖 **Read Full Abstract** $\rightarrow$ Positive signal (Moderate weight).
* 🔖 **Save to Library** $\rightarrow$ Strong positive signal (Anchors persona $\mathbf{p}_u$).
* ⏭️ **Skip** $\rightarrow$ Weak negative signal (Downweights candidate cluster).
* 🚫 **Not Interested / Dismiss** $\rightarrow$ Strong negative signal (Excluded from future candidate sets).

#### 2. SQLite Telemetry Schema:
```sql
CREATE TABLE IF NOT EXISTS user_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    action TEXT CHECK(action IN ('read', 'save', 'skip', 'dismiss')),
    dwell_time_seconds INTEGER DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 3. Real-Time Memory Feedback Loop:
When the user clicks an action in Streamlit:
1. The event is written to SQLite `user_interactions`.
2. If `action == 'read'` or `'save'`:
   * Fetch embedding $\mathbf{d}_{\text{paper}}$.
   * Compute divergence $\Delta_t = 1 - \cos(\mathbf{d}_{\text{paper}}, \mathbf{m}_{t-1})$.
   * Update $\mathbf{m}_t$ and re-rank the feed immediately!
3. If `action == 'skip'` or `'dismiss'`:
   * Add `paper_id` to session exclusion mask.

---

## 4. How "Skips" and Negative Signals are Defined

| Scenario | Positive Signal ("Read") | Implicit Skip ("Ignored") | Explicit Negative ("Dismiss") |
| :--- | :--- | :--- | :--- |
| **OpenAlex Timeline (Option 1)** | Papers cited in target publication | Candidate papers retrieved in Top-10 but never cited | Candidate papers from completely unrelated disciplines |
| **Simulation (Option 2)** | Candidate paper matching active cluster | Candidate papers in Top-10 not matching active cluster | Candidates from the discarded topic during pivot |
| **CiteULike (Option 3)** | Papers saved in user library | Papers in corpus not saved by user | Randomly sampled unread papers (standard negative sampling) |
| **Live UI (Option 4)** | User clicks "Read" or "Bookmark" | User scrolls past without clicking | User clicks "Skip" or "Not Interested" button |

---

## 5. Summary Recommendation for Your Project Report & Presentation

For the IRE semester project submission and presentation:

1. **Lead with Option 1 (OpenAlex Citation Graph):**
   * Emphasize: *"We use real OpenAlex citation bibliographies as ground-truth reading records, adhering to a strict temporal cutoff ($t \le 2023$ for input, $t=2024$ for prediction) to avoid temporal data leakage."*
2. **Present the Head-to-Head Simulation (Option 2):**
   * Showcase the 100-user empirical benchmark from [`plan.md`](file:///d:/IRE%20Project/IRE-Project/plan.md), demonstrating that PaperFlow's gated divergence reset achieves **+33% NDCG@10 overall** and **+68% during cold start** compared to V-SKNN.
3. **Demonstrate Option 4 Live in the Web UI:**
   * Let the professor or TA interact with the Streamlit app live: click a paper, trigger a topic pivot, and watch the feed adapt in real time!
