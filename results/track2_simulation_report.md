# Track 2: LLM Session Simulation Report

- **Simulator Model:** `minimaxai/minimax-m3.1-flash`
- **Simulation Paradigm:** Agent4Rec / RecAgent-style LLM Researcher Simulation
- **Evaluation Units:** 15 personas x 3 seeds = 45 units per arm (Pre-registered budget fallback applied per docs/design.md §13.2 point 7: 15 personas x 3 seeds)
- **Protocol:** 5 Topic A turns -> Topic B Shift search -> 10 Topic B turns -> Topic A Re-engagement search -> 3 Re-engagement turns
- **Status:** Complete (Verified zero cost, 100% reproducible with released rating cache)

## 1. Summary of Arms

| Arm | Mean Pivot Latency $k_{\text{pivot}}$ (turns) | Retention@1 | Retention@2 | Retention@3 |
| :--- | :---: | :---: | :---: | :---: |
| **persona_only** | 10.00 | 0.513 | 0.513 | 0.513 |
| **session_only** | 8.78 | 0.609 | 0.629 | 0.624 |
| **static_blend** | 10.00 | 0.536 | 0.544 | 0.538 |
| **hybrid** | 9.38 | 0.518 | 0.578 | 0.578 |
| **hybrid_gated** | 9.38 | 0.518 | 0.578 | 0.578 |

## 2. Hypothesis $H_3$ Testing (Paired Bootstrap, $N=2000$, 95% CI)

| Hypothesis | Comparison | Metric | Mean Diff | 95% CI | Result |
| :--- | :--- | :--- | :---: | :---: | :---: |
| $H_{3a}$ | Hybrid vs. Persona-Only | Pivot Latency reduction ($k_{\text{pivot}}$ turns) | +0.622 | [+0.000, +1.711] | Not Supported |
| $H_{3b}$ | Hybrid vs. Static-Blend | Pivot Latency reduction ($k_{\text{pivot}}$ turns) | +0.622 | [+0.000, +1.711] | Not Supported |
| $H_{3c}$ | Hybrid vs. Session-Only | Retention@3 gain | -0.047 | [-0.171, +0.076] | Not Supported |

## 3. Findings and Honest Interpretation

- **Pivot Dynamics:** Persona-Only has no session input and stays censored at $k_{\text{pivot}}=10.0$ turns. The Hybrid system exhibits lower mean latency than Persona-Only in aggregate, but when properly cluster-bootstrapped by persona (accounting for seed dependence per persona), the 95% CI reflects that pivot behavior is concentrated in specific persona sub-trajectories rather than occurring universally.
- **Topic Pivot Gate:** At the evaluated threshold $\gamma = 0.35$, the topic shift cosine delta rarely breached the threshold in the test window, resulting in Hybrid and Hybrid+Gate exhibiting identical or near-identical behaviors.
- **Retention Dynamics:** Session-Only retains high topic re-engagement (Retention@3) through reactive session steering. The Hybrid system balances persona and session vectors, but does not demonstrate a statistically significant retention advantage over Session-Only ($H_{3c}$ is not supported).
- **Model Disclosure:** The simulation was executed with `minimaxai/minimax-m3.1-flash` (NVIDIA API). All ratings were generated with deterministic seed-anchored prompts and cached for zero-cost exact reproducibility.

> [!NOTE]
> Track 2 results serve as a behavioral simulation check of session adaptation dynamics in accordance with docs/design.md §13.2. Track 2 numbers are strictly separated from Track 1 historical offline replay numbers.

