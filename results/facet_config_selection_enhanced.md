# Facet-Active Variant Selection (tuning split)

Rule (design §9): among grid points with w_facet > 0 in order of tuning-split NDCG@10, the first whose relative NDCG@10 loss against the optimum has a 95% paired-bootstrap CI upper end <= 5% and whose Top-10 items have >= 10% facet or new-method explanations. 80 tuning authors, 10,000 resamples.

- Configuration: pool 250, channel depth 500, w_pop 0.2, adaptive_graph True
- Grid optimum (the headline system): w_cite=1.0, w_facet=0.0, w_nov=0.25, lambda_div=0.0, NDCG@10 0.0308, templates citation 29%, persona 71%

| w_cite | w_facet | w_nov | lambda_div | NDCG@10 | Rel. loss [95% CI] | Facet / new-method share | Non-inferior | Enough facets | Accepted |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1.0 | 0.25 | 0.0 | 0.5 | 0.0308 | +0.1% [-25.7%, +20.6%] | 0.0% | no | no | no |
| 1.0 | 0.25 | 0.0 | 0.0 | 0.0305 | +0.9% [-23.4%, +22.9%] | 0.0% | no | no | no |
| 1.0 | 0.25 | 0.0 | 0.1 | 0.0305 | +0.9% [-23.4%, +22.9%] | 0.0% | no | no | no |
| 1.0 | 0.25 | 0.0 | 0.3 | 0.0296 | +3.9% [-21.7%, +26.7%] | 0.0% | no | no | no |
| 1.0 | 0.25 | 0.25 | 0.5 | 0.0293 | +4.8% [-14.4%, +16.5%] | 0.0% | no | no | no |
| 1.0 | 0.25 | 0.0 | 0.2 | 0.0287 | +6.9% [-18.6%, +31.2%] | 0.0% | no | no | no |
| 1.0 | 0.5 | 0.0 | 0.5 | 0.0286 | +7.0% [-35.0%, +35.0%] | 0.5% | no | no | no |
| 1.0 | 0.5 | 0.0 | 0.0 | 0.0282 | +8.3% [-32.8%, +36.0%] | 0.5% | no | no | no |
| 1.0 | 0.5 | 0.0 | 0.1 | 0.0282 | +8.3% [-32.8%, +36.0%] | 0.5% | no | no | no |
| 1.0 | 0.5 | 0.0 | 0.2 | 0.0282 | +8.5% [-32.4%, +36.1%] | 0.5% | no | no | no |

**Selected: none.** No facet-active configuration met both conditions; report this, with the accuracy or explainability cost that ruled each candidate out.

*Template shares count the dominant feature of each Top-10 item, i.e. the explanation the system would show; grounding of those explanations is checked by the faithfulness audit.*
