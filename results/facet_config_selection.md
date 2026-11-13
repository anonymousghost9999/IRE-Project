# Facet-Active Variant Selection (tuning split)

Rule (design §9): among grid points with w_facet > 0 in order of tuning-split NDCG@10, the first whose relative NDCG@10 loss against the optimum has a 95% paired-bootstrap CI upper end <= 5% and whose Top-10 items have >= 10% facet or new-method explanations. 80 tuning authors, 10,000 resamples.

- Configuration: pool 100, channel depth 200, w_pop 0.0, adaptive_graph True
- Grid optimum (the headline system): w_cite=1.0, w_facet=0.25, w_nov=0.25, lambda_div=0.5, NDCG@10 0.0241, templates citation 29%, persona 71%

| w_cite | w_facet | w_nov | lambda_div | NDCG@10 | Rel. loss [95% CI] | Facet / new-method share | Non-inferior | Enough facets | Accepted |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1.0 | 0.25 | 0.25 | 0.5 | 0.0241 | +0.0% [+0.0%, +0.0%] | 0.0% | yes | no | no |
| 1.0 | 0.25 | 0.5 | 0.5 | 0.0240 | +0.6% [-8.0%, +12.7%] | 0.9% | no | no | no |
| 1.0 | 0.5 | 0.25 | 0.2 | 0.0238 | +1.3% [-26.4%, +20.1%] | 0.4% | no | no | no |
| 1.0 | 0.5 | 0.25 | 0.3 | 0.0238 | +1.3% [-26.4%, +20.1%] | 0.4% | no | no | no |
| 1.0 | 0.5 | 0.25 | 0.5 | 0.0238 | +1.3% [-26.4%, +20.1%] | 0.4% | no | no | no |
| 1.0 | 0.5 | 0.25 | 0.0 | 0.0234 | +2.8% [-25.7%, +23.6%] | 0.4% | no | no | no |
| 1.0 | 0.5 | 0.25 | 0.1 | 0.0234 | +2.8% [-25.7%, +23.6%] | 0.4% | no | no | no |
| 1.0 | 0.25 | 0.25 | 0.2 | 0.0232 | +3.8% [+0.0%, +13.3%] | 0.0% | no | no | no |
| 1.0 | 0.25 | 0.25 | 0.3 | 0.0232 | +3.8% [+0.0%, +13.3%] | 0.0% | no | no | no |
| 1.0 | 0.25 | 0.25 | 0.1 | 0.0231 | +4.3% [+0.0%, +13.4%] | 0.0% | no | no | no |

**Selected: none.** No facet-active configuration met both conditions; report this, with the accuracy or explainability cost that ruled each candidate out.

*Template shares count the dominant feature of each Top-10 item, i.e. the explanation the system would show; grounding of those explanations is checked by the faithfulness audit.*
