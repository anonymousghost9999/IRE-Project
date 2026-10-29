# PaperFeed

PaperFeed is an AI-powered scientific paper recommendation and feed generation engine that extracts typed scientific facets (Task, Method, Metric, Dataset, Topic) from abstracts and computes grounded relevance and novelty.

## SciERC Benchmark Evaluation (Phase 3.4)

The facet extraction pipeline evaluates against the official SciERC Gold Standard (Luan et al., 2018).

### 1. SciERC Data Setup
SciERC dataset files are git-ignored and expected under `data/raw/scierc/`.

Download the processed test split:
- **Official source (UW):** [http://nlp.cs.washington.edu/sciIE/](http://nlp.cs.washington.edu/sciIE/) (`sciERC_processed.tar.gz`)
- **Hugging Face mirror:** [https://huggingface.co/datasets/sthoran/scierc_processed_data](https://huggingface.co/datasets/sthoran/scierc_processed_data)

To download directly:
```bash
mkdir -p data/raw/scierc
curl -L -o data/raw/scierc/test.json https://huggingface.co/datasets/sthoran/scierc_processed_data/resolve/main/test.json
```

### 2. Running Evaluation
```bash
# Standard evaluation (threshold 0.5)
python scripts/evaluate_scierc.py

# Threshold sweep across confidence levels
python scripts/evaluate_scierc.py --sweep 0.3,0.4,0.5,0.6,0.7

# Evaluate with SciERC native prompt labels directly
python scripts/evaluate_scierc.py --native-labels
```

### 3. Benchmark Reports
Outputs are written to:
- `results/scierc_benchmark_report.json`
- `results/scierc_benchmark_report.md`
