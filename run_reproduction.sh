#!/bin/bash
# ==============================================================================
# Master Reproduction Script: Paper Feed SOTA Benchmark
# Reproduces the 100-user empirical benchmark on 3,901 real arXiv research papers.
# ==============================================================================

set -e

# Detect python binary (prefer .venv if available)
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    STREAMLIT=".venv/bin/streamlit"
else
    PYTHON="python3"
    STREAMLIT="streamlit"
fi

echo "=============================================================================="
echo "  PAPER FEED: SOTA REPRODUCIBILITY PIPELINE"
echo "  Python executable: $PYTHON"
echo "=============================================================================="

# Check if data exists; if not, prompt or download
if [ ! -f "data/corpus.jsonl" ]; then
    echo "[1/3] Corpus not found. Fetching 3,901 papers from arXiv API..."
    $PYTHON scripts/fetch_corpus.py
else
    echo "[1/3] Corpus found: data/corpus.jsonl (3,901 papers)."
fi

# Check if embeddings exist; if not, encode
if [ ! -f "data/embeddings.npy" ]; then
    echo "[2/3] Embeddings not found. Encoding papers on GPU/CPU..."
    $PYTHON scripts/embed_corpus.py
else
    echo "[2/3] Embeddings found: data/embeddings.npy."
fi

# Run the 100-user benchmark
echo "[3/3] Running head-to-head SOTA benchmark across 100 researchers..."
$PYTHON scripts/benchmark_sota_vs_sota.py

echo ""
echo "=============================================================================="
echo "  EXPERIMENT COMPLETE!"
echo "  Results saved to: data/sota_benchmark_results.json"
echo ""
echo "  To launch the interactive dashboard:"
echo "    $STREAMLIT run ui/app.py --server.port 8501"
echo "=============================================================================="
