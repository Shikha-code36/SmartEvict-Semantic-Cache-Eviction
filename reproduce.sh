#!/usr/bin/env bash
# Reproduces every table in paper/PAPER_DRAFT.md Section 4 (and
# results/RESULTS.md, results/ABLATIONS.md), plus the paper's figures.
#
# Idempotent: skips a step if its output file already exists, so re-running
# after an interruption (e.g. waiting on LMSYS-Chat-1M's manual HF access
# approval) only does the remaining work. Delete a results/data file to
# force that step to rerun.
#
# Usage: ./reproduce.sh
# Requires: Python 3.10+, and (for LMSYS-Chat-1M only) an approved
#           Hugging Face access request + HF_TOKEN in a .env file at the
#           repo root -- see datasets.md.
set -u  # not `set -e`: LMSYS's gated access may not be approved yet on a
        # fresh checkout, and that alone shouldn't abort the whole script.

cd "$(dirname "$0")"
FAILS=0

step() { echo; echo "=== $1 ==="; }
need_file() { [ -f "$1" ]; }
run_or_warn() {  # run_or_warn <description> <cmd...>
    desc=$1; shift
    if ! "$@"; then
        echo "!! $desc FAILED — continuing (see error above). Common cause:"
        echo "   LMSYS-Chat-1M access not yet approved on huggingface.co, or"
        echo "   HF_TOKEN missing from .env. See datasets.md."
        FAILS=$((FAILS + 1))
    fi
}

step "1/6  Install package + all extras (numpy, faiss, datasets, sentence-transformers, gptcache, matplotlib)"
pip install -e ".[all]" || { echo "pip install failed — aborting."; exit 1; }

step "2/6  Sanity test suite (~30s)"
python tests/test_all.py || { echo "Sanity tests failed — aborting before spending time on real benchmarks."; exit 1; }

step "3/6  Download datasets (skips already-downloaded files; see datasets.md for access requirements)"
mkdir -p data
need_file data/lmsys_trace.json    || run_or_warn "LMSYS-Chat-1M download"    smartevict-download-lmsys    --n 50000 --out data/lmsys_trace.json
need_file data/wildchat_trace.json || run_or_warn "WildChat-1M download"      smartevict-download-wildchat --n 50000 --out data/wildchat_trace.json
need_file data/bitext_trace.json   || run_or_warn "Bitext download"           smartevict-download-bitext   --out data/bitext_trace.json

step "4/6  Headline benchmarks (paper Section 4.1 table)"
mkdir -p results
need_file results/benchmark_synthetic.json \
    || smartevict-benchmark --n 20000 --cache-size 400 \
         --out results/benchmark_synthetic.json

if need_file data/lmsys_trace.json; then
    need_file results/benchmark_multiseed_hashing.json \
        || smartevict-benchmark --trace data/lmsys_trace.json --seeds 0 1 2 3 4 \
             --out results/benchmark_multiseed_hashing.json
    need_file results/benchmark_multiseed_minilm.json \
        || smartevict-benchmark --trace data/lmsys_trace.json --seeds 0 1 2 3 4 \
             --embedder minilm --out results/benchmark_multiseed_minilm.json
fi

if need_file data/wildchat_trace.json; then
    need_file results/benchmark_wildchat_c400.json \
        || smartevict-benchmark --trace data/wildchat_trace.json --n 50000 \
             --cache-size 400 --seeds 0 1 2 3 4 --out results/benchmark_wildchat_c400.json
    need_file results/benchmark_wildchat_c100.json \
        || smartevict-benchmark --trace data/wildchat_trace.json --n 50000 \
             --cache-size 100 --seeds 0 1 2 3 4 --out results/benchmark_wildchat_c100.json
    need_file results/benchmark_wildchat_c50.json \
        || smartevict-benchmark --trace data/wildchat_trace.json --n 50000 \
             --cache-size 50  --seeds 0 1 2 3 4 --out results/benchmark_wildchat_c50.json
fi

if need_file data/bitext_trace.json; then
    need_file results/benchmark_bitext_c400.json \
        || smartevict-benchmark --trace data/bitext_trace.json --cache-size 400 \
             --seeds 0 1 2 3 4 --out results/benchmark_bitext_c400.json
fi

step "5/6  Ablations (paper Section 4.5; results/ABLATIONS.md)"
need_file results/feature_ablation_synthetic.json \
    || python -m smartevict.benchmark.run_feature_ablation --out results/feature_ablation_synthetic.json
need_file results/architecture_ablation_synthetic.json \
    || python -m smartevict.benchmark.run_architecture_ablation --out results/architecture_ablation_synthetic.json
need_file results/hparam_sweep_synthetic.json \
    || python -m smartevict.benchmark.run_hparam_sweep --out results/hparam_sweep_synthetic.json
if need_file data/lmsys_trace.json; then
    need_file results/feature_ablation_lmsys.json \
        || python -m smartevict.benchmark.run_feature_ablation --trace data/lmsys_trace.json --out results/feature_ablation_lmsys.json
    need_file results/architecture_ablation_lmsys.json \
        || python -m smartevict.benchmark.run_architecture_ablation --trace data/lmsys_trace.json --out results/architecture_ablation_lmsys.json
    need_file results/hparam_sweep_lmsys.json \
        || python -m smartevict.benchmark.run_hparam_sweep --trace data/lmsys_trace.json --out results/hparam_sweep_lmsys.json
fi

step "6/6  Paper figures (local-only: paper/ is gitignored, not part of the public repo)"
if [ -f paper/figures/make_figures.py ]; then
    python paper/figures/make_figures.py
else
    echo "paper/ not present (expected on a fresh clone — the paper draft is"
    echo "published separately on arXiv, not tracked in this repo). Skipping."
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "Done. All steps completed — see results/ and paper/figures/."
else
    echo "Done, with $FAILS step(s) skipped/failed (see !! warnings above)."
    echo "Most likely cause: LMSYS-Chat-1M access not yet approved — rerun this"
    echo "script once access is granted; already-completed steps are skipped."
fi
