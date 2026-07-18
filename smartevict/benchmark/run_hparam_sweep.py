"""Hyperparameter sensitivity sweep: k_tail, cache_size, similarity threshold.
One axis varied at a time against the shipped defaults (k_tail=8,
cache_size=400, threshold=0.8), not a full cross product -- keeps total runs
bounded (9 configs x 3 seeds instead of 27 configs x 3 seeds) while still
showing whether the defaults were a lucky pick or the results are robust to
them.

Each cell rebuilds the dataset from scratch (unlike the feature ablation):
k_tail/cache_size/threshold all change the underlying LRU replay and
candidate features, not just which columns of an already-built X get used.

Usage:
  python -m smartevict.benchmark.run_hparam_sweep                       # synthetic med-dup
  python -m smartevict.benchmark.run_hparam_sweep --trace data/lmsys_trace.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from smartevict.data.generate_synthetic import generate_workload
from smartevict.features.embeddings import HashingEmbedder
from smartevict.simulator.cache_sim import run_simulation
from smartevict.policies.eviction import LRUPolicy, LearnedPolicy
from smartevict.model.train import build_dataset, train_model


def eval_cell(records, emb, args, k_tail, cache_size, threshold, seeds):
    split = int(len(records) * args.train_frac)
    tr_rec, te_rec = records[:split], records[split:]
    tr_emb, te_emb = emb[:split], emb[split:]
    t0 = te_rec[0]["t"]
    te_rec = [{**r, "t": r["t"] - t0} for r in te_rec]

    lru_res = run_simulation(te_rec, te_emb, cache_size, threshold, LRUPolicy())
    X, y, groups, _ = build_dataset(tr_rec, tr_emb, cache_size, threshold, k_tail,
                                    gamma=args.gamma, horizon=args.horizon)

    seed_saved = []
    for seed in seeds:
        net, _ = train_model(X, y, groups, epochs=args.epochs, seed=seed, verbose=False)
        res = run_simulation(te_rec, te_emb, cache_size, threshold,
                             LearnedPolicy(net, k_tail=k_tail))
        seed_saved.append(res.tokens_saved)
    arr = np.array(seed_saved, dtype=np.float64)
    delta = (arr / lru_res.tokens_saved - 1) * 100
    return {
        "k_tail": k_tail, "cache_size": cache_size, "threshold": threshold,
        "lru_tokens_saved": lru_res.tokens_saved,
        "tokens_saved_mean": float(arr.mean()), "tokens_saved_std": float(arr.std()),
        "vs_lru_pct_mean": float(delta.mean()), "vs_lru_pct_std": float(delta.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, default=None,
                    help="JSON trace file (e.g. data/lmsys_trace.json); default: synthetic med-dup")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--tail-frac", type=float, default=0.35, help="synthetic med-dup default")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--horizon", type=float, default=5000.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k-tail-default", type=int, default=8)
    ap.add_argument("--cache-size-default", type=int, default=400)
    ap.add_argument("--threshold-default", type=float, default=0.8)
    ap.add_argument("--out", type=str, default="results/hparam_sweep.json")
    args = ap.parse_args()

    if args.trace:
        with open(args.trace) as f:
            records = json.load(f)
    else:
        records = generate_workload(n_requests=args.n, tail_frac=args.tail_frac, seed=7)

    embed = HashingEmbedder(dim=args.dim).embed
    emb = embed([r["text"] for r in records])

    kd, cd, td = args.k_tail_default, args.cache_size_default, args.threshold_default
    axes = {
        "k_tail": [(k, cd, td) for k in (4, 8, 16)],
        "cache_size": [(kd, c, td) for c in (200, 400, 800)],
        "threshold": [(kd, cd, t) for t in (0.75, 0.8, 0.85)],
    }

    out = {"config": {"defaults": {"k_tail": kd, "cache_size": cd, "threshold": td},
                      "seeds": args.seeds, "trace": args.trace},
          "axes": {}}
    for axis_name, cells in axes.items():
        print(f"[{axis_name}]")
        rows = []
        for k_tail, cache_size, threshold in cells:
            row = eval_cell(records, emb, args, k_tail, cache_size, threshold, args.seeds)
            rows.append(row)
            print(f"  k_tail={k_tail:<3} cache_size={cache_size:<4} threshold={threshold:<5} "
                  f"vs LRU: {row['vs_lru_pct_mean']:+6.2f}% +/- {row['vs_lru_pct_std']:.2f}%")
        out["axes"][axis_name] = rows

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
