"""Architecture ablation: DuelingEvictionNet (~9.5K params) vs
LinearEvictionNet (7 params) on the same 6 features, same training data --
isolates "does the dueling/nonlinear structure earn its extra capacity"
from "does having *any* learned model beat non-learned heuristics" (that
question is answered separately in results/RESULTS.md against GDSF).

Usage:
  python -m smartevict.benchmark.run_architecture_ablation                    # synthetic med-dup
  python -m smartevict.benchmark.run_architecture_ablation --trace data/lmsys_trace.json
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
from smartevict.model.dueling_net import DuelingEvictionNet
from smartevict.model.linear_net import LinearEvictionNet


def run(records, args) -> dict:
    embed = HashingEmbedder(dim=args.dim).embed
    emb = embed([r["text"] for r in records])

    split = int(len(records) * args.train_frac)
    tr_rec, te_rec = records[:split], records[split:]
    tr_emb, te_emb = emb[:split], emb[split:]
    t0 = te_rec[0]["t"]
    te_rec = [{**r, "t": r["t"] - t0} for r in te_rec]

    print(f"building dataset ({split} train / {len(te_rec)} test requests)")
    X, y, groups, _ = build_dataset(tr_rec, tr_emb, args.cache_size, args.threshold,
                                    args.k_tail, gamma=args.gamma, horizon=args.horizon)
    print(f"{len(np.unique(groups))} eviction decisions, {len(X)} candidate samples")

    lru_res = run_simulation(te_rec, te_emb, args.cache_size, args.threshold, LRUPolicy())
    print(f"lru tokens_saved={lru_res.tokens_saved}")

    out = {
        "config": {"n_requests": len(records), "cache_size": args.cache_size,
                   "threshold": args.threshold, "k_tail": args.k_tail,
                   "seeds": args.seeds, "trace": args.trace},
        "lru_tokens_saved": lru_res.tokens_saved,
        "architectures": {},
    }
    for name, net_cls in [("dueling (current)", DuelingEvictionNet),
                          ("linear", LinearEvictionNet)]:
        seed_saved, n_params = [], None
        for seed in args.seeds:
            net, _ = train_model(X, y, groups, epochs=args.epochs, seed=seed,
                                 verbose=False, net_cls=net_cls)
            n_params = net.n_params
            res = run_simulation(te_rec, te_emb, args.cache_size, args.threshold,
                                 LearnedPolicy(net, k_tail=args.k_tail))
            seed_saved.append(res.tokens_saved)
        arr = np.array(seed_saved, dtype=np.float64)
        delta = (arr / lru_res.tokens_saved - 1) * 100
        out["architectures"][name] = {
            "n_params": n_params,
            "tokens_saved_mean": float(arr.mean()), "tokens_saved_std": float(arr.std()),
            "vs_lru_pct_mean": float(delta.mean()), "vs_lru_pct_std": float(delta.std()),
        }
        print(f"  {name:20s} ({n_params:>6,} params)  vs LRU: "
              f"{delta.mean():+6.2f}% +/- {delta.std():.2f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, default=None,
                    help="JSON trace file (e.g. data/lmsys_trace.json); default: synthetic med-dup")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--tail-frac", type=float, default=0.35, help="synthetic med-dup default")
    ap.add_argument("--cache-size", type=int, default=400)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--k-tail", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--horizon", type=float, default=5000.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=str, default="results/architecture_ablation.json")
    args = ap.parse_args()

    if args.trace:
        with open(args.trace) as f:
            records = json.load(f)
    else:
        records = generate_workload(n_requests=args.n, tail_frac=args.tail_frac, seed=7)

    out = run(records, args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
