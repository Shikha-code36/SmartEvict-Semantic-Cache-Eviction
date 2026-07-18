# Ablation studies

Follow-up studies to [RESULTS.md](RESULTS.md), digging into *why* the
numbers there look the way they do — specifically, why GDSF (a simple,
non-learned heuristic) matches or beats the learned RL policy in most
regimes tested. Read RESULTS.md first; this document assumes its context
(K-tail candidate sampling, the 6-feature set, the discounted-demand
training target) and its honest-reporting conventions.

**Where each ablation runs:** feature/architecture ablations run on *both*
the synthetic med-dup regime and the real LMSYS-Chat-1M trace (the LMSYS
trace was already downloaded locally for RESULTS.md, so re-running it here
was free). The hyperparameter sweep also runs on both. Where the two
regimes tell different stories, that's flagged explicitly — don't average
them together.

All numbers below are 3-seed means (only the learned net's training seed
varies; LRU/dataset construction are deterministic per config), which is
fewer seeds than RESULTS.md's 5-seed headline numbers — these are
ablations meant to show *direction and relative size* of effects, not to
be the final citable numbers.

Reproduce:
```bash
python -m smartevict.benchmark.run_feature_ablation [--trace data/lmsys_trace.json]
python -m smartevict.benchmark.run_architecture_ablation [--trace data/lmsys_trace.json]
python -m smartevict.benchmark.run_hparam_sweep [--trace data/lmsys_trace.json]
```

## 1. Feature ablation

Which of the 6 features (`age, cost, hits, idle, mean_gap, staleness` —
see `smartevict/features/extract.py`) actually drive the learned policy's
performance. Trained/evaluated with `LearnedPolicy(net, feature_indices=...)`
on the corresponding column subset.

**Synthetic (med-dup):**

| Variant | vs LRU |
|---|---|
| full (all 6) | +6.03% ± 0.40% |
| drop-age | +5.53% ± 0.53% |
| drop-cost | +4.11% ± 0.21% |
| drop-hits | **+7.11% ± 0.81%** (↑ above full) |
| drop-idle | +5.99% ± 0.12% |
| drop-mean_gap | +5.76% ± 0.19% |
| drop-staleness | +5.72% ± 0.40% |
| cost-only | +0.43% ± 0.00% |
| recency-only (age+idle) | +0.53% ± 0.42% |
| no-cost (5 feats) | +4.11% ± 0.21% |

**LMSYS real trace:**

| Variant | vs LRU |
|---|---|
| full (all 6) | +16.78% ± 1.40% |
| drop-age | +14.84% ± 1.22% |
| **drop-cost** | **+2.86% ± 0.95%** (collapse) |
| drop-hits | +15.28% ± 1.16% |
| drop-idle | +15.35% ± 0.98% |
| drop-mean_gap | +16.13% ± 0.38% |
| drop-staleness | +16.86% ± 1.48% (≈ full — this feature is noise) |
| cost-only | +1.03% ± 2.79% |
| recency-only (age+idle) | +10.05% ± 5.39% |
| no-cost (5 feats) | +2.86% ± 0.95% |

**Reading this honestly — the headline number is misleading without
context.** Dropping `cost` on real data collapses performance (+16.78% →
+2.86%), which looks like strong evidence the net is doing sophisticated
cost-aware ranking. **It mostly isn't, for a structural reason**: the
training target itself is `y = Σ γ^Δt · response_tokens`, linearly scaled
by cost. A model with no cost feature can't calibrate the *magnitude* of
its predictions at all, and MSE punishes magnitude error heavily regardless
of whether the *ranking* (which candidate to evict) was already correct.
So this ablation mainly shows "the net needs cost to predict the right
number," not "the net learned to use cost the way GDSF's cost term does."

**The feature that actually explains the RESULTS.md gap to GDSF is the
opposite one: `drop-hits` barely hurts the net** (+16.78% → +15.28% on
real data; on synthetic it *improves* the net, +6.03% → +7.11%). GDSF's
entire real-data advantage comes from hard-gating on `hit_count == 0` (see
RESULTS.md). The net has that exact signal available and evidently isn't
leaning on it — removing it costs almost nothing. That's the real
explanation for the GDSF gap: not a missing feature, but that per-candidate
MSE regression against a cost-scaled target doesn't push the net to
discover the sharp hit-count threshold GDSF exploits by construction.

`staleness` (idle/age ratio) is close to pure noise in both regimes —
dropping it doesn't hurt, and on real data it *very slightly* helps. A
future revision could drop it from the shipped feature set with little
expected cost.

## 2. Architecture ablation

`DuelingEvictionNet` (current, 9,474 params: trunk + value/advantage split)
vs. `LinearEvictionNet` (new, 7 params: `Q(x) = w·x + b`, same 6 features,
same MSE+Adam training loop) — see `smartevict/model/linear_net.py`.
Isolates "does the dueling/nonlinear structure earn its complexity" from
"does having any learned model beat heuristics" (answered separately, and
negatively for real data, in RESULTS.md).

| Regime | Dueling (9,474 params) | Linear (7 params) |
|---|---|---|
| Synthetic (med-dup) | +6.03% ± 0.40% | +4.48% ± 1.81% |
| LMSYS real trace | +16.78% ± 1.40% | +6.81% ± 0.07% |

On synthetic data the 7-parameter linear formula captures most of the
dueling net's value — the architecture gap is real but modest. **On real
data the dueling net delivers roughly 2.5x the linear model's result**
(+16.78% vs +6.81%), so the nonlinear structure is earning its complexity
specifically on real-world traffic, plausibly because the true
feature-to-value relationship has interactions (e.g. hit-count × cost,
which is nonlinear) a plain weighted sum can't represent. Notably the
linear model is also far more *stable* (std 0.07 vs 1.40 on LMSYS) —
less capacity gives less seed-to-seed variance, a real trade-off for a
production deployment that values predictability. Neither architecture
comes close to GDSF's +27.7% on the same trace (RESULTS.md) — more
capacity narrows the gap to a simple heuristic, but doesn't close it.

## 3. Hyperparameter sensitivity sweep

`k_tail`, `cache_size`, `similarity threshold` swept one axis at a time
against the shipped defaults (`k_tail=8, cache_size=400, threshold=0.8`),
not a full cross product (9 configs × 3 seeds instead of 27 × 3).

**Synthetic (med-dup):**

| Axis | Value | vs LRU |
|---|---|---|
| k_tail | 4 | +4.02% ± 0.45% |
| | **8 (default)** | +6.03% ± 0.40% |
| | 16 | **+9.61% ± 0.62%** |
| cache_size | 200 | +5.78% ± 0.73% |
| | **400 (default)** | +6.03% ± 0.40% |
| | 800 | +2.23% ± 0.07% |
| threshold | 0.75 | **+9.74% ± 0.80%** |
| | **0.80 (default)** | +6.03% ± 0.40% |
| | 0.85 | **−0.75% ± 0.12%** (worse than LRU) |

**LMSYS real trace:**

| Axis | Value | vs LRU |
|---|---|---|
| k_tail | 4 | +14.88% ± 5.18% |
| | **8 (default)** | +16.78% ± 1.40% |
| | 16 | **+18.74% ± 0.22%** |
| cache_size | 200 | +20.39% ± 9.81% (high variance) |
| | **400 (default)** | +16.78% ± 1.40% |
| | 800 | **+24.42% ± 0.47%** |
| threshold | 0.75 | **+37.93% ± 1.64%** |
| | **0.80 (default)** | +16.78% ± 1.40% |
| | 0.85 | +8.21% ± 0.99% |

**The shipped defaults are conservative, not tuned for best results.**
Larger `k_tail` (16) and a looser `threshold` (0.75) both help
substantially in both regimes — the defaults were picked once early in the
project and never swept until now. `cache_size` is regime-dependent (worse
at 800 on synthetic, better at 800 on real data) rather than a clean trend
either direction.

**One result that needs flagging as a real limitation, not just "smaller
gains": on synthetic data at `threshold=0.85`, the learned policy performs
*worse than plain LRU*** (−0.75% ± 0.12%). This doesn't happen on the real
trace (threshold=0.85 there still yields +8.21%), so it isn't universal,
but it's evidence the learned policy is not robust to threshold changes in
every regime — a stricter similarity match changes which entries even
become eviction candidates in ways the offline-trained net wasn't
necessarily prepared for. This is worth investigating further before
claiming robustness in a paper, and is exactly the kind of failure a
hyperparameter sweep is supposed to surface.

## Summary for the paper

- The learned model's advantage over recency-only LRU/FIFO is real and
  consistent (RESULTS.md). Its advantage over a simple cost-aware
  heuristic (GDSF) is not — GDSF wins or ties in most regimes tested.
- The learned model's real-data performance depends heavily on the `cost`
  feature (mechanically, via the target definition) but barely uses
  `hit_count` (the feature GDSF's advantage is built on) — this is the
  clearest evidence for *why* the RL approach underperforms GDSF: it isn't
  learning the right decision rule, not that it lacks the input to learn
  it.
- The dueling architecture's extra capacity over a linear model matters
  more on real data than synthetic data, but doesn't close the gap to
  GDSF in either regime.
- The shipped hyperparameters are conservative — larger K-tail and looser
  similarity thresholds both show headroom — and at least one
  configuration (synthetic, threshold=0.85) shows the learned policy
  underperforming plain LRU, a real robustness gap rather than just a
  smaller win.
