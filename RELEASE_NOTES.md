# Release v2.0.0

## Summary

Version 2 is a research update to the accompanying manuscript, strengthening its scholarly positioning and statistical grounding. The empirical findings, methodology, and conclusions from v1 are unchanged; this release adds the context and rigor a peer-review process would expect around them.

## Highlights

- Added a comprehensive Related Work section, positioning SmartEvict against classical cache eviction (LRU-K, 2Q, ARC, LIRS, GDSF), learned cache eviction (Cold-RL, Learning Relaxed Belady), semantic response caches (GPTCache), and KV-cache eviction in LLM serving (StreamingLLM, H2O, Scissorhands, vLLM, SGLang).
- Expanded the Discussion with a dedicated caveat on embedding-model choice, surfacing a limitation that previously lived only in Threats to Validity.
- Added a compact feature/architecture ablation table (Table 4) directly in the manuscript, alongside the existing full ablation write-up in `results/ABLATIONS.md`.
- Added a statistical significance check (paired t-test) supporting the "statistically tied" reading of the WildChat-1M three-way comparison.
- Substantially expanded the bibliography to properly cite every system and prior work discussed in Related Work.
- Minor wording refinements for precision (e.g., the abstract's characterization of the reuse-density relationship now explicitly notes it is drawn from three real-world traces).

## Notes

- No changes to the simulator, policies, training pipeline, or reported experimental results — this release is limited to the manuscript's framing, citations, and statistical support.

---

# Release v1.0.0

## Summary

SmartEvict v1.0.0 introduces the first published release of the repository alongside a GitHub release. This release highlights the project's empirical comparison between learned and heuristic eviction policies for semantic LLM caches.

## Highlights

- Learned vs heuristic comparison: benchmarked learned eviction against LRU, FIFO, and cost-aware heuristics.
- Benchmark suite: reproducible experiments for synthetic and real workloads, including LMSYS-Chat-1M, WildChat-1M, and Bitext.
- Reproducibility: `reproduce.sh` regenerates every benchmark table and figure from the repository.
- Accompanying paper: a full manuscript with results, methods, and discussion.

## Notes

- This release marks the repository's first stable public snapshot as `v1.0.0`.
- DOI, arXiv, and GitHub Actions badges will be added once the respective artifacts exist.
