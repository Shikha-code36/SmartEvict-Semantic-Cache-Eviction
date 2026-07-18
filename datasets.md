# Datasets

Four workloads are used in the accompanying paper (published separately
on arXiv — see the README for the link once available) and in this
repo's results write-ups (`results/RESULTS.md`, `results/ABLATIONS.md`):
one generated locally, three downloaded from Hugging Face. All three
downloaders write the same record schema, so the rest of the pipeline
(`smartevict-benchmark`, ablation scripts, `paper/figures/make_figures.py`)
consumes any of them unchanged:

```json
[{"t": <float, arrival order/time>, "text": <str, user prompt>, "response_tokens": <int, regeneration-cost proxy>}, ...]
```

`response_tokens` is estimated at ~4 characters/token (a heuristic, not a
real tokenizer call — see each script's docstring) unless noted otherwise.

## Synthetic (no download)

Generated on the fly by `smartevict.data.generate_synthetic`
(Zipf-distributed intents + paraphrase noise + a unique long tail), swept
across three duplicate-density regimes (high/med/low). No dataset file is
written to disk; `smartevict-benchmark --n 20000 --cache-size 400` runs
this generator directly. Used in Results §4.1 (three synthetic rows) and
throughout the ablations (§4.5).

## LMSYS-Chat-1M

- **Source:** [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) — 1M real conversation logs from the Chatbot Arena / Vicuna demo.
- **Access:** gated. Requires a free Hugging Face account, a manually-approved access request on the dataset page (not instant — minutes to a day+), and a read-scoped token.
- **Setup:**
  ```bash
  pip install -e ".[lmsys]"
  # then, with HF_TOKEN=hf_... in a .env file at the repo root (gitignored):
  smartevict-download-lmsys --n 50000 --out data/lmsys_trace.json
  ```
- **Used as:** 50,000 requests, time-ordered 60/40 train/test split (30K
  train, 20K held-out). This is the sparsest-reuse real trace evaluated
  (hit ratio 0.10–0.14, untouched-candidate fraction 17.7%) and the one
  where GDSF's advantage over the learned policy is largest (§4.2, §4.3).
  Benchmarked under both `HashingEmbedder` and real MiniLM sentence
  embeddings (`--embedder minilm`, requires `pip install -e ".[minilm]"`,
  first run downloads `all-MiniLM-L6-v2`, ~90MB) as a robustness check
  (§5.3).

## WildChat-1M

- **Source:** [`allenai/WildChat-1M`](https://huggingface.co/datasets/allenai/WildChat-1M) — real logged ChatGPT usage, a different traffic source from LMSYS's Chatbot Arena comparisons, used to check whether findings generalize across real-chat sources.
- **Access:** not gated (as of this writing); still requires a Hugging Face account/token via the same `.env` mechanism if the dataset ever adds gating.
- **Setup:**
  ```bash
  pip install -e ".[lmsys]"   # same `datasets`/`python-dotenv` extra covers this too
  smartevict-download-wildchat --n 50000 --out data/wildchat_trace.json
  ```
- **Used as:** 50,000 requests, evaluated at three cache sizes (400/100/50)
  for the cache-pressure sweep (§4.4). This is the *densest*-reuse real
  trace evaluated (untouched-candidate fraction 11.3%, hit ratio
  ≈0.22–0.23) — the workload where all cost-aware policies are within
  noise of each other (§4.2).

## Bitext customer support

- **Source:** [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — a flat instruction/response corpus organized by intent (~1000 paraphrased variants per intent, 27 intents), not a natural chat log.
- **Access:** not gated.
- **Important preprocessing note:** the raw dataset order is one long
  contiguous block per intent (only 2 intent switches observed in the
  first 2000 rows), which is not organic interleaved traffic — fed in
  as-is, every cache window is dominated by a single intent and eviction
  policy choice becomes nearly irrelevant. `download_bitext.py` shuffles
  rows with a fixed seed (default 0, reproducible) before assigning
  arrival order, to approximate concurrent multi-user traffic where
  different intents interleave.
- **Setup:**
  ```bash
  pip install -e ".[lmsys]"
  smartevict-download-bitext --out data/bitext_trace.json
  ```
- **Used as:** 26,872 requests, cache size 400. Reuse density
  (untouched-candidate fraction 15.1%) and GDSF's advantage over the
  learned policy (2.8pp) both sit between WildChat-1M and LMSYS-Chat-1M
  (§4.3) — the intermediate point on the reuse-density trend line
  (Figure 3).

## Regenerating Section 4's tables from these datasets

See `reproduce.sh` at the repo root for the exact commands that turn
these four workloads into every table in `paper/PAPER_DRAFT.md` Section 4
and every number in `results/RESULTS.md` / `results/ABLATIONS.md`.
