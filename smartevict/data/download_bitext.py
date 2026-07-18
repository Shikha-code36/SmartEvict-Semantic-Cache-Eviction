"""Download + prep a real workload from the Bitext customer-support dataset
(run locally; needs Hugging Face access, no gating/license acceptance).
Same record schema as download_lmsys.py / download_wildchat.py, so the
existing benchmark/simulator code consumes it unchanged.

Produces:
    [{"t": float, "text": str, "response_tokens": int}, ...]

Usage:
    pip install datasets python-dotenv
    python -m smartevict.data.download_bitext --out data/bitext_trace.json

Unlike LMSYS/WildChat (multi-turn chat logs with natural arrival order),
this dataset is a flat instruction/response corpus organized by intent
(~1000 paraphrased variants per intent, 27 intents), and the raw dataset
order is one long contiguous block per intent (confirmed: only 2 intent
switches in the first 2000 rows) -- NOT organic interleaved traffic. Fed in
as-is, that ordering makes eviction-policy choice nearly irrelevant (every
cache window is dominated by one intent at a time, so plain LRU already
handles it), which produced a flat, uninformative benchmark result. To
approximate real concurrent multi-user traffic where different intents
interleave, rows are shuffled (fixed seed, so the trace is reproducible)
before `t` is assigned as row position in the shuffled order. This is
deliberately the "dense, repetitive customer-support traffic" counterpart
to LMSYS/WildChat's sparser general-chat traffic -- see results/RESULTS.md
for why reuse density is the variable that matters for this project's
central finding.
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    ap.add_argument("--out", default="data/bitext_trace.json")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed, for reproducible interleaving of intents")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train", streaming=True,
                      token=os.environ.get("HF_TOKEN"))

    rows = []
    for row in ds:
        user = row.get("instruction")
        asst = row.get("response")
        if not user or not asst:
            continue
        rows.append((user, asst))

    random.Random(args.seed).shuffle(rows)

    records = []
    for t, (user, asst) in enumerate(rows):
        records.append({
            "t": float(t),
            "text": user[:2000],
            # ~4 chars/token heuristic; swap in a real tokenizer if you want
            "response_tokens": max(1, len(asst) // 4),
        })

    with open(args.out, "w") as f:
        json.dump(records, f)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
