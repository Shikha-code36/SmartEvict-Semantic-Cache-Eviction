"""Download + prep a real workload from WildChat-1M (run locally; needs
Hugging Face access). Same record schema as download_lmsys.py, so the
existing benchmark/simulator code consumes it unchanged.

Produces:
    [{"t": float, "text": str, "response_tokens": int}, ...]

Usage:
    pip install datasets python-dotenv
    # put HF_TOKEN=hf_xxx in a .env file in the repo root (gitignored) if the
    # dataset ever requires accepting terms on the HF hub
    python -m smartevict.data.download_wildchat --n 50000 --out data/wildchat_trace.json

WildChat-1M is real logged ChatGPT usage (not Chatbot Arena comparisons like
lmsys-chat-1m), so it's a different traffic source for the same "real
chatbot conversation" domain -- useful for checking whether findings on
lmsys-chat-1m generalize or are trace-specific.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--dataset", default="allenai/WildChat-1M")
    ap.add_argument("--out", default="data/wildchat_trace.json")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # See download_lmsys.py: quiets a harmless retry warning from streaming
    # reads that abandon an in-flight prefetched shard when we stop early.
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train", streaming=True,
                      token=os.environ.get("HF_TOKEN"))
    ds = ds.take(args.n * 2)  # generous cap: some rows get skipped below

    records, t = [], 0.0
    for row in ds:
        conv = row.get("conversation") or []
        user = next((m["content"] for m in conv if m["role"] == "user"), None)
        asst = next((m["content"] for m in conv if m["role"] == "assistant"), None)
        if not user or not asst:
            continue
        t += 1.0
        records.append({
            "t": t,
            "text": user[:2000],
            # ~4 chars/token heuristic; swap in a real tokenizer if you want
            "response_tokens": max(1, len(asst) // 4),
        })
        if len(records) >= args.n:
            break

    with open(args.out, "w") as f:
        json.dump(records, f)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
