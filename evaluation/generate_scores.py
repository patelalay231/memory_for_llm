"""Aggregate metrics by category and overall (BLEU, F1, LLM, latency)."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import EVAL_METRICS_FILE

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def main():
    path = Path(EVAL_METRICS_FILE)
    if not path.exists():
        print(f"Run evals first. Expected: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten
    all_items = []
    for key in data:
        all_items.extend(data[key])

    if HAS_PANDAS:
        df = pd.DataFrame(all_items)
        df["category"] = pd.to_numeric(df["category"], errors="coerce")
        result = df.groupby("category").agg({
            "bleu_score": "mean",
            "f1_score": "mean",
            "llm_score": "mean",
            "speaker_1_memory_time": "mean",
            "speaker_2_memory_time": "mean",
            "response_time": "mean",
        }).round(4)
        result["count"] = df.groupby("category").size()
        print("Mean scores per category:")
        print(result)
        overall = df.agg({
            "bleu_score": "mean",
            "f1_score": "mean",
            "llm_score": "mean",
            "speaker_1_memory_time": "mean",
            "speaker_2_memory_time": "mean",
            "response_time": "mean",
        }).round(4)
        print("\nOverall means:")
        print(overall)
    else:
        # Fallback without pandas
        by_cat = {}
        for r in all_items:
            c = r.get("category")
            if c is None:
                continue
            c = str(c)
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(r)
        print("Mean per category:")
        for c in sorted(by_cat.keys()):
            rows = by_cat[c]
            n = len(rows)
            b = sum(x.get("bleu_score", 0) or 0 for x in rows) / n
            f = sum(x.get("f1_score", 0) or 0 for x in rows) / n
            l = sum(x.get("llm_score", 0) or 0 for x in rows) / n
            t1 = sum(x.get("speaker_1_memory_time", 0) or 0 for x in rows) / n
            t2 = sum(x.get("speaker_2_memory_time", 0) or 0 for x in rows) / n
            rt = sum(x.get("response_time", 0) or 0 for x in rows) / n
            print(f"  category {c}: bleu={b:.4f} f1={f:.4f} llm={l:.4f} s1_time={t1:.4f}s s2_time={t2:.4f}s resp_time={rt:.4f}s count={n}")
        n = len(all_items)
        if n:
            print("\nOverall:")
            print(f"  bleu={sum(x.get('bleu_score',0) or 0 for x in all_items)/n:.4f} f1={sum(x.get('f1_score',0) or 0 for x in all_items)/n:.4f} llm={sum(x.get('llm_score',0) or 0 for x in all_items)/n:.4f}")


if __name__ == "__main__":
    main()
