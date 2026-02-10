"""Compute BLEU, F1, LLM judge per question; save metrics + latency."""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import get_config, SEARCH_RESULTS_FILE, EVAL_METRICS_FILE
from evaluation.metrics.utils import calculate_metrics, calculate_bleu_scores
from evaluation.metrics.llm_judge import evaluate_llm_judge
from core.api.memory_api import MemoryAPI
from tqdm import tqdm


def process_item(item_data, llm_provider):
    k, v = item_data
    local = []
    for item in v:
        gt = str(item["answer"])
        pred = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])
        if category == "5":
            continue
        metrics = calculate_metrics(pred, gt)
        bleu = calculate_bleu_scores(pred, gt)
        llm_score = evaluate_llm_judge(question, gt, pred, llm_provider)
        rec = {
            "question": question,
            "answer": gt,
            "response": pred,
            "category": category,
            "bleu_score": bleu["bleu1"],
            "f1_score": metrics["f1"],
            "llm_score": llm_score,
            "speaker_1_memory_time": item.get("speaker_1_memory_time"),
            "speaker_2_memory_time": item.get("speaker_2_memory_time"),
            "response_time": item.get("response_time"),
        }
        local.append(rec)
    return k, local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=str(SEARCH_RESULTS_FILE))
    parser.add_argument("--output_file", type=str, default=str(EVAL_METRICS_FILE))
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = get_config()
    memory_api = MemoryAPI(config)
    llm_provider = memory_api.memory_store.llm_provider

    results = defaultdict(list)
    for item_data in tqdm(data.items(), desc="Eval"):
        k, local = process_item(item_data, llm_provider)
        results[k].extend(local)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
