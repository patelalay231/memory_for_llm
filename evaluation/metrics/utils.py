"""BLEU and F1 metrics (aligned with mem0 evaluation)."""
from typing import Dict

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

try:
    nltk.download("punkt", quiet=True)
except Exception:
    pass


def simple_tokenize(text: str) -> list:
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f"bleu{n}"] = score
    return scores


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    if not prediction or not reference:
        return {"exact_match": 0, "f1": 0.0, "bleu1": 0.0}
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    exact_match = int(prediction.lower() == reference.lower())
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(ref_tokens)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    bleu = calculate_bleu_scores(prediction, reference)
    return {"exact_match": exact_match, "f1": f1, **bleu}
