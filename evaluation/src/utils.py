"""Shared evaluation helpers."""
import json
from pathlib import Path


def load_locomo(path: Path) -> list:
    """Load LOCOMO dataset (list of {conversation, qa})."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
