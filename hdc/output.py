"""Serialization and output formatting for HDC results."""

import json
import pickle
from pathlib import Path
from .model import HDCClassifier

def save_classifier(classifier: HDCClassifier, path: Path) -> None:
    """Serialize the classifier to a pickle file."""
    path.write_bytes(pickle.dumps(classifier))

def load_classifier(path: Path) -> HDCClassifier:
    """Load a classifier from a pickle file."""
    return pickle.loads(path.read_bytes())

def export_results_json(results: dict, path: Path) -> None:
    """Export evaluation results to JSON."""
    path.write_text(json.dumps(results, indent=2))
