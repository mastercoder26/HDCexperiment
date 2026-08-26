"""Query engine for advanced HDC inference."""

import numpy as np
from .model import HDCClassifier, Record

class QueryEngine:
    """Provides advanced query capabilities over an HDC model."""
    
    def __init__(self, classifier: HDCClassifier):
        self.classifier = classifier
        
    def top_k(self, record: Record, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k most similar labels for a record."""
        encoded = self.classifier.encode(record)
        scores = {
            label: self.classifier.hdc.similarity(encoded, prototype)
            for label, prototype in self.classifier.prototypes.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]
        
    def threshold_query(self, record: Record, threshold: float = 0.5) -> list[str]:
        """Return all labels with similarity above the given threshold."""
        encoded = self.classifier.encode(record)
        results = []
        for label, prototype in self.classifier.prototypes.items():
            if self.classifier.hdc.similarity(encoded, prototype) >= threshold:
                results.append(label)
        return results
