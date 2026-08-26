"""Training loop abstractions for HDC."""

from typing import Sequence
from .model import HDCClassifier, LabeledRecord

class TrainingLoop:
    """A standard training loop for HDC classifiers."""
    
    def __init__(self, classifier: HDCClassifier):
        self.classifier = classifier
        
    def run(self, records: Sequence[LabeledRecord], epochs: int = 1) -> None:
        """Run the training loop for the specified number of epochs."""
        for epoch in range(epochs):
            # HDC is typically one-shot, but iterative retraining can refine prototypes
            self.classifier.train(records)
            # In a real scenario, you'd adjust prototypes by subtracting errors here
