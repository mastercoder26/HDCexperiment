"""Asynchronous wrappers for HDC operations."""

import asyncio
from typing import Sequence
from .model import Record, HDCClassifier

async def predict_batch_async(classifier: HDCClassifier, records: Sequence[Record]) -> list[dict[str, object]]:
    """Predict labels for a batch of records asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, classifier.predict_batch, records)
