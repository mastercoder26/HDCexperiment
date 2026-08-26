"""Streaming data pipeline for large datasets."""

from typing import Iterable, Iterator
from .model import Record, LabeledRecord

def stream_records(file_path: str) -> Iterator[Record]:
    """Yield records one by one from a file (e.g. JSON Lines) to save memory."""
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def batch_stream(iterable: Iterable, batch_size: int = 100) -> Iterator[list]:
    """Group an iterable into batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
