"""Common utility functions for HDC operations."""

import argparse
from typing import Any

def comma_separated_integers(value: str) -> list[int]:
    """Convert a comma-separated string into a list of integers."""
    try:
        numbers = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not numbers:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return numbers

def chunk_sequence(sequence: list[Any], chunk_size: int) -> list[list[Any]]:
    """Divide a sequence into evenly sized chunks."""
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]
