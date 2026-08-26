"""Validation module for HDC records and data inputs."""

from typing import Any
from .model import Record

class ValidationError(Exception):
    """Exception raised for invalid records."""
    pass

def validate_record(record: Record, required_fields: set[str] = None) -> Record:
    """Validate a single categorical record."""
    if not isinstance(record, dict):
        raise ValidationError(f"Record must be a dictionary, got {type(record)}")
        
    if not record:
        raise ValidationError("Record cannot be empty")
        
    for key, value in record.items():
        if value is None:
            raise ValidationError(f"Field '{key}' cannot be null/None")
        if not isinstance(key, str):
            raise ValidationError(f"Keys must be strings, got {type(key)} for key {key}")
            
    if required_fields:
        missing = required_fields - set(record.keys())
        if missing:
            raise ValidationError(f"Record is missing required fields: {missing}")
            
    return record
