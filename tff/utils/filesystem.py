#!/usr/bin/env python3
"""
Filesystem I/O utilities for reading and writing JSON, JSONL, and text files.
"""

import json
import os
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel

# Extended JSONValue type that includes all types SafeJSONEncoder can handle
JSONValue = (
    Mapping[str, "JSONValue"]
    | list["JSONValue"]
    | str
    | int
    | float
    | bool
    | None
    | Path  # Serialized as string
    | datetime  # Serialized as ISO format string
    | date  # Serialized as ISO format string
    | time  # Serialized as ISO format string
    | timedelta  # Serialized as dict with days/seconds/microseconds
    | Decimal  # Serialized as float
    | UUID  # Serialized as string
    | set["JSONValue"]  # Serialized as list
    | frozenset[Any]  # Serialized as list
    | bytes  # Serialized as UTF-8 string
    | bytearray  # Serialized as UTF-8 string
    | Enum  # Serialized as .value
    | complex  # Serialized as dict with real/imag
    | ZoneInfo  # Serialized as timezone key string
)


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles common non-serializable Python types.

    Supported types:
    - pathlib.Path (PosixPath, WindowsPath, etc.)
    - datetime, date, time, timedelta
    - Decimal
    - UUID
    - set, frozenset
    - bytes, bytearray
    - Enum
    - complex numbers
    - ZoneInfo (timezone objects)
    - Custom objects with __dict__ attribute
    """

    def default(self, obj):  # pyright: ignore[reportImplicitOverride, reportImplicitOverride, reportIncompatibleMethodOverride, reportMissingParameterType]
        # Path objects (PosixPath, WindowsPath, etc.)
        if isinstance(obj, Path):
            return str(obj)

        # Datetime objects
        elif isinstance(obj, (datetime, date, time)):
            return obj.isoformat()

        # Timedelta
        elif isinstance(obj, timedelta):
            return {
                "__type__": "timedelta",
                "days": obj.days,
                "seconds": obj.seconds,
                "microseconds": obj.microseconds,
            }

        # Decimal
        elif isinstance(obj, Decimal):
            return float(obj)

        # UUID
        elif isinstance(obj, UUID):
            return str(obj)

        # Set and frozenset
        elif isinstance(obj, (set, frozenset)):
            return list(obj)

        # Bytes and bytearray
        elif isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8", errors="replace")

        # Enum
        elif isinstance(obj, Enum):
            return obj.value

        # Complex numbers
        elif isinstance(obj, complex):
            return {"__type__": "complex", "real": obj.real, "imag": obj.imag}

        # ZoneInfo (timezone objects)
        elif isinstance(obj, ZoneInfo):
            return str(obj.key)

        # Custom objects with __dict__
        elif hasattr(obj, "__dict__"):
            return obj.__dict__

        # Let the base class raise TypeError for truly unsupported types
        return super().default(obj)


def safe_json_dumps(data: Any, **kwargs):  # pyright: ignore[reportMissingParameterType]
    """
    Some useful kwargs: `indent=4, ensure_ascii=False`
    """
    return json.dumps(data, cls=SafeJSONEncoder, **kwargs)


def read_json_data(filepath: Path, obj_class: type[BaseModel] | None = None) -> Any:
    """Extracts a dictionary from a local JSON file path."""
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as file:
            data = json.load(file)
            return obj_class.model_validate(data) if obj_class is not None else data
    else:
        print(f"File: '{filepath}' doesn't seem to exist, returning empty dictionary")
        return {}


def write_json_data(filepath: Path, data: JSONValue):
    """Writes out a dictionary to a JSON file with human-readable indents."""
    with open(filepath, mode="w", encoding="utf-8") as f:
        file_content = safe_json_dumps(
            data,
            indent=4,
            ensure_ascii=False,
        )
        _ = f.write(file_content)


def write_txt_data(filepath: Path, data: str):
    """Dumps a string to a text file."""
    with open(filepath, mode="w", encoding="utf-8") as f:
        _ = f.write(data)


def read_txt_data(filepath: Path) -> list[str]:
    """Reads in text from a filepath as an array of strings."""
    with open(filepath) as f:
        return f.readlines()


def read_jsonl_data(filepath: Path, obj_class: type[BaseModel] | None = None) -> Any:
    """Reads data from a JSONL file into a list of dictionaries."""
    if os.path.exists(filepath):
        data = []
        with open(filepath, encoding="utf-8") as file:
            for line in file:
                data.append(
                    obj_class.model_validate(json.loads(line))
                    if obj_class is not None
                    else json.loads(line)
                )
        return data
    else:
        print(f"File: '{filepath}' doesn't seem to exist")
        return []


def write_jsonl_data(filepath: Path, data: list[JSONValue], append: bool = False):
    """Writes a list of dictionaries to a JSONL file."""
    mode = "a" if append else "w"
    with open(filepath, mode=mode, encoding="utf-8") as f:
        for i, item in enumerate(data):
            try:
                _ = f.write(
                    json.dumps(item, separators=(",", ":"), cls=SafeJSONEncoder)
                )
                if i != len(data) - 1 or append:
                    _ = f.write("\n")
            except Exception as e:
                print(f"Error writing item: {item}")
                print(f"Exception: {e}")
