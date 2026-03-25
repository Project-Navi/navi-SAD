"""Gzipped JSONL readers for raw and derived records."""

import gzip
import json
from collections.abc import Iterator
from pathlib import Path


class RawRecordReader:
    """Iterate over raw inference records from a gzipped JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def __iter__(self) -> Iterator[dict]:
        with gzip.open(self._path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


class DerivedRecordReader:
    """Iterate over derived analysis records from a gzipped JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def __iter__(self) -> Iterator[dict]:
        with gzip.open(self._path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
