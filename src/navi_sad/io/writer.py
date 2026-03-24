"""Gzipped JSONL writer for raw inference records."""

import gzip
import json
from dataclasses import asdict
from pathlib import Path

from navi_sad.core.types import RawSampleRecord


class RawRecordWriter:
    """Append-only gzipped JSONL writer.

    Usage:
        with RawRecordWriter(path) as w:
            w.write(record)
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._file = gzip.open(self._path, "wt", encoding="utf-8")

    def write(self, record: RawSampleRecord) -> None:
        d = asdict(record)
        self._file.write(json.dumps(d) + "\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RawRecordWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
