"""Append-only audit log enabling deterministic replay.

Every execution appends one record capturing everything needed to
reconstruct the outcome later *without calling the backend*:
(input state, model version, output value, probabilities, confidence,
threshold routing decision, idempotency key).

POC note on PII: records store the full input state. A production
platform would store a state hash plus a reference to the case record in
its system of record. The hash is already recorded here alongside the
state to make that migration trivial.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._records: list[dict] = []
        if self._path and self._path.exists():
            for line in self._path.read_text().splitlines():
                if line.strip():
                    self._records.append(json.loads(line))

    def append(self, record: dict) -> int:
        record = {"audit_id": len(self._records), "ts": time.time(), **record}
        self._records.append(record)
        if self._path:
            with self._path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        return record["audit_id"]

    def get(self, audit_id: int) -> dict:
        if not 0 <= audit_id < len(self._records):
            raise KeyError(f"no audit record {audit_id}")
        return self._records[audit_id]

    def records(self) -> list[dict]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
