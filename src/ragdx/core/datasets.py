from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

from ragdx.schemas.models import DatasetRecord


def load_jsonl(path: str | Path) -> List[DatasetRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(DatasetRecord(**json.loads(line)))
    return records


def load_json(path: str | Path) -> List[DatasetRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", [])
    return [DatasetRecord(**row) for row in payload]


def load_csv(path: str | Path) -> List[DatasetRecord]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            contexts = row.get("contexts", "")
            ref_contexts = row.get("reference_contexts", "")
            rows.append(
                DatasetRecord(
                    question=row["question"],
                    ground_truth=row.get("ground_truth") or None,
                    answer=row.get("answer") or None,
                    contexts=[x for x in contexts.split("||") if x],
                    reference_contexts=[x for x in ref_contexts.split("||") if x],
                    metadata={k: v for k, v in row.items() if k not in {"question", "ground_truth", "answer", "contexts", "reference_contexts"}},
                )
            )
    return rows


def load_records(path: str | Path) -> List[DatasetRecord]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        return load_json(path)
    if suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported dataset file type: {suffix}")


def save_records_jsonl(records: Iterable[DatasetRecord], path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")
    return path
