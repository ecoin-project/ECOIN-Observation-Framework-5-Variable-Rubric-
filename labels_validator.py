#!/usr/bin/env python3
"""Validate ECOIN observation JSONL labels against observation_schema_v1.json.

Usage:
    python labels_validator.py \
        --input sample_labels_v1.jsonl \
        --schema observation_schema_v1.json

Exit codes:
    0 = all records valid
    1 = one or more records invalid
    2 = file/configuration error
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

EXPECTED_SCORE_KEYS = [
    "anxiety_intensity",
    "solution_seeking_pressure",
    "delegated_agency",
    "exploratory_reduction",
    "value_legibility_gap",
]

EXPECTED_CONFIDENCE = {"low", "medium", "high"}


@dataclass
class ValidationErrorItem:
    line_number: int
    path: str
    message: str

    def format(self) -> str:
        return f"Line {self.line_number} | {self.path}: {self.message}"


class SchemaLoadError(Exception):
    pass


class RecordValidator:
    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema
        self.required_top_level = self._extract_required_top_level(schema)

    @staticmethod
    def _extract_required_top_level(schema: dict[str, Any]) -> set[str]:
        required = schema.get("required")
        if not isinstance(required, list):
            raise SchemaLoadError("Schema is missing a top-level 'required' list.")
        return set(required)

    def validate_record(self, record: Any, line_number: int) -> list[ValidationErrorItem]:
        errors: list[ValidationErrorItem] = []

        if not isinstance(record, dict):
            return [ValidationErrorItem(line_number, "$", "Record must be a JSON object.")]

        # top-level required and additionalProperties:false behavior from schema v1
        missing = sorted(self.required_top_level - set(record.keys()))
        for key in missing:
            errors.append(ValidationErrorItem(line_number, "$", f"Missing required field '{key}'."))

        allowed_keys = {
            "sample_id",
            "date",
            "source",
            "text",
            "scores",
            "rationales",
            "confidence",
            "notes",
        }
        extra = sorted(set(record.keys()) - allowed_keys)
        for key in extra:
            errors.append(ValidationErrorItem(line_number, "$", f"Unexpected field '{key}'."))

        # validate present fields
        self._validate_non_empty_string(record, "sample_id", line_number, errors, allow_empty=False)
        self._validate_date(record, "date", line_number, errors)
        self._validate_non_empty_string(record, "source", line_number, errors, allow_empty=False)
        self._validate_non_empty_string(record, "text", line_number, errors, allow_empty=False)
        self._validate_scores(record.get("scores"), line_number, errors)
        self._validate_rationales(record.get("rationales"), line_number, errors)
        self._validate_confidence(record.get("confidence"), line_number, errors)
        self._validate_non_empty_string(record, "notes", line_number, errors, allow_empty=True)

        return errors

    def _validate_non_empty_string(
        self,
        record: dict[str, Any],
        key: str,
        line_number: int,
        errors: list[ValidationErrorItem],
        *,
        allow_empty: bool,
    ) -> None:
        if key not in record:
            return
        value = record[key]
        if not isinstance(value, str):
            errors.append(ValidationErrorItem(line_number, f"$.{key}", "Must be a string."))
            return
        if not allow_empty and value.strip() == "":
            errors.append(ValidationErrorItem(line_number, f"$.{key}", "Must not be empty."))

    def _validate_date(
        self,
        record: dict[str, Any],
        key: str,
        line_number: int,
        errors: list[ValidationErrorItem],
    ) -> None:
        if key not in record:
            return
        value = record[key]
        if not isinstance(value, str):
            errors.append(ValidationErrorItem(line_number, f"$.{key}", "Must be a string in YYYY-MM-DD format."))
            return
        parts = value.split("-")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            errors.append(ValidationErrorItem(line_number, f"$.{key}", "Must be in YYYY-MM-DD format."))
            return
        year, month, day = parts
        if len(year) != 4 or len(month) != 2 or len(day) != 2:
            errors.append(ValidationErrorItem(line_number, f"$.{key}", "Must be in YYYY-MM-DD format."))
            return
        try:
            import datetime as dt
            dt.date(int(year), int(month), int(day))
        except ValueError as exc:
            errors.append(ValidationErrorItem(line_number, f"$.{key}", f"Invalid calendar date: {exc}."))

    def _validate_scores(self, scores: Any, line_number: int, errors: list[ValidationErrorItem]) -> None:
        if scores is None:
            return
        if not isinstance(scores, dict):
            errors.append(ValidationErrorItem(line_number, "$.scores", "Must be an object."))
            return

        extra = sorted(set(scores.keys()) - set(EXPECTED_SCORE_KEYS))
        for key in extra:
            errors.append(ValidationErrorItem(line_number, "$.scores", f"Unexpected key '{key}'."))

        missing = sorted(set(EXPECTED_SCORE_KEYS) - set(scores.keys()))
        for key in missing:
            errors.append(ValidationErrorItem(line_number, "$.scores", f"Missing required key '{key}'."))

        for key in EXPECTED_SCORE_KEYS:
            if key not in scores:
                continue
            value = scores[key]
            if isinstance(value, int) and not isinstance(value, bool):
                if value < 0 or value > 3:
                    errors.append(ValidationErrorItem(line_number, f"$.scores.{key}", "Integer score must be between 0 and 3."))
            elif value == "NA":
                pass
            else:
                errors.append(ValidationErrorItem(line_number, f"$.scores.{key}", "Must be an integer 0-3 or the string 'NA'."))

    def _validate_rationales(self, rationales: Any, line_number: int, errors: list[ValidationErrorItem]) -> None:
        if rationales is None:
            return
        if not isinstance(rationales, dict):
            errors.append(ValidationErrorItem(line_number, "$.rationales", "Must be an object."))
            return

        extra = sorted(set(rationales.keys()) - set(EXPECTED_SCORE_KEYS))
        for key in extra:
            errors.append(ValidationErrorItem(line_number, "$.rationales", f"Unexpected key '{key}'."))

        missing = sorted(set(EXPECTED_SCORE_KEYS) - set(rationales.keys()))
        for key in missing:
            errors.append(ValidationErrorItem(line_number, "$.rationales", f"Missing required key '{key}'."))

        for key in EXPECTED_SCORE_KEYS:
            if key not in rationales:
                continue
            value = rationales[key]
            if not isinstance(value, str):
                errors.append(ValidationErrorItem(line_number, f"$.rationales.{key}", "Must be a string."))
            elif value.strip() == "":
                errors.append(ValidationErrorItem(line_number, f"$.rationales.{key}", "Must not be empty."))

    def _validate_confidence(self, value: Any, line_number: int, errors: list[ValidationErrorItem]) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            errors.append(ValidationErrorItem(line_number, "$.confidence", "Must be a string."))
            return
        if value not in EXPECTED_CONFIDENCE:
            allowed = ", ".join(sorted(EXPECTED_CONFIDENCE))
            errors.append(ValidationErrorItem(line_number, "$.confidence", f"Must be one of: {allowed}."))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ECOIN observation JSONL labels.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSONL file containing labeled records.",
    )
    parser.add_argument(
        "--schema",
        default="observation_schema_v1.json",
        help="Path to the schema file. Default: observation_schema_v1.json",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Stop printing after this many errors, while still counting all invalid lines if possible. Default: 50",
    )
    return parser.parse_args(argv)


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise SchemaLoadError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaLoadError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaLoadError(f"Expected a JSON object in {path}.")
    return data


def iter_jsonl_records(path: Path) -> Iterable[tuple[int, Any, str | None]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield idx, None, f"Invalid JSON: {exc}"


def validate_file(input_path: Path, schema_path: Path, max_errors: int) -> int:
    schema = load_json_file(schema_path)
    validator = RecordValidator(schema)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 2

    all_errors: list[ValidationErrorItem] = []
    total_records = 0
    invalid_records = 0

    for line_number, record, parse_error in iter_jsonl_records(input_path):
        total_records += 1
        if parse_error is not None:
            invalid_records += 1
            all_errors.append(ValidationErrorItem(line_number, "$", parse_error))
            continue

        record_errors = validator.validate_record(record, line_number)
        if record_errors:
            invalid_records += 1
            all_errors.extend(record_errors)

    if total_records == 0:
        print("ERROR: No JSONL records were found in the input file.", file=sys.stderr)
        return 2

    if all_errors:
        print(f"Validation failed: {invalid_records}/{total_records} records invalid.")
        for error in all_errors[:max_errors]:
            print(f"- {error.format()}")
        if len(all_errors) > max_errors:
            remaining = len(all_errors) - max_errors
            print(f"... and {remaining} more error(s).")
        return 1

    print(f"Validation passed: {total_records}/{total_records} records valid.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return validate_file(Path(args.input), Path(args.schema), args.max_errors)
    except SchemaLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
