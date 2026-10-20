"""Load and validate suites from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from evalkit.errors import EvalkitError, SuiteError
from evalkit.models import Suite
from evalkit.scorers import build_scorer


def parse_suite(data: Any, *, source: str = "<suite>") -> Suite:
    """Validate an already-parsed mapping, including every scorer's parameters."""
    if not isinstance(data, dict):
        raise SuiteError(f"{source}: top level must be a mapping")
    try:
        suite = Suite.model_validate(data)
    except ValidationError as exc:
        raise SuiteError(f"{source}: {exc}") from exc
    for case in suite.cases:
        specs = suite.scorers_for(case)
        if not specs:
            raise SuiteError(f"{source}: case {case.id!r} has no scorers (and no defaults)")
        for spec in specs:
            try:
                build_scorer(spec)
            except EvalkitError as exc:
                raise SuiteError(f"{source}: case {case.id!r}: {exc}") from exc
    return suite


def load_suite(path: str | Path) -> Suite:
    """Read a suite YAML file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SuiteError(f"cannot read suite {path}: {exc.strerror or exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SuiteError(f"{path}: invalid YAML: {exc}") from exc
    return parse_suite(data, source=str(path))
