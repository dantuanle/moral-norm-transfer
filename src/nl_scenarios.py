"""Loading and validation for natural-language reciprocal dilemma scenarios."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_PRIORS = {"cooperation", "defection"}
VALID_DIFFICULTIES = {"easy", "hard"}
VALID_OPTIONS = {"A", "B"}


@dataclass(frozen=True)
class NLScenario:
    id: str
    domain: str
    difficulty: str
    prior: str
    prompt: str
    cooperative_option: str


def _require_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Scenario is missing non-empty string field {key!r}: {record}")
    return value.strip()


def _validate_scenario(scenario: NLScenario) -> None:
    if scenario.prior not in VALID_PRIORS:
        raise ValueError(f"{scenario.id}: prior must be one of {VALID_PRIORS}, got {scenario.prior!r}")

    if scenario.difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            f"{scenario.id}: difficulty must be one of {VALID_DIFFICULTIES}, got {scenario.difficulty!r}"
        )

    if scenario.cooperative_option not in VALID_OPTIONS:
        raise ValueError(
            f"{scenario.id}: cooperative_option must be one of {VALID_OPTIONS}, "
            f"got {scenario.cooperative_option!r}"
        )

    required_prompt_snippets = [
        "Option A:",
        "Option B:",
        "Choose exactly one option: A or B.",
        "Answer with only the option.",
    ]

    for snippet in required_prompt_snippets:
        if snippet not in scenario.prompt:
            raise ValueError(f"{scenario.id}: prompt is missing required snippet {snippet!r}")


def load_nl_scenarios(path: str | Path) -> list[NLScenario]:
    """Load and validate natural-language reciprocal dilemma scenarios."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected YAML list of scenarios in {path}, got {type(raw).__name__}")

    scenarios: list[NLScenario] = []

    for record in raw:
        if not isinstance(record, dict):
            raise ValueError(f"Expected each scenario to be a mapping, got {record!r}")

        scenario = NLScenario(
            id=_require_string(record, "id"),
            domain=_require_string(record, "domain"),
            difficulty=_require_string(record, "difficulty"),
            prior=_require_string(record, "prior"),
            prompt=_require_string(record, "prompt"),
            cooperative_option=_require_string(record, "cooperative_option"),
        )

        _validate_scenario(scenario)
        scenarios.append(scenario)

    ids = [scenario.id for scenario in scenarios]
    duplicate_ids = [scenario_id for scenario_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate scenario ids found: {duplicate_ids}")

    by_domain: dict[str, list[str]] = defaultdict(list)
    for scenario in scenarios:
        by_domain[scenario.domain].append(scenario.prior)

    for domain, priors in by_domain.items():
        prior_counts = Counter(priors)
        if prior_counts["cooperation"] != 1 or prior_counts["defection"] != 1:
            raise ValueError(
                f"Domain {domain!r} must have exactly one cooperation and one defection scenario, "
                f"got {dict(prior_counts)}"
            )

    return scenarios