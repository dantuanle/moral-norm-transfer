from collections import Counter

import pytest

from src.nl_scenarios import load_nl_scenarios


SCENARIO_PATH = "data/eval/nl_scenarios.yaml"


def test_load_nl_scenarios_count():
    scenarios = load_nl_scenarios(SCENARIO_PATH)
    assert len(scenarios) == 20


def test_load_nl_scenarios_unique_ids():
    scenarios = load_nl_scenarios(SCENARIO_PATH)
    ids = [scenario.id for scenario in scenarios]
    assert len(ids) == len(set(ids))


def test_load_nl_scenarios_unique_domains():
    scenarios = load_nl_scenarios(SCENARIO_PATH)
    domains = {scenario.domain for scenario in scenarios}
    assert len(domains) == 10


def test_each_domain_has_one_coop_and_one_defect_scenario():
    scenarios = load_nl_scenarios(SCENARIO_PATH)

    priors_by_domain = {}
    for scenario in scenarios:
        priors_by_domain.setdefault(scenario.domain, []).append(scenario.prior)

    for priors in priors_by_domain.values():
        assert Counter(priors) == Counter({"cooperation": 1, "defection": 1})


def test_all_current_scenarios_use_a_as_cooperative_option():
    scenarios = load_nl_scenarios(SCENARIO_PATH)
    assert {scenario.cooperative_option for scenario in scenarios} == {"A"}


def test_all_prompts_have_required_format():
    scenarios = load_nl_scenarios(SCENARIO_PATH)

    for scenario in scenarios:
        assert "Option A:" in scenario.prompt
        assert "Option B:" in scenario.prompt
        assert "Choose exactly one option: A or B." in scenario.prompt
        assert "Answer with only the option." in scenario.prompt


def test_invalid_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_nl_scenarios("data/eval/does_not_exist.yaml")