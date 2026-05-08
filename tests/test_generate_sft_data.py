import json
from collections import Counter

import pytest

from src.generate_sft_data import (
    SPLIT_SOURCE,
    generate_deon_sft_records,
    write_jsonl,
)


REQUIRED_FIELDS = {
    "prompt",
    "completion",
    "prev_self_token",
    "prev_opp_token",
    "target_action",
    "split_source",
}


def test_generate_deon_sft_records_count():
    records = generate_deon_sft_records(8)
    assert len(records) == 8


def test_generate_deon_sft_records_required_fields_exist_exactly():
    records = generate_deon_sft_records(4)
    for record in records:
        assert set(record) == REQUIRED_FIELDS


def test_generate_deon_sft_records_prior_opp_is_always_action1():
    records = generate_deon_sft_records(8)
    assert {record["prev_opp_token"] for record in records} == {"action1"}


def test_generate_deon_sft_records_completion_is_always_action1():
    records = generate_deon_sft_records(8)
    assert {record["completion"] for record in records} == {"action1"}


def test_generate_deon_sft_records_target_action_is_always_action1():
    records = generate_deon_sft_records(8)
    assert {record["target_action"] for record in records} == {"action1"}


def test_generate_deon_sft_records_split_source_is_expected_value():
    records = generate_deon_sft_records(8)
    assert {record["split_source"] for record in records} == {SPLIT_SOURCE}


def test_generate_deon_sft_records_prev_self_token_balanced_for_even_n():
    records = generate_deon_sft_records(10)
    prev_self_counts = Counter(record["prev_self_token"] for record in records)
    assert prev_self_counts == Counter({"action1": 5, "action2": 5})


def test_generate_deon_sft_records_prompts_include_previous_tokens():
    records = generate_deon_sft_records(12)

    for record in records:
        assert f"Your previous option: {record['prev_self_token']}" in record["prompt"]
        assert (
            f"Other player's previous option: {record['prev_opp_token']}"
            in record["prompt"]
        )


def test_generate_deon_sft_records_prompts_use_training_tokens_only():
    records = generate_deon_sft_records(8)

    for record in records:
        assert "action1" in record["prompt"]
        assert "action2" in record["prompt"]
        assert "action3" not in record["prompt"]
        assert "action4" not in record["prompt"]


def test_write_jsonl_creates_valid_jsonl(tmp_path):
    records = generate_deon_sft_records(3)
    output_path = tmp_path / "sft.jsonl"

    write_jsonl(records, output_path)

    loaded_records = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert loaded_records == records


def test_generate_deon_sft_records_negative_n_examples_raises_value_error():
    with pytest.raises(ValueError):
        generate_deon_sft_records(-1)

def test_generate_deon_sft_records_is_deterministic_for_same_seed():
    records_a = generate_deon_sft_records(10, seed=123)
    records_b = generate_deon_sft_records(10, seed=123)
    assert records_a == records_b