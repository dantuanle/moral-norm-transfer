import csv

from src import eval_model
from src.eval_model import (
    CSV_COLUMNS,
    EvalExample,
    add_personas_to_examples,
    build_new_token_ipd_examples,
    build_nl_ipd_examples,
    build_original_ipd_examples,
    evaluate_examples,
    write_csv,
)


def test_build_original_ipd_examples_returns_four_examples():
    examples = build_original_ipd_examples()

    assert len(examples) == 4
    assert {example.suite for example in examples} == {"original_ipd"}


def test_build_new_token_ipd_examples_returns_four_examples():
    examples = build_new_token_ipd_examples()

    assert len(examples) == 4
    assert {example.suite for example in examples} == {"new_token_ipd"}


def test_original_examples_use_action1_action2_only():
    examples = build_original_ipd_examples()

    for example in examples:
        assert example.valid_choices == ["action1", "action2"]
        assert example.cooperative_choice == "action1"
        assert "action1" in example.prompt
        assert "action2" in example.prompt
        assert "action3" not in example.prompt
        assert "action4" not in example.prompt


def test_new_token_examples_use_action3_action4_only():
    examples = build_new_token_ipd_examples()

    for example in examples:
        assert example.valid_choices == ["action3", "action4"]
        assert example.cooperative_choice == "action3"
        assert "action3" in example.prompt
        assert "action4" in example.prompt
        assert "action1" not in example.prompt
        assert "action2" not in example.prompt


def test_original_prior_labels_are_correct():
    examples = build_original_ipd_examples()

    priors_by_id = {example.prompt_id: example.prior for example in examples}

    assert priors_by_id["prev_self=action1|prev_opp=action1"] == "cooperation"
    assert priors_by_id["prev_self=action2|prev_opp=action1"] == "cooperation"
    assert priors_by_id["prev_self=action1|prev_opp=action2"] == "defection"
    assert priors_by_id["prev_self=action2|prev_opp=action2"] == "defection"


def test_new_token_prior_labels_are_correct():
    examples = build_new_token_ipd_examples()

    priors_by_id = {example.prompt_id: example.prior for example in examples}

    assert priors_by_id["prev_self=action3|prev_opp=action3"] == "cooperation"
    assert priors_by_id["prev_self=action4|prev_opp=action3"] == "cooperation"
    assert priors_by_id["prev_self=action3|prev_opp=action4"] == "defection"
    assert priors_by_id["prev_self=action4|prev_opp=action4"] == "defection"


def test_build_nl_ipd_examples_returns_twenty_examples():
    examples = build_nl_ipd_examples()

    assert len(examples) == 20
    assert {example.suite for example in examples} == {"nl_ipd"}


def test_nl_examples_use_a_b_valid_choices():
    examples = build_nl_ipd_examples()

    assert {tuple(example.valid_choices) for example in examples} == {("A", "B")}


def test_nl_cooperative_choice_can_be_a_or_b():
    examples = build_nl_ipd_examples()

    assert {example.cooperative_choice for example in examples} == {"A", "B"}


def test_add_personas_to_examples_prepends_persona_and_sets_persona_id():
    examples = [
        EvalExample(
            suite="test",
            prompt_id="example-1",
            prompt="Choose exactly one option.",
            valid_choices=["A", "B"],
            cooperative_choice="A",
            prior="cooperation",
            persona="none",
        )
    ]
    personas = {"steady": "You prefer steady outcomes."}

    expanded = add_personas_to_examples(examples, personas)

    assert len(expanded) == 1
    assert expanded[0].persona == "steady"
    assert expanded[0].prompt == (
        "You prefer steady outcomes.\n\nChoose exactly one option."
    )


def test_add_personas_to_examples_with_no_personas_leaves_examples_unchanged():
    examples = build_original_ipd_examples()

    assert add_personas_to_examples(examples, {}) == examples


def test_evaluate_examples_uses_generation_and_parsing(monkeypatch):
    examples = [
        EvalExample(
            suite="test",
            prompt_id="parseable",
            prompt="Prompt one",
            valid_choices=["A", "B"],
            cooperative_choice="A",
            prior="cooperation",
            persona="none",
        ),
        EvalExample(
            suite="test",
            prompt_id="unparseable",
            prompt="Prompt two",
            valid_choices=["A", "B"],
            cooperative_choice="A",
            prior="defection",
            persona="none",
        ),
    ]

    def fake_generate_one(model, tokenizer, prompt, **kwargs):
        if prompt == "Prompt one":
            return "Answer: A"
        return "I choose A because it is better."

    monkeypatch.setattr(eval_model, "generate_one", fake_generate_one)

    rows = evaluate_examples(
        model=object(),
        tokenizer=object(),
        examples=examples,
        model_label="fake",
        seed=7,
        max_new_tokens=3,
    )

    assert rows[0]["model"] == "fake"
    assert rows[0]["parsed_output"] == "A"
    assert rows[0]["cooperative"] is True
    assert rows[0]["parseable"] is True
    assert rows[0]["seed"] == 7
    assert rows[1]["parsed_output"] is None
    assert rows[1]["cooperative"] is False
    assert rows[1]["parseable"] is False


def test_write_csv_creates_readable_csv_with_expected_columns(tmp_path):
    rows = [
        {
            "model": "fake",
            "suite": "test",
            "prompt_id": "example-1",
            "prior": "cooperation",
            "persona": "none",
            "seed": 0,
            "prompt": "Choose one.",
            "raw_output": "A",
            "parsed_output": "A",
            "cooperative_choice": "A",
            "cooperative": True,
            "parseable": True,
        }
    ]
    output_path = tmp_path / "eval.csv"

    write_csv(rows, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        loaded_rows = list(reader)

    assert reader.fieldnames == CSV_COLUMNS
    assert loaded_rows[0]["model"] == "fake"
    assert loaded_rows[0]["parsed_output"] == "A"
    assert loaded_rows[0]["cooperative"] == "True"
    assert loaded_rows[0]["parseable"] == "True"
