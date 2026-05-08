import pytest
from src.prompts import (
    add_persona,
    make_ipd_prompt,
    make_new_token_ipd_prompt,
    make_original_ipd_prompt,
)

def test_make_ipd_prompt_rejects_wrong_number_of_tokens():
    with pytest.raises(ValueError):
        make_ipd_prompt("action1", "action2", ("action1",))  # type: ignore[arg-type]

def test_original_prompt_uses_training_tokens_only():
    prompt = make_original_ipd_prompt("action1", "action2")

    assert "action1" in prompt
    assert "action2" in prompt
    assert "action3" not in prompt
    assert "action4" not in prompt


def test_new_token_prompt_uses_new_tokens_only():
    prompt = make_new_token_ipd_prompt("action3", "action4")

    assert "action3" in prompt
    assert "action4" in prompt
    assert "action1" not in prompt
    assert "action2" not in prompt


def test_prompt_includes_all_four_payoff_entries():
    prompt = make_ipd_prompt("action1", "action2")

    assert "action1/action1: 3,3" in prompt
    assert "action1/action2: 0,4" in prompt
    assert "action2/action1: 4,0" in prompt
    assert "action2/action2: 1,1" in prompt


def test_prompt_includes_previous_self_and_opponent_actions():
    prompt = make_ipd_prompt("action1", "action2")

    assert "Your previous option: action1" in prompt
    assert "Other player's previous option: action2" in prompt


def test_prompt_does_not_include_banned_words():
    prompt = make_ipd_prompt("action1", "action2")
    prompt_lower = prompt.lower()

    for banned_word in ["cooperate", "defect", "prisoner's dilemma", "moral", "fair"]:
        assert banned_word not in prompt_lower


def test_add_persona_prepends_persona_text():
    prompt = "Choose one action."
    persona = "You prefer steady outcomes."

    result = add_persona(prompt, persona)

    assert result.startswith(f"{persona}\n\n")
    assert result.endswith(prompt)

def test_add_persona_strips_extra_whitespace():
    prompt = "  Choose one option.  "
    persona = "  You prefer steady outcomes.  "

    result = add_persona(prompt, persona)

    assert result == "You prefer steady outcomes.\n\nChoose one option."