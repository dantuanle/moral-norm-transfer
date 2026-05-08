import pytest

from src.games import (
    C,
    D,
    IPD_PAYOFFS,
    NEW_ACTION_TOKENS,
    NL_ACTION_TOKENS,
    TRAIN_ACTION_TOKENS,
    cooperative_token,
    defective_token,
    deon_preferred_action,
    payoff,
    token_to_action,
)


def test_ipd_payoff_matrix_values():
    assert IPD_PAYOFFS[(C, C)] == (3, 3)
    assert IPD_PAYOFFS[(C, D)] == (0, 4)
    assert IPD_PAYOFFS[(D, C)] == (4, 0)
    assert IPD_PAYOFFS[(D, D)] == (1, 1)


def test_payoff_function():
    assert payoff(C, C) == (3, 3)
    assert payoff(C, D) == (0, 4)
    assert payoff(D, C) == (4, 0)
    assert payoff(D, D) == (1, 1)


def test_payoff_rejects_invalid_action():
    with pytest.raises(ValueError):
        payoff("X", C)  # type: ignore[arg-type]


def test_deon_preferred_action():
    assert deon_preferred_action(C) == C
    assert deon_preferred_action(D) is None


def test_deon_preferred_action_rejects_invalid_action():
    with pytest.raises(ValueError):
        deon_preferred_action("X")  # type: ignore[arg-type]


def test_train_action_tokens():
    assert TRAIN_ACTION_TOKENS["action1"] == C
    assert TRAIN_ACTION_TOKENS["action2"] == D
    assert cooperative_token(TRAIN_ACTION_TOKENS) == "action1"
    assert defective_token(TRAIN_ACTION_TOKENS) == "action2"


def test_new_action_tokens():
    assert NEW_ACTION_TOKENS["action3"] == C
    assert NEW_ACTION_TOKENS["action4"] == D
    assert cooperative_token(NEW_ACTION_TOKENS) == "action3"
    assert defective_token(NEW_ACTION_TOKENS) == "action4"


def test_nl_action_tokens():
    assert NL_ACTION_TOKENS["A"] == C
    assert NL_ACTION_TOKENS["B"] == D
    assert cooperative_token(NL_ACTION_TOKENS) == "A"
    assert defective_token(NL_ACTION_TOKENS) == "B"


def test_token_to_action():
    assert token_to_action("action1", TRAIN_ACTION_TOKENS) == C
    assert token_to_action("action2", TRAIN_ACTION_TOKENS) == D
    assert token_to_action("action3", NEW_ACTION_TOKENS) == C
    assert token_to_action("action4", NEW_ACTION_TOKENS) == D
    assert token_to_action("A", NL_ACTION_TOKENS) == C
    assert token_to_action("B", NL_ACTION_TOKENS) == D


def test_token_to_action_rejects_unknown_token():
    with pytest.raises(ValueError):
        token_to_action("action999", TRAIN_ACTION_TOKENS)