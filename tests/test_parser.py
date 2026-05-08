import pytest

from src.parse_outputs import parse_choice


@pytest.mark.parametrize(
    ("text", "valid_choices", "expected"),
    [
        ("action1", ["action1", "action2"], "action1"),
        (" action1\n", ["action1", "action2"], "action1"),
        ('"action1"', ["action1", "action2"], "action1"),
        ("'action1'", ["action1", "action2"], "action1"),
        ("Answer: action1", ["action1", "action2"], "action1"),
        ("Final: action1", ["action1", "action2"], "action1"),
        ("FINAL: A", ["A", "B"], "A"),
        ("A.", ["A", "B"], "A"),
    ],
)
def test_parse_choice_accepts_short_outputs(text, valid_choices, expected):
    assert parse_choice(text, valid_choices) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I choose action1 because it is better.",
        "A, because that is fair.",
        "The answer is action1.",
        "Answer: I choose action1.",
        "FINAL: A, because that is fair.",
    ],
)
def test_parse_choice_rejects_long_explanations(text):
    assert parse_choice(text, ["action1", "action2", "A", "B"]) is None


@pytest.mark.parametrize(
    ("text", "valid_choices"),
    [
        ("Action1", ["action1", "action2"]),
        ("Answer: Action1", ["action1", "action2"]),
        ("a", ["A", "B"]),
        ("b.", ["A", "B"]),
    ],
)
def test_parse_choice_is_case_sensitive(text, valid_choices):
    assert parse_choice(text, valid_choices) is None


def test_parse_choice_allows_lowercase_when_explicitly_valid():
    assert parse_choice("a", ["A", "a"]) == "a"

def test_parse_choice_requires_non_empty_valid_choices():
    with pytest.raises(ValueError):
        parse_choice("action1", [])