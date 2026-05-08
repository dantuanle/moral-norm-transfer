"""Game definitions and simple norm logic for the moral-norm-transfer project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Action = Literal["C", "D"]


C: Action = "C"
D: Action = "D"


# Standard IPD payoff matrix:
# T = 4, R = 3, P = 1, S = 0
# Values are (agent_payoff, opponent_payoff).
IPD_PAYOFFS: dict[tuple[Action, Action], tuple[int, int]] = {
    (C, C): (3, 3),
    (C, D): (0, 4),
    (D, C): (4, 0),
    (D, D): (1, 1),
}


TRAIN_ACTION_TOKENS: dict[str, Action] = {
    "action1": C,
    "action2": D,
}


NEW_ACTION_TOKENS: dict[str, Action] = {
    "action3": C,
    "action4": D,
}


NL_ACTION_TOKENS: dict[str, Action] = {
    "A": C,
    "B": D,
}


@dataclass(frozen=True)
class IPDState:
    """One-step memory state for IPD-style prompts."""

    prev_self_action: Action
    prev_opp_action: Action


def payoff(agent_action: Action, opponent_action: Action) -> tuple[int, int]:
    """Return (agent_payoff, opponent_payoff) for one IPD round."""
    try:
        return IPD_PAYOFFS[(agent_action, opponent_action)]
    except KeyError as exc:
        raise ValueError(
            f"Invalid IPD action pair: {(agent_action, opponent_action)}"
        ) from exc


def deon_preferred_action(prev_opp_action: Action) -> Action | None:
    """Return the action preferred by the deontological norm.

    Tennant's deontological norm is conditional:
    do not defect against an opponent who previously cooperated.

    Therefore:
    - if previous opponent action was C, the norm prescribes C;
    - if previous opponent action was D, the norm is silent.
    """
    if prev_opp_action == C:
        return C
    if prev_opp_action == D:
        return None
    raise ValueError(f"Invalid previous opponent action: {prev_opp_action}")


def token_to_action(token: str, token_map: dict[str, Action]) -> Action:
    """Map an action token to C/D."""
    try:
        return token_map[token]
    except KeyError as exc:
        raise ValueError(f"Unknown action token {token!r}") from exc


def cooperative_token(token_map: dict[str, Action]) -> str:
    """Return the token corresponding to cooperation in a token map."""
    coop_tokens = [token for token, action in token_map.items() if action == C]
    if len(coop_tokens) != 1:
        raise ValueError(f"Expected exactly one cooperative token, got {coop_tokens}")
    return coop_tokens[0]


def defective_token(token_map: dict[str, Action]) -> str:
    """Return the token corresponding to defection in a token map."""
    defect_tokens = [token for token, action in token_map.items() if action == D]
    if len(defect_tokens) != 1:
        raise ValueError(f"Expected exactly one defective token, got {defect_tokens}")
    return defect_tokens[0]