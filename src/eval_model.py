"""Evaluate causal language models on IPD-style action-choice suites."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.nl_scenarios import load_nl_scenarios
from src.parse_outputs import parse_choice
from src.prompts import (
    add_persona,
    make_new_token_ipd_prompt,
    make_original_ipd_prompt,
)


CSV_COLUMNS = [
    "model",
    "suite",
    "prompt_id",
    "prior",
    "persona",
    "seed",
    "prompt",
    "raw_output",
    "parsed_output",
    "cooperative_choice",
    "cooperative",
    "parseable",
]


@dataclass(frozen=True)
class EvalExample:
    suite: str
    prompt_id: str
    prompt: str
    valid_choices: list[str]
    cooperative_choice: str
    prior: str
    persona: str


def build_original_ipd_examples() -> list[EvalExample]:
    """Build the four original-token IPD evaluation states."""
    states = [
        ("action1", "action1"),
        ("action2", "action1"),
        ("action1", "action2"),
        ("action2", "action2"),
    ]

    examples: list[EvalExample] = []
    for prev_self, prev_opp in states:
        examples.append(
            EvalExample(
                suite="original_ipd",
                prompt_id=f"prev_self={prev_self}|prev_opp={prev_opp}",
                prompt=make_original_ipd_prompt(prev_self, prev_opp),
                valid_choices=["action1", "action2"],
                cooperative_choice="action1",
                prior="cooperation" if prev_opp == "action1" else "defection",
                persona="none",
            )
        )
    return examples


def build_new_token_ipd_examples() -> list[EvalExample]:
    """Build the four held-out-token IPD evaluation states."""
    states = [
        ("action3", "action3"),
        ("action4", "action3"),
        ("action3", "action4"),
        ("action4", "action4"),
    ]

    examples: list[EvalExample] = []
    for prev_self, prev_opp in states:
        examples.append(
            EvalExample(
                suite="new_token_ipd",
                prompt_id=f"prev_self={prev_self}|prev_opp={prev_opp}",
                prompt=make_new_token_ipd_prompt(prev_self, prev_opp),
                valid_choices=["action3", "action4"],
                cooperative_choice="action3",
                prior="cooperation" if prev_opp == "action3" else "defection",
                persona="none",
            )
        )
    return examples


def build_nl_ipd_examples(
    path: str | Path = "data/eval/nl_scenarios.yaml",
) -> list[EvalExample]:
    """Build natural-language IPD examples from the scenario YAML file."""
    return [
        EvalExample(
            suite="nl_ipd",
            prompt_id=scenario.id,
            prompt=scenario.prompt,
            valid_choices=["A", "B"],
            cooperative_choice=scenario.cooperative_option,
            prior=scenario.prior,
            persona="none",
        )
        for scenario in load_nl_scenarios(path)
    ]


def add_personas_to_examples(
    examples: list[EvalExample],
    personas: dict[str, str],
) -> list[EvalExample]:
    """Return examples with each persona prepended to each prompt."""
    if not personas:
        return examples

    expanded: list[EvalExample] = []
    for persona_id, persona_text in personas.items():
        for example in examples:
            expanded.append(
                EvalExample(
                    suite=example.suite,
                    prompt_id=example.prompt_id,
                    prompt=add_persona(example.prompt, persona_text),
                    valid_choices=list(example.valid_choices),
                    cooperative_choice=example.cooperative_choice,
                    prior=example.prior,
                    persona=persona_id,
                )
            )
    return expanded


def load_model_and_tokenizer(
    model_name: str,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """Load a causal LM, tokenizer, and optional PEFT adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=dtype,
    )

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def _model_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return "cpu"


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 5,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Generate and decode only the continuation for one prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(_model_device(model))
    input_length = inputs["input_ids"].shape[-1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **generation_kwargs)
    new_token_ids = output_ids[0][input_length:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def evaluate_examples(
    model: Any,
    tokenizer: Any,
    examples: list[EvalExample],
    model_label: str,
    seed: int = 0,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 5,
) -> list[dict[str, Any]]:
    """Run generation and strict parsing for each evaluation example."""
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    rows: list[dict[str, Any]] = []
    for example in examples:
        raw_output = generate_one(
            model,
            tokenizer,
            example.prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        parsed_output = parse_choice(raw_output, example.valid_choices)
        parseable = parsed_output is not None
        cooperative = parsed_output == example.cooperative_choice

        rows.append(
            {
                "model": model_label,
                "suite": example.suite,
                "prompt_id": example.prompt_id,
                "prior": example.prior,
                "persona": example.persona,
                "seed": seed,
                "prompt": example.prompt,
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "cooperative_choice": example.cooperative_choice,
                "cooperative": cooperative,
                "parseable": parseable,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write evaluation rows to CSV with a stable column order."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_examples_for_suites(suites: list[str]) -> list[EvalExample]:
    builders = {
        "original_ipd": build_original_ipd_examples,
        "new_token_ipd": build_new_token_ipd_examples,
        "nl_ipd": build_nl_ipd_examples,
    }

    examples: list[EvalExample] = []
    for suite in suites:
        try:
            builder = builders[suite]
        except KeyError as exc:
            raise ValueError(
                f"Unknown suite {suite!r}; expected one of {sorted(builders)}"
            ) from exc
        examples.extend(builder())
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="google/gemma-2-2b-it")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--model_label", default="base")
    parser.add_argument(
        "--suites",
        default="original_ipd,new_token_ipd,nl_ipd",
        help="Comma-separated suite names.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    args = parser.parse_args()

    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    try:
        examples = _build_examples_for_suites(suites)
    except ValueError as exc:
        parser.error(str(exc))

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adapter_path)
    rows = evaluate_examples(
        model,
        tokenizer,
        examples,
        model_label=args.model_label,
        seed=args.seed,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
