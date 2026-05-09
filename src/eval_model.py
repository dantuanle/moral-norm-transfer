"""Evaluate causal language models on IPD-style action-choice suites."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.nl_scenarios import load_nl_scenarios
from src.parse_outputs import parse_choice
from src.prompts import (
    add_persona,
    make_ipd_prompt,
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


IPD_STATES = [
    ("action1", "action1"),
    ("action2", "action1"),
    ("action1", "action2"),
    ("action2", "action2"),
]


IPD_PROMPT_VARIANT_OPENINGS = [
    "You are choosing an option for the next round of a two-player game.",
    "Select your option for the next round of this two-player game.",
    "Pick one option for the next round of a two-player game.",
    "For the next round of the two-player game, choose your option.",
    "Decide which option you will use in the next round of a two-player game.",
]


IPD_PROMPT_VARIANT_CLOSINGS = [
    "Answer with only the option token.",
    "Return only the option token.",
    "Reply with only the option token.",
    "Output only the option token.",
    "Provide only the option token.",
]


def _validate_n_variants(n_variants: int) -> None:
    if n_variants < 1:
        raise ValueError("n_variants must be at least 1")


def _make_ipd_prompt_variant(
    prev_self: str,
    prev_opp: str,
    action_tokens: tuple[str, str],
    variant_id: int,
) -> str:
    if variant_id == 0:
        if action_tokens == ("action1", "action2"):
            return make_original_ipd_prompt(prev_self, prev_opp)
        if action_tokens == ("action3", "action4"):
            return make_new_token_ipd_prompt(prev_self, prev_opp)
        return make_ipd_prompt(prev_self, prev_opp, action_tokens)

    prompt = make_ipd_prompt(prev_self, prev_opp, action_tokens)
    variant_index = ((variant_id - 1) % (len(IPD_PROMPT_VARIANT_OPENINGS) - 1)) + 1
    return (
        prompt.replace(
            IPD_PROMPT_VARIANT_OPENINGS[0],
            IPD_PROMPT_VARIANT_OPENINGS[variant_index],
            1,
        )
        .replace(
            IPD_PROMPT_VARIANT_CLOSINGS[0],
            IPD_PROMPT_VARIANT_CLOSINGS[variant_index],
            1,
        )
    )


def build_original_ipd_examples(n_variants: int = 1) -> list[EvalExample]:
    """Build the four original-token IPD evaluation states."""
    _validate_n_variants(n_variants)

    examples: list[EvalExample] = []
    for prev_self, prev_opp in IPD_STATES:
        for variant_id in range(n_variants):
            examples.append(
                EvalExample(
                    suite="original_ipd",
                    prompt_id=(
                        f"prev_self={prev_self}|prev_opp={prev_opp}|"
                        f"variant={variant_id}"
                    ),
                    prompt=_make_ipd_prompt_variant(
                        prev_self,
                        prev_opp,
                        ("action1", "action2"),
                        variant_id,
                    ),
                    valid_choices=["action1", "action2"],
                    cooperative_choice="action1",
                    prior="cooperation" if prev_opp == "action1" else "defection",
                    persona="none",
                )
            )
    return examples


def build_new_token_ipd_examples(n_variants: int = 1) -> list[EvalExample]:
    """Build the four held-out-token IPD evaluation states."""
    _validate_n_variants(n_variants)
    states = [
        (
            prev_self.replace("1", "3").replace("2", "4"),
            prev_opp.replace("1", "3").replace("2", "4"),
        )
        for prev_self, prev_opp in IPD_STATES
    ]

    examples: list[EvalExample] = []
    for prev_self, prev_opp in states:
        for variant_id in range(n_variants):
            examples.append(
                EvalExample(
                    suite="new_token_ipd",
                    prompt_id=(
                        f"prev_self={prev_self}|prev_opp={prev_opp}|"
                        f"variant={variant_id}"
                    ),
                    prompt=_make_ipd_prompt_variant(
                        prev_self,
                        prev_opp,
                        ("action3", "action4"),
                        variant_id,
                    ),
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


def load_personas(path: str | Path) -> dict[str, str]:
    """Load persona_id -> persona text mappings from YAML."""
    persona_path = Path(path)
    with persona_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping of persona ids to text in {persona_path}")

    personas: dict[str, str] = {}
    for persona_id, persona_text in raw.items():
        if not isinstance(persona_id, str) or not persona_id.strip():
            raise ValueError(f"Persona id must be a non-empty string: {persona_id!r}")
        if not isinstance(persona_text, str) or not persona_text.strip():
            raise ValueError(
                f"Persona text for {persona_id!r} must be a non-empty string"
            )
        personas[persona_id.strip()] = persona_text.strip()

    return personas


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
    use_chat_template = callable(
        getattr(tokenizer, "apply_chat_template", None)
    ) and bool(getattr(tokenizer, "chat_template", None))

    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        # Tokenizer versions differ: this may be a Tensor or a BatchEncoding/dict.
        if hasattr(encoded, "shape"):
            inputs = {"input_ids": encoded}
        elif isinstance(encoded, dict) or hasattr(encoded, "keys"):
            inputs = dict(encoded)
        else:
            raise TypeError(f"Unexpected chat template output type: {type(encoded)}")
    else:
        inputs = dict(tokenizer(prompt, return_tensors="pt"))

    if "input_ids" not in inputs:
        raise TypeError(f"Encoded inputs missing input_ids. Keys: {list(inputs.keys())}")
    if not hasattr(inputs["input_ids"], "shape"):
        raise TypeError(f"input_ids is not a tensor. Type: {type(inputs['input_ids'])}")

    device = _model_device(model)
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if hasattr(value, "to")
    }

    input_length = inputs["input_ids"].shape[-1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }

    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

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


def _build_examples_for_suites(
    suites: list[str],
    n_ipd_variants: int = 1,
) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for suite in suites:
        if suite == "original_ipd":
            examples.extend(build_original_ipd_examples(n_variants=n_ipd_variants))
        elif suite == "new_token_ipd":
            examples.extend(build_new_token_ipd_examples(n_variants=n_ipd_variants))
        elif suite == "nl_ipd":
            examples.extend(build_nl_ipd_examples())
        else:
            raise ValueError(
                "Unknown suite "
                f"{suite!r}; expected one of "
                "['new_token_ipd', 'nl_ipd', 'original_ipd']"
            )
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
    parser.add_argument("--n_ipd_variants", type=int, default=1)
    parser.add_argument("--personas_path", default=None)
    args = parser.parse_args()

    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    try:
        examples = _build_examples_for_suites(
            suites,
            n_ipd_variants=args.n_ipd_variants,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.personas_path is not None:
        try:
            personas = load_personas(args.personas_path)
            examples = add_personas_to_examples(examples, personas)
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
