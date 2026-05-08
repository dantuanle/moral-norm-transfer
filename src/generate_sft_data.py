"""Generate narrow SFT data for the prior-opponent-cooperation IPD norm."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.games import TRAIN_ACTION_TOKENS, cooperative_token, defective_token
from src.prompts import make_original_ipd_prompt


SPLIT_SOURCE = "prior_opp_coop_only"


def generate_deon_sft_records(n_examples: int, seed: int = 0) -> list[dict[str, str]]:
    """Generate SFT records where the previous opponent action was cooperative."""
    if n_examples < 0:
        raise ValueError("n_examples must be non-negative")

    coop_token = cooperative_token(TRAIN_ACTION_TOKENS)
    defect_token = defective_token(TRAIN_ACTION_TOKENS)
    prev_self_tokens = [coop_token, defect_token] * (n_examples // 2)
    if n_examples % 2:
        prev_self_tokens.append(coop_token)

    rng = random.Random(seed)
    rng.shuffle(prev_self_tokens)

    records = []
    for prev_self_token in prev_self_tokens:
        prompt = make_original_ipd_prompt(
            prev_self_token=prev_self_token,
            prev_opp_token=coop_token,
        )
        records.append(
            {
                "prompt": prompt,
                "completion": coop_token,
                "prev_self_token": prev_self_token,
                "prev_opp_token": coop_token,
                "target_action": coop_token,
                "split_source": SPLIT_SOURCE,
            }
        )

    return records


def write_jsonl(records: list[dict[str, str]], path: str | Path) -> None:
    """Write records to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_train", type=int, required=True)
    parser.add_argument("--n_val", type=int, required=True)
    parser.add_argument("--out_train", type=Path, required=True)
    parser.add_argument("--out_val", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_records = generate_deon_sft_records(args.n_train, seed=args.seed)
    val_records = generate_deon_sft_records(args.n_val, seed=args.seed + 1)

    write_jsonl(train_records, args.out_train)
    write_jsonl(val_records, args.out_val)


if __name__ == "__main__":
    main()
