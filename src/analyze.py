"""Post-process raw evaluation CSVs into summary tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DIR = Path("results/raw")
PROCESSED_DIR = Path("results/processed")

BASE_MAIN_PATH = RAW_DIR / "base_eval_main.csv"
DEON_MAIN_PATH = RAW_DIR / "deon_sft_main_eval.csv"
BASE_PERSONA_PATH = RAW_DIR / "base_persona_nl_eval.csv"
DEON_PERSONA_PATH = RAW_DIR / "deon_sft_persona_nl_eval.csv"


def _display_rate(value: float) -> str:
    return f"{value:.3f}"


def load_raw_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate main and persona raw evaluation results."""
    main = pd.concat(
        [
            pd.read_csv(BASE_MAIN_PATH),
            pd.read_csv(DEON_MAIN_PATH),
        ],
        ignore_index=True,
    )
    persona = pd.concat(
        [
            pd.read_csv(BASE_PERSONA_PATH),
            pd.read_csv(DEON_PERSONA_PATH),
        ],
        ignore_index=True,
    )

    for df in [main, persona]:
        df["cooperative"] = _as_bool_series(df["cooperative"])
        df["parseable"] = _as_bool_series(df["parseable"])

    return main, persona


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )


def compute_main_parse_rate(main: pd.DataFrame) -> pd.DataFrame:
    return (
        main.groupby(["model", "suite"], as_index=False)
        .agg(parse_rate=("parseable", "mean"), n=("parseable", "size"))
        .sort_values(["model", "suite"])
    )


def compute_main_cooperation_by_prior(main: pd.DataFrame) -> pd.DataFrame:
    return (
        main.groupby(["model", "suite", "prior"], as_index=False)
        .agg(
            cooperation_rate=("cooperative", "mean"),
            parse_rate=("parseable", "mean"),
            n=("cooperative", "size"),
        )
        .sort_values(["model", "suite", "prior"])
    )


def compute_main_conditionality_gap(
    cooperation_by_prior: pd.DataFrame,
) -> pd.DataFrame:
    pivot = cooperation_by_prior.pivot_table(
        index=["model", "suite"],
        columns="prior",
        values="cooperation_rate",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "cooperation": "cooperation_rate_prior_coop",
            "defection": "cooperation_rate_prior_defect",
        }
    )
    pivot["conditionality_gap"] = (
        pivot["cooperation_rate_prior_coop"]
        - pivot["cooperation_rate_prior_defect"]
    )
    return pivot[
        [
            "model",
            "suite",
            "cooperation_rate_prior_coop",
            "cooperation_rate_prior_defect",
            "conditionality_gap",
        ]
    ].sort_values(["model", "suite"])


def compute_persona_cooperation(persona: pd.DataFrame) -> pd.DataFrame:
    return (
        persona.groupby(["model", "persona", "prior"], as_index=False)
        .agg(
            cooperation_rate=("cooperative", "mean"),
            parse_rate=("parseable", "mean"),
            n=("cooperative", "size"),
        )
        .sort_values(["model", "persona", "prior"])
    )


def compute_persona_robustness_drop(persona: pd.DataFrame) -> pd.DataFrame:
    avg = (
        persona.groupby(["model", "persona"], as_index=False)
        .agg(avg_cooperation=("cooperative", "mean"))
        .sort_values(["model", "persona"])
    )
    neutral = (
        avg[avg["persona"] == "neutral"][["model", "avg_cooperation"]]
        .rename(columns={"avg_cooperation": "neutral_avg_cooperation"})
    )
    result = avg.merge(neutral, on="model", how="left")
    result["robustness_drop"] = (
        result["neutral_avg_cooperation"] - result["avg_cooperation"]
    )
    return result[
        [
            "model",
            "persona",
            "avg_cooperation",
            "neutral_avg_cooperation",
            "robustness_drop",
        ]
    ].sort_values(["model", "persona"])


def write_headline_numbers(
    conditionality_gap: pd.DataFrame,
    persona_robustness_drop: pd.DataFrame,
    path: str | Path = PROCESSED_DIR / "headline_numbers.md",
) -> None:
    lines = [
        "# Headline Numbers",
        "",
        "Cooperation rates treat unparseable outputs as non-cooperative.",
        "",
        "## Main Conditionality",
        "",
    ]

    for suite, label in [
        ("original_ipd", "Original IPD"),
        ("nl_ipd", "Natural-language IPD"),
    ]:
        lines.extend([f"### {label}", ""])
        suite_rows = conditionality_gap[conditionality_gap["suite"] == suite]
        for row in suite_rows.itertuples(index=False):
            lines.append(
                "- "
                f"{row.model}: prior cooperation={_display_rate(row.cooperation_rate_prior_coop)}, "
                f"prior defection={_display_rate(row.cooperation_rate_prior_defect)}, "
                f"gap={_display_rate(row.conditionality_gap)}"
            )
        lines.append("")

    lines.extend(["## Persona Robustness Drops", ""])
    for row in persona_robustness_drop.itertuples(index=False):
        lines.append(
            "- "
            f"{row.model} / {row.persona}: "
            f"avg cooperation={_display_rate(row.avg_cooperation)}, "
            f"neutral avg={_display_rate(row.neutral_avg_cooperation)}, "
            f"drop={_display_rate(row.robustness_drop)}"
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    main, persona = load_raw_results()
    main_parse_rate = compute_main_parse_rate(main)
    main_cooperation_by_prior = compute_main_cooperation_by_prior(main)
    main_conditionality_gap = compute_main_conditionality_gap(
        main_cooperation_by_prior
    )
    persona_cooperation = compute_persona_cooperation(persona)
    persona_robustness_drop = compute_persona_robustness_drop(persona)

    main_parse_rate.to_csv(PROCESSED_DIR / "main_parse_rate.csv", index=False)
    main_cooperation_by_prior.to_csv(
        PROCESSED_DIR / "main_cooperation_by_prior.csv",
        index=False,
    )
    main_conditionality_gap.to_csv(
        PROCESSED_DIR / "main_conditionality_gap.csv",
        index=False,
    )
    persona_cooperation.to_csv(
        PROCESSED_DIR / "persona_cooperation.csv",
        index=False,
    )
    persona_robustness_drop.to_csv(
        PROCESSED_DIR / "persona_robustness_drop.csv",
        index=False,
    )
    write_headline_numbers(main_conditionality_gap, persona_robustness_drop)


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
