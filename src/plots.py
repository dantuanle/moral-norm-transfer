"""Create matplotlib figures from processed evaluation summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROCESSED_DIR = Path("results/processed")
FIGURES_DIR = Path("results/figures")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _grouped_bar(
    rows: list[dict[str, str]],
    x_key: str,
    group_key: str,
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x_values = sorted({row[x_key] for row in rows})
    groups = sorted({row[group_key] for row in rows})
    values = {
        (row[x_key], row[group_key]): float(row[value_key])
        for row in rows
    }

    x_positions = list(range(len(x_values)))
    width = 0.8 / max(len(groups), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(x_values) * 1.2), 5))
    for group_index, group in enumerate(groups):
        offsets = [
            position - 0.4 + width / 2 + group_index * width
            for position in x_positions
        ]
        heights = [values.get((x_value, group), 0.0) for x_value in x_values]
        ax.bar(offsets, heights, width=width, label=group)

    all_values = [float(row[value_key]) for row in rows]
    ymin = min(0.0, min(all_values, default=0.0) - 0.05)
    ymax = max(1.0, max(all_values, default=1.0) + 0.05)
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_values, rotation=30, ha="right")
    ax.legend(title=group_key)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_main_conditionality_gap() -> None:
    rows = _read_csv(PROCESSED_DIR / "main_conditionality_gap.csv")
    _grouped_bar(
        rows=rows,
        x_key="suite",
        group_key="model",
        value_key="conditionality_gap",
        title="Main Conditionality Gap by Suite",
        ylabel="Coop after prior coop - coop after prior defect",
        output_path=FIGURES_DIR / "fig1_main_conditionality_gap.png",
    )


def plot_persona_avg_cooperation() -> None:
    rows = _read_csv(PROCESSED_DIR / "persona_robustness_drop.csv")
    _grouped_bar(
        rows=rows,
        x_key="persona",
        group_key="model",
        value_key="avg_cooperation",
        title="Persona Average Cooperation",
        ylabel="Average cooperation rate",
        output_path=FIGURES_DIR / "fig2_persona_avg_cooperation.png",
    )


def plot_persona_robustness_drop() -> None:
    rows = _read_csv(PROCESSED_DIR / "persona_robustness_drop.csv")
    _grouped_bar(
        rows=rows,
        x_key="persona",
        group_key="model",
        value_key="robustness_drop",
        title="Persona Robustness Drop",
        ylabel="Neutral avg cooperation - persona avg cooperation",
        output_path=FIGURES_DIR / "fig3_persona_robustness_drop.png",
    )


def run_plots() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_main_conditionality_gap()
    plot_persona_avg_cooperation()
    plot_persona_robustness_drop()


def main() -> None:
    run_plots()


if __name__ == "__main__":
    main()
