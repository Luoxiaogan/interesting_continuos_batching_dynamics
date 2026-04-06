#!/usr/bin/env python3
"""
Redraw coupling comparison plots with a TeX-like publication style.

This script reads an existing experiment output directory containing
`trajectory.csv` and `config.json`, then recreates the comparison figure
using a Computer Modern-like serif style that matches the paper figures
more closely than the default Matplotlib output.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def load_trajectory(csv_path: Path) -> list[dict]:
    """Load trajectory rows from CSV and cast numeric fields."""
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "batch":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def cmu_rc_params() -> dict:
    """
    Return TeX-like style settings.

    The paper figures are standalone TikZ documents with default LaTeX fonts,
    so their effective base family is Computer Modern Roman. Matplotlib ships
    `cmr10`, which is the closest available font in this environment.
    """
    return {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "axes.titlesize": 9,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
    }


def register_cmu_fonts() -> None:
    """Register local CMU font files so Matplotlib can resolve `CMU Serif` reliably."""
    font_dir = Path("/Users/moonshot/Library/Fonts")
    for pattern in ("cmunrm.ttf", "cmunbx.ttf", "cmunbi.ttf", "cmunti.ttf"):
        font_path = font_dir / pattern
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def redraw_comparison(trajectory: list[dict], config: dict, png_path: Path, pdf_path: Path) -> None:
    """Redraw the comparison plot in a TeX-like publication style."""
    batches = [row["batch"] for row in trajectory]
    theory_adm = [row["theory_admission"] for row in trajectory]
    sim_adm = [row["sim_admission"] for row in trajectory]
    sim_evic = [row["sim_eviction_total"] for row in trajectory]
    evic_negative = [-value for value in sim_evic]

    with plt.rc_context(cmu_rc_params()):
        fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)

        ax.bar(
            batches,
            evic_negative,
            width=0.86,
            color="#d95f5f",
            alpha=0.28,
            edgecolor="#c74848",
            linewidth=0.4,
            label="simulation eviction",
            zorder=1,
        )
        ax.plot(
            batches,
            theory_adm,
            color="#2b6cb0",
            linewidth=1.15,
            marker="o",
            markersize=2.4,
            markerfacecolor="white",
            markeredgewidth=0.55,
            label="theory admission",
            zorder=3,
        )
        ax.plot(
            batches,
            sim_adm,
            color="#2f855a",
            linewidth=1.15,
            marker="s",
            markersize=2.3,
            markerfacecolor="white",
            markeredgewidth=0.55,
            label="simulation admission",
            zorder=4,
        )

        ax.axhline(0.0, color="black", linewidth=0.55, alpha=0.8, zorder=2)
        ax.grid(True, axis="y", color="0.85", linewidth=0.45, alpha=0.8)

        diverge_batches = [
            batch
            for batch, theory_value, eviction_value in zip(batches, theory_adm, sim_evic)
            if theory_value < 0.0 or eviction_value > 1e-2
        ]
        if diverge_batches:
            start = min(diverge_batches) - 0.5
            end = max(diverge_batches) + 0.5
            ax.axvspan(start, end, color="#f6e58d", alpha=0.18, zorder=0)
            ax.text(
                start + 0.8,
                ax.get_ylim()[1] * 0.9,
                "divergence region",
                fontsize=7,
                color="0.35",
                va="top",
            )

        ax.set_xlabel("batch index")
        ax.set_ylabel("requests")
        ax.set_title("Theory vs. simulation", fontweight="bold", pad=4)
        ax.text(
            0.99,
            0.97,
            rf"$l_0={config['l0']},\ l_A={config['l_A']},\ l_B={config['l_B']}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="0.35",
        )
        ax.legend(
            loc="lower right",
            frameon=True,
            fancybox=False,
            edgecolor="0.75",
            framealpha=0.95,
            borderpad=0.3,
            handlelength=1.6,
        )
        ax.margins(x=0.01)

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    register_cmu_fonts()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment output directory containing trajectory.csv and config.json.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    trajectory_path = output_dir / "trajectory.csv"
    config_path = output_dir / "config.json"

    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {trajectory_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    trajectory = load_trajectory(trajectory_path)
    with config_path.open() as f:
        config = json.load(f)

    png_path = output_dir / "comparison_cmu.png"
    pdf_path = output_dir / "comparison_cmu.pdf"
    redraw_comparison(trajectory, config, png_path, pdf_path)

    print(f"Saved publication-style comparison to {png_path}")
    print(f"Saved publication-style comparison to {pdf_path}")


if __name__ == "__main__":
    main()
