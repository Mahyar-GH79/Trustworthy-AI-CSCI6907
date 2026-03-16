import os
import json
import argparse
import matplotlib.pyplot as plt


def configure_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["figure.dpi"] = 160


def load_loose_metrics(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expect structure produced by nim_trustworthiness_v2.py
    loose = data["results"]["loose"]
    met_random = loose["random"]["metrics"]
    met_optimal = loose["optimal"]["metrics"]

    return met_random, met_optimal


def plot_win_rate(met_random, met_optimal, out_dir: str):
    configure_plot_style()

    labels = ["vs Random", "vs Optimal"]
    values = [met_random["win_rate"], met_optimal["win_rate"]]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(labels, values, edgecolor="black", linewidth=0.8)

    ax.set_title("Win Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("Win rate", fontsize=12)
    ax.set_ylim(0, 1.0)

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "win_rate.png"))
    plt.close(fig)


def plot_cheat_rate(met_random, met_optimal, out_dir: str):
    configure_plot_style()

    labels = ["vs Random", "vs Optimal"]

    frac_any = [
        met_random["games_with_any_violation_frac"],
        met_optimal["games_with_any_violation_frac"],
    ]
    per_turn = [
        met_random["violations_per_llm_turn"],
        met_optimal["violations_per_llm_turn"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    # Panel A: fraction of games with any violation
    ax = axes[0]
    bars = ax.bar(labels, frac_any, edgecolor="black", linewidth=0.8)
    ax.set_title("Cheat Rate: Games with Any Violation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Fraction of games", fontsize=12)
    ax.set_ylim(0, 1.0)
    for b, v in zip(bars, frac_any):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # Panel B: violations per LLM turn
    ax = axes[1]
    bars = ax.bar(labels, per_turn, edgecolor="black", linewidth=0.8)
    ax.set_title("Cheat Rate: Violations per LLM Turn", fontsize=13, fontweight="bold")
    ax.set_ylabel("Violations per turn", fontsize=12)
    ymax = max(0.05, max(per_turn) * 1.35)
    ax.set_ylim(0, ymax)
    for b, v in zip(bars, per_turn):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02 * ymax,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cheat_rate.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        type=str,
        default="nim_experiment_log_v2.json",
        help="Path to nim_experiment_log_v2.json",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="loose_only_figures",
        help="Output directory for figures",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    met_random, met_optimal = load_loose_metrics(args.json)

    plot_win_rate(met_random, met_optimal, args.out)
    plot_cheat_rate(met_random, met_optimal, args.out)

    print(f"Saved figures to: {args.out}")
    print("Created: win_rate.png and cheat_rate.png")


if __name__ == "__main__":
    main()
