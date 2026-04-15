#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "results_log.csv"
FIG_DIR = ROOT / "figures"

POSITIVE_METRICS = [
    "creativity_1_10",
    "novelty_1_10",
    "accuracy_1_10",
    "helpfulness_1_10",
    "specificity_1_10",
    "reliability_1_10",
    "trustworthiness_1_10",
    "clarity_1_10",
    "actionability_1_10",
    "depth_1_10",
]
RISK_METRICS = ["bias_risk_1_10", "overconfidence_risk_1_10"]

TASK_ORDER = [
    "Hypothesis brainstorming",
    "Literature summarization",
    "Experimental recommendations",
]
TASK_SHORT = {
    "Hypothesis brainstorming": "Hypothesis",
    "Literature summarization": "Literature",
    "Experimental recommendations": "Experiment",
}
METRIC_LABELS = {
    "creativity_1_10": "Creativity",
    "novelty_1_10": "Novelty",
    "accuracy_1_10": "Accuracy",
    "helpfulness_1_10": "Helpfulness",
    "specificity_1_10": "Specificity",
    "reliability_1_10": "Reliability",
    "trustworthiness_1_10": "Trustworthiness",
    "clarity_1_10": "Clarity",
    "actionability_1_10": "Actionability",
    "depth_1_10": "Depth",
}
COLORS = {
    "Hypothesis brainstorming": "#355C7D",
    "Literature summarization": "#6C7A89",
    "Experimental recommendations": "#C06C45",
}


def setup_style():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#C8CDD3",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": "#E7EAEE",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data():
    df = pd.read_csv(INPUT_CSV)
    for column in POSITIVE_METRICS + RISK_METRICS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["research_task"] = pd.Categorical(df["research_task"], categories=TASK_ORDER, ordered=True)
    df["quality_mean"] = df[POSITIVE_METRICS].mean(axis=1)
    df["risk_mean"] = df[RISK_METRICS].mean(axis=1)
    df["composite_score"] = df["quality_mean"] - 0.35 * df["risk_mean"]
    return df


def savefig(fig, stem):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def plot_main_figure(df):
    summary = (
        df.groupby("research_task", observed=True)[["quality_mean", "risk_mean"]]
        .mean()
        .loc[TASK_ORDER]
    )
    summary.index = [TASK_SHORT[idx] for idx in summary.index]

    heatmap_df = (
        df.groupby("research_task", observed=True)[POSITIVE_METRICS]
        .mean()
        .loc[TASK_ORDER]
    )
    heatmap_df.index = [TASK_SHORT[idx] for idx in heatmap_df.index]
    heatmap_df = heatmap_df.rename(columns=METRIC_LABELS)

    fig = plt.figure(figsize=(12.8, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.45], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(summary))
    width = 0.34
    ax1.bar(
        x - width / 2,
        summary["quality_mean"],
        width,
        color="#446A8A",
        label="Quality",
        linewidth=0,
    )
    ax1.bar(
        x + width / 2,
        summary["risk_mean"],
        width,
        color="#D28B5A",
        label="Risk",
        linewidth=0,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary.index)
    ax1.set_ylim(0, 10)
    ax1.set_ylabel("Average score")
    ax1.set_title("Task-Level Summary")
    ax1.grid(axis="y")
    ax1.legend(loc="upper right")
    add_panel_label(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap=sns.light_palette("#355C7D", as_cmap=True),
        linewidths=0.7,
        linecolor="white",
        vmin=5,
        vmax=10,
        cbar_kws={"shrink": 0.82, "label": "Score"},
        ax=ax2,
    )
    ax2.set_title("Fine-Grained Metric Breakdown")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    add_panel_label(ax2, "B")

    fig.suptitle("Evaluation of GPT-Assisted Scientific Research Tasks", y=1.02, fontsize=15, fontweight="bold")
    savefig(fig, "icml_main_figure")


def plot_prompt_ranking(df):
    ranked = df.sort_values("composite_score", ascending=False)
    top = ranked.head(6).copy()
    top["label"] = top["entry_id"].astype(str) + ". " + top["prompt_goal"]

    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    sns.barplot(
        data=top,
        x="composite_score",
        y="label",
        hue="research_task",
        dodge=False,
        palette=COLORS,
        ax=ax,
    )
    ax.set_title("Top-Scoring Prompts")
    ax.set_xlabel("Composite score")
    ax.set_ylabel("")
    ax.grid(axis="x")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [TASK_SHORT[label] for label in labels],
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(bottom=0.23)
    savefig(fig, "icml_top_prompts")


def plot_quality_risk_frontier(df):
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    sns.scatterplot(
        data=df,
        x="risk_mean",
        y="quality_mean",
        hue="research_task",
        palette=COLORS,
        s=72,
        edgecolor="white",
        linewidth=0.7,
        ax=ax,
    )
    group_means = (
        df.groupby("research_task", observed=True)[["risk_mean", "quality_mean"]]
        .mean()
        .loc[TASK_ORDER]
    )
    for task, row in group_means.iterrows():
        ax.scatter(
            row["risk_mean"],
            row["quality_mean"],
            s=180,
            marker="D",
            color=COLORS[task],
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
        ax.text(
            row["risk_mean"] + 0.03,
            row["quality_mean"] + 0.03,
            TASK_SHORT[task],
            fontsize=8.5,
        )
    ax.set_xlim(1.8, 4.2)
    ax.set_ylim(6.5, 9.4)
    ax.set_xlabel("Average risk score")
    ax.set_ylabel("Average quality score")
    ax.set_title("Quality vs. Risk by Prompt")
    ax.grid(True, alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [TASK_SHORT[label] for label in labels], title="", loc="lower right")
    savefig(fig, "quality_risk_frontier")


def plot_task_metric_barplot(df):
    summary = (
        df.groupby("research_task", observed=True)[POSITIVE_METRICS]
        .mean()
        .rename(columns=METRIC_LABELS)
        .loc[TASK_ORDER]
        .reset_index()
    )
    long_df = summary.melt(id_vars="research_task", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    sns.barplot(
        data=long_df,
        x="Metric",
        y="Score",
        hue="research_task",
        palette=COLORS,
        ax=ax,
    )
    ax.set_ylim(0, 10)
    ax.set_xlabel("")
    ax.set_ylabel("Average score")
    ax.set_title("Metric Comparison Across Task Types")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [TASK_SHORT[label] for label in labels], title="", ncol=3, loc="upper center")
    savefig(fig, "task_metric_barplot")


def main():
    setup_style()
    df = load_data()
    plot_main_figure(df)
    plot_prompt_ranking(df)
    plot_quality_risk_frontier(df)
    plot_task_metric_barplot(df)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
