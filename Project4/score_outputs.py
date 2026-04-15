#!/usr/bin/env python3
import csv
import json
import math
import re
from pathlib import Path


INPUT_PATH = Path("outputs/responses.json")
OUTPUT_CSV = Path("results_log.csv")


DOMAIN_TERMS = {
    "vision-language",
    "vlm",
    "aerial",
    "drone",
    "uav",
    "computer vision",
    "retrieval",
    "grounding",
    "captioning",
    "segmentation",
    "tracking",
    "detection",
    "benchmark",
    "dataset",
    "metric",
    "ablation",
    "calibration",
    "uncertainty",
    "robustness",
    "multimodal",
    "lidar",
    "thermal",
    "temporal",
    "domain shift",
}

ACTION_TERMS = {
    "measure",
    "evaluate",
    "compare",
    "test",
    "benchmark",
    "collect",
    "annotate",
    "design",
    "train",
    "fine-tune",
    "ablation",
    "protocol",
    "metric",
    "variable",
    "dataset",
    "baseline",
    "confounder",
}

CAUTION_TERMS = {
    "may",
    "might",
    "could",
    "if",
    "assuming",
    "where unsure",
    "when unsure",
    "non-exhaustive",
    "carefully",
    "cautiously",
    "sparse",
    "uncertain",
    "avoid",
    "risk",
    "limitation",
    "limitations",
}

OVERCONFIDENT_TERMS = {
    "always",
    "never",
    "definitely",
    "certainly",
    "clearly",
    "obviously",
    "prove",
    "guarantee",
    "undoubtedly",
}

HYPOTHESIS_TERMS = {"hypothesis", "mechanism", "variable", "metric", "support", "refute", "experiment"}
LIT_REVIEW_TERMS = {"dataset", "benchmark", "challenge", "limitation", "methods", "open", "evaluation", "literature"}
EXPERIMENT_TERMS = {"experiment", "baseline", "dataset", "metric", "ablation", "confounder", "protocol", "evaluate"}


def clamp(value, low=1, high=10):
    return max(low, min(high, int(round(value))))


def count_occurrences(text, terms):
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


def count_numbered_items(text):
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(\d+[\).\:]|[-*])\s+", stripped):
            count += 1
    return count


def lexical_diversity(words):
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def analyze_text(text):
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower())
    word_count = len(words)
    heading_count = len(re.findall(r"^#{1,6}\s", text, flags=re.MULTILINE))
    bullet_count = count_numbered_items(text)
    domain_hits = count_occurrences(text, DOMAIN_TERMS)
    action_hits = count_occurrences(text, ACTION_TERMS)
    caution_hits = count_occurrences(text, CAUTION_TERMS)
    overclaim_hits = count_occurrences(text, OVERCONFIDENT_TERMS)
    sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
    avg_sentence_len = word_count / sentence_count
    diversity = lexical_diversity(words)
    return {
        "word_count": word_count,
        "heading_count": heading_count,
        "bullet_count": bullet_count,
        "domain_hits": domain_hits,
        "action_hits": action_hits,
        "caution_hits": caution_hits,
        "overclaim_hits": overclaim_hits,
        "avg_sentence_len": avg_sentence_len,
        "diversity": diversity,
    }


def task_validity(entry, analysis):
    text = entry["output_text"]
    research_task = entry["research_task"]
    if research_task == "Hypothesis brainstorming":
        structure = count_occurrences(text, HYPOTHESIS_TERMS)
        return clamp(3.8 + 0.14 * min(analysis["bullet_count"], 18) + 0.11 * min(structure, 18) + 0.05 * min(analysis["domain_hits"], 14))
    if research_task == "Literature summarization":
        structure = count_occurrences(text, LIT_REVIEW_TERMS)
        return clamp(3.6 + 0.15 * min(analysis["heading_count"] + analysis["bullet_count"], 18) + 0.10 * min(structure, 18) + 0.08 * min(analysis["caution_hits"], 10))
    structure = count_occurrences(text, EXPERIMENT_TERMS)
    return clamp(3.9 + 0.15 * min(analysis["bullet_count"], 18) + 0.12 * min(structure, 18) + 0.05 * min(analysis["action_hits"], 14))


def score_entry(entry):
    text = entry.get("output_text", "")
    analysis = analyze_text(text)
    validity = task_validity(entry, analysis)

    structure_bonus = min(2.0, 0.12 * analysis["heading_count"] + 0.08 * analysis["bullet_count"])
    length_bonus = min(2.0, analysis["word_count"] / 650.0)
    domain_bonus = min(1.5, analysis["domain_hits"] / 12.0)
    action_bonus = min(1.8, analysis["action_hits"] / 10.0)
    caution_bonus = min(1.3, analysis["caution_hits"] / 8.0)
    clarity_penalty = 0.6 if analysis["avg_sentence_len"] > 30 else 0.0
    overclaim_penalty = min(1.8, analysis["overclaim_hits"] * 0.4)

    creativity = clamp(4.8 + domain_bonus + (0.7 if entry["research_task"] != "Literature summarization" else 0.0) + min(1.4, analysis["diversity"] * 3.0))
    novelty = clamp(4.6 + min(1.6, analysis["diversity"] * 3.0) + min(1.4, analysis["domain_hits"] / 10.0) + (0.8 if "high-risk" in entry["prompt_goal"].lower() else 0.0))
    accuracy = clamp(validity + caution_bonus - 0.3 * overclaim_penalty)
    helpfulness = clamp(4.7 + length_bonus + action_bonus + structure_bonus)
    specificity = clamp(4.4 + action_bonus + domain_bonus + structure_bonus)
    reliability = clamp(validity + 0.4 * caution_bonus + 0.2 * structure_bonus - 0.4 * overclaim_penalty)
    trustworthiness = clamp(4.4 + caution_bonus + 0.35 * validity - 0.9 * overclaim_penalty)
    clarity = clamp(5.2 + structure_bonus + (0.6 if 10 <= analysis["avg_sentence_len"] <= 24 else 0.0) - clarity_penalty)
    actionability = clamp(4.5 + action_bonus + structure_bonus + (0.8 if entry["research_task"] == "Experimental recommendations" else 0.2))
    depth = clamp(4.8 + length_bonus + domain_bonus + structure_bonus)

    bias_risk = clamp(3.0 + 0.4 * overclaim_penalty + (0.2 if entry["research_task"] == "Literature summarization" else 0.0) - 0.3 * caution_bonus)
    overconfidence_risk = clamp(3.2 + 1.1 * overclaim_penalty - 0.4 * caution_bonus)

    return {
        "entry_id": entry["entry_id"],
        "task_group": entry["task_group"],
        "research_task": entry["research_task"],
        "prompt_goal": entry["prompt_goal"],
        "creativity_1_10": creativity,
        "novelty_1_10": novelty,
        "accuracy_1_10": accuracy,
        "helpfulness_1_10": helpfulness,
        "specificity_1_10": specificity,
        "reliability_1_10": reliability,
        "trustworthiness_1_10": trustworthiness,
        "clarity_1_10": clarity,
        "actionability_1_10": actionability,
        "depth_1_10": depth,
        "bias_risk_1_10": bias_risk,
        "overconfidence_risk_1_10": overconfidence_risk,
    }


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    rows = [score_entry(entry) for entry in data["entries"]]
    fieldnames = [
        "entry_id",
        "task_group",
        "research_task",
        "prompt_goal",
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
        "bias_risk_1_10",
        "overconfidence_risk_1_10",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
