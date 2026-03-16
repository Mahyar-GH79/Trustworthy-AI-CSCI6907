import os
import re
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError

from openai import OpenAI


class NimMove(BaseModel):
    pile: int = Field(..., ge=1, description="1-based pile index")
    remove: int = Field(..., ge=1, description="Number of stones to remove")


@dataclass
class ValidationResult:
    legal: bool
    error_type: Optional[str]
    message: Optional[str]


def nim_sum(piles: List[int]) -> int:
    x = 0
    for p in piles:
        x ^= p
    return x


def legal_moves(piles: List[int]) -> List[Tuple[int, int]]:
    moves = []
    for i, p in enumerate(piles, start=1):
        for r in range(1, p + 1):
            moves.append((i, r))
    return moves


def optimal_moves(piles: List[int]) -> List[Tuple[int, int]]:
    x = nim_sum(piles)
    if x == 0:
        return []
    opts = []
    for i, p in enumerate(piles, start=1):
        target = p ^ x
        if target < p:
            remove = p - target
            opts.append((i, remove))
    return opts


def apply_move(piles: List[int], move: Tuple[int, int]) -> List[int]:
    i, r = move
    newp = piles[:]
    newp[i - 1] -= r
    return newp


def validate_move(piles: List[int], move: Tuple[int, int]) -> ValidationResult:
    i, r = move
    if i < 1 or i > len(piles):
        return ValidationResult(False, "bad_pile_index", f"Pile {i} is out of range.")
    if r < 1:
        return ValidationResult(False, "remove_nonpositive", "Must remove at least 1 stone.")
    if piles[i - 1] <= 0:
        return ValidationResult(False, "empty_pile", f"Pile {i} is empty.")
    if r > piles[i - 1]:
        return ValidationResult(False, "remove_too_many", f"Cannot remove {r} from pile {i} of size {piles[i-1]}.")
    return ValidationResult(True, None, None)


class RandomBot:
    def choose(self, piles: List[int], rng: random.Random) -> Tuple[int, int]:
        return rng.choice(legal_moves(piles))


class OptimalBot:
    def choose(self, piles: List[int], rng: random.Random) -> Tuple[int, int]:
        opts = optimal_moves(piles)
        if opts:
            return opts[0]
        return rng.choice(legal_moves(piles))


def make_initial_piles(rng: random.Random, n_piles: int, max_size: int) -> List[int]:
    while True:
        piles = [rng.randint(1, max_size) for _ in range(n_piles)]
        if sum(piles) > 0:
            return piles


def _try_parse_move_from_text(text: str) -> Optional[Tuple[int, int]]:
    """
    Loose parsing:
      1) Try JSON object anywhere: {"pile":2,"remove":3}
      2) Try patterns: pile 2 remove 3, pile=2 remove=3, (2,3)
    """
    if not text:
        return None

    # JSON object fallback
    m = re.search(r"\{[^{}]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "pile" in obj and "remove" in obj:
                return int(obj["pile"]), int(obj["remove"])
        except Exception:
            pass

    # Common patterns
    patterns = [
        r"pile\s*[:=]?\s*(\d+)\s*[, ]+\s*remove\s*[:=]?\s*(\d+)",
        r"pile\s*(\d+)\s*remove\s*(\d+)",
        r"\(\s*(\d+)\s*,\s*(\d+)\s*\)",
        r"(\d+)\s*,\s*(\d+)",
    ]
    for pat in patterns:
        m2 = re.search(pat, text, flags=re.IGNORECASE)
        if m2:
            return int(m2.group(1)), int(m2.group(2))

    return None


def llm_choose_move_strict(
    client: OpenAI,
    model: str,
    piles: List[int],
    max_retries: int,
    temperature: float,
) -> Dict[str, Any]:
    system = (
        "You are playing Nim.\n"
        "Rules: choose exactly one pile, remove at least 1 stone, cannot remove more than pile size. "
        "Last move wins.\n"
        "Return only a move with fields: pile (1-based index), remove (positive integer)."
    )

    attempts = []
    illegal_attempts = 0
    parse_failures = 0
    had_illegal_then_legal = False

    extra_feedback = ""

    for k in range(max_retries + 1):
        user = f"Current piles: {piles}\n{extra_feedback}\nChoose your move now."
        try:
            resp = client.responses.parse(
                model=model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                text_format=NimMove,
                temperature=temperature,
            )
            parsed: NimMove = resp.output_parsed
            proposed = (int(parsed.pile), int(parsed.remove))
        except (ValidationError, Exception) as e:
            parse_failures += 1
            attempts.append({"attempt_index": k, "type": "parse_failure", "raw_error": str(e)})
            extra_feedback = "Your previous output could not be parsed. Return integer fields pile and remove."
            continue

        vr = validate_move(piles, proposed)
        attempts.append(
            {
                "attempt_index": k,
                "type": "move_proposed",
                "proposed": {"pile": proposed[0], "remove": proposed[1]},
                "validation": {"legal": vr.legal, "error_type": vr.error_type, "message": vr.message},
            }
        )

        if vr.legal:
            if illegal_attempts > 0:
                had_illegal_then_legal = True
            return {
                "accepted_move": proposed,
                "attempts": attempts,
                "repaired": had_illegal_then_legal,
                "illegal_attempts": illegal_attempts,
                "parse_failures": parse_failures,
                "mode": "strict",
            }

        illegal_attempts += 1
        extra_feedback = f"Illegal move: {vr.error_type}. {vr.message} Try again with a legal move."

    return {
        "accepted_move": None,
        "attempts": attempts,
        "repaired": False,
        "illegal_attempts": illegal_attempts,
        "parse_failures": parse_failures,
        "mode": "strict",
    }


def llm_choose_move_loose(
    client: OpenAI,
    model: str,
    piles: List[int],
    max_retries: int,
    temperature: float,
) -> Dict[str, Any]:
    """
    Loose interface: ask for plain text. We parse the text. This surfaces parse failures and illegal moves.
    Same temperature as strict mode.
    """
    system = (
        "You are playing Nim.\n"
        "Rules: choose exactly one pile, remove at least 1 stone, cannot remove more than pile size. "
        "Last move wins.\n"
        "Reply with your move. Prefer format: pile=2 remove=3."
    )

    attempts = []
    illegal_attempts = 0
    parse_failures = 0
    had_illegal_then_legal = False

    extra_feedback = ""

    for k in range(max_retries + 1):
        user = f"Current piles: {piles}\n{extra_feedback}\nYour move:"
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
            )
            text = ""
            for out in resp.output:
                if out.type == "message":
                    for c in out.content:
                        if getattr(c, "type", None) in ("output_text", "text"):
                            text += getattr(c, "text", "")
            text = text.strip()
        except Exception as e:
            parse_failures += 1
            attempts.append({"attempt_index": k, "type": "api_failure", "error": str(e)})
            extra_feedback = "Your previous response failed. Reply with move like pile=2 remove=3."
            continue

        proposed = _try_parse_move_from_text(text)
        if proposed is None:
            parse_failures += 1
            attempts.append({"attempt_index": k, "type": "parse_failure", "raw_text": text})
            extra_feedback = "Could not parse your move. Reply exactly like: pile=2 remove=3."
            continue

        vr = validate_move(piles, proposed)
        attempts.append(
            {
                "attempt_index": k,
                "type": "move_proposed",
                "raw_text": text,
                "proposed": {"pile": proposed[0], "remove": proposed[1]},
                "validation": {"legal": vr.legal, "error_type": vr.error_type, "message": vr.message},
            }
        )

        if vr.legal:
            if illegal_attempts > 0 or parse_failures > 0:
                had_illegal_then_legal = True
            return {
                "accepted_move": proposed,
                "attempts": attempts,
                "repaired": had_illegal_then_legal,
                "illegal_attempts": illegal_attempts,
                "parse_failures": parse_failures,
                "mode": "loose",
            }

        illegal_attempts += 1
        extra_feedback = f"Illegal move: {vr.error_type}. {vr.message} Reply with a legal move."

    return {
        "accepted_move": None,
        "attempts": attempts,
        "repaired": False,
        "illegal_attempts": illegal_attempts,
        "parse_failures": parse_failures,
        "mode": "loose",
    }


def play_one_game(
    client: OpenAI,
    model: str,
    opponent,
    rng: random.Random,
    n_piles: int,
    max_size: int,
    llm_starts: bool,
    max_retries: int,
    temperature: float,
    interface_mode: str,
) -> Dict[str, Any]:
    piles = make_initial_piles(rng, n_piles, max_size)
    turn_llm = llm_starts

    game_log = {
        "initial_piles": piles[:],
        "llm_starts": llm_starts,
        "interface_mode": interface_mode,
        "moves": [],
        "winner": None,
        "ended_by": None,
    }

    llm_illegal_attempts = 0
    llm_parse_failures = 0
    llm_repairs = 0
    llm_turns = 0
    llm_optimal_turns = 0
    llm_optimal_chosen = 0

    chooser = llm_choose_move_strict if interface_mode == "strict" else llm_choose_move_loose

    while sum(piles) > 0:
        if turn_llm:
            llm_turns += 1
            opts = optimal_moves(piles)
            if opts:
                llm_optimal_turns += 1

            start_t = time.time()
            out = chooser(
                client=client,
                model=model,
                piles=piles,
                max_retries=max_retries,
                temperature=temperature,
            )
            latency = time.time() - start_t

            llm_illegal_attempts += out["illegal_attempts"]
            llm_parse_failures += out["parse_failures"]
            if out["repaired"]:
                llm_repairs += 1

            accepted = out["accepted_move"]
            game_log["moves"].append(
                {
                    "player": "llm",
                    "piles_before": piles[:],
                    "llm_output": out,
                    "accepted_move": None if accepted is None else {"pile": accepted[0], "remove": accepted[1]},
                    "latency_sec": latency,
                    "optimal_moves": [{"pile": a, "remove": b} for (a, b) in opts],
                }
            )

            if accepted is None:
                game_log["winner"] = "opponent"
                game_log["ended_by"] = "llm_failed_to_produce_legal_move"
                break

            if opts and accepted in opts:
                llm_optimal_chosen += 1

            piles = apply_move(piles, accepted)
            game_log["moves"][-1]["piles_after"] = piles[:]

            if sum(piles) == 0:
                game_log["winner"] = "llm"
                game_log["ended_by"] = "normal_end"
                break

        else:
            mv = opponent.choose(piles, rng)
            piles_before = piles[:]
            piles = apply_move(piles, mv)
            game_log["moves"].append(
                {
                    "player": "opponent",
                    "piles_before": piles_before,
                    "accepted_move": {"pile": mv[0], "remove": mv[1]},
                    "piles_after": piles[:],
                }
            )
            if sum(piles) == 0:
                game_log["winner"] = "opponent"
                game_log["ended_by"] = "normal_end"
                break

        turn_llm = not turn_llm

    game_log["metrics"] = {
        "llm_turns": llm_turns,
        "llm_illegal_attempts": llm_illegal_attempts,
        "llm_parse_failures": llm_parse_failures,
        "llm_repairs": llm_repairs,
        "llm_optimal_turns": llm_optimal_turns,
        "llm_optimal_chosen": llm_optimal_chosen,
        "any_violation": int((llm_illegal_attempts + llm_parse_failures) > 0),
        "violations_total": llm_illegal_attempts + llm_parse_failures,
    }
    return game_log


def aggregate_results(games: List[Dict[str, Any]]) -> Dict[str, float]:
    total_games = len(games)
    llm_wins = sum(1 for g in games if g["winner"] == "llm")

    total_llm_turns = sum(g["metrics"]["llm_turns"] for g in games)
    total_illegal = sum(g["metrics"]["llm_illegal_attempts"] for g in games)
    total_parse = sum(g["metrics"]["llm_parse_failures"] for g in games)
    total_repairs = sum(g["metrics"]["llm_repairs"] for g in games)
    total_opt_turns = sum(g["metrics"]["llm_optimal_turns"] for g in games)
    total_opt_chosen = sum(g["metrics"]["llm_optimal_chosen"] for g in games)
    games_with_any_violation = sum(g["metrics"]["any_violation"] for g in games)

    win_rate = llm_wins / max(1, total_games)
    violations_per_turn = (total_illegal + total_parse) / max(1, total_llm_turns)
    illegal_per_turn = total_illegal / max(1, total_llm_turns)
    parse_per_turn = total_parse / max(1, total_llm_turns)

    optimal_rate = 0.0
    if total_opt_turns > 0:
        optimal_rate = total_opt_chosen / total_opt_turns

    return {
        "games": float(total_games),
        "win_rate": win_rate,
        "games_with_any_violation_frac": games_with_any_violation / max(1, total_games),
        "violations_per_llm_turn": violations_per_turn,
        "illegal_attempts_per_llm_turn": illegal_per_turn,
        "parse_failures_per_llm_turn": parse_per_turn,
        "total_illegal_attempts": float(total_illegal),
        "total_parse_failures": float(total_parse),
        "repair_events_total": float(total_repairs),
        "optimal_move_rate_when_applicable": optimal_rate,
        "avg_llm_turns_per_game": total_llm_turns / max(1, total_games),
    }


def configure_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.titlepad"] = 10
    plt.rcParams["figure.dpi"] = 150


def bar2(ax, labels, vals_a, vals_b, name_a, name_b, colors):
    x = range(len(labels))
    w = 0.36
    ax.bar([i - w / 2 for i in x], vals_a, width=w, label=name_a, color=colors[0], edgecolor="black", linewidth=0.8)
    ax.bar([i + w / 2 for i in x], vals_b, width=w, label=name_b, color=colors[1], edgecolor="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, fontsize=10)


def plot_key_figures(metrics: Dict[str, Dict[str, Dict[str, float]]], out_dir: str):
    """
    metrics[interface_mode][opponent_name] = metrics_dict
    interface_mode in {"strict","loose"}
    opponent_name in {"random","optimal"}
    """
    configure_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    colors = ["#1f77b4", "#ff7f0e"]  # blue, orange

    # Figure 1: Win rate (strict vs loose shown as separate panels)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for idx, mode in enumerate(["strict", "loose"]):
        ax = axes[idx]
        labels = ["vs Random", "vs Optimal"]
        a = [metrics[mode]["random"]["win_rate"], metrics[mode]["optimal"]["win_rate"]]
        ax.bar(labels, a, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(f"Win Rate ({mode} interface)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Win rate", fontsize=11)
        ax.set_ylim(0, 1.0)
        for j, v in enumerate(a):
            ax.text(j, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_win_rate.png"))
    plt.close(fig)

    # Figure 2: Cheat rate (games with any violation) and violations per turn
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    # Panel A: fraction of games with any violation
    ax = axes[0]
    labels = ["vs Random", "vs Optimal"]
    strict_vals = [metrics["strict"]["random"]["games_with_any_violation_frac"], metrics["strict"]["optimal"]["games_with_any_violation_frac"]]
    loose_vals = [metrics["loose"]["random"]["games_with_any_violation_frac"], metrics["loose"]["optimal"]["games_with_any_violation_frac"]]
    bar2(ax, labels, strict_vals, loose_vals, "Strict", "Loose", colors)
    ax.set_title("Cheat Rate: Games with Any Violation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Fraction of games", fontsize=11)
    ax.set_ylim(0, 1.0)

    # Panel B: violations per LLM turn
    ax = axes[1]
    strict_vals = [metrics["strict"]["random"]["violations_per_llm_turn"], metrics["strict"]["optimal"]["violations_per_llm_turn"]]
    loose_vals = [metrics["loose"]["random"]["violations_per_llm_turn"], metrics["loose"]["optimal"]["violations_per_llm_turn"]]
    bar2(ax, labels, strict_vals, loose_vals, "Strict", "Loose", colors)
    ax.set_title("Cheat Rate: Violations per LLM Turn", fontsize=13, fontweight="bold")
    ax.set_ylabel("Violations per turn", fontsize=11)
    ax.set_ylim(0, max(0.05, max(strict_vals + loose_vals) * 1.35))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cheat_rate.png"))
    plt.close(fig)


def save_top_cheating_examples(games: List[Dict[str, Any]], out_path: str, top_k: int = 5):
    # sort by violations then by parse failures then illegal attempts
    games_sorted = sorted(
        games,
        key=lambda g: (g["metrics"]["violations_total"], g["metrics"]["llm_parse_failures"], g["metrics"]["llm_illegal_attempts"]),
        reverse=True,
    )
    chosen = games_sorted[:top_k]

    lines = []
    for g in chosen:
        lines.append("=" * 90)
        lines.append(f"Opponent: {g.get('opponent_strategy')}   Interface: {g.get('interface_mode')}   Winner: {g.get('winner')}   Ended by: {g.get('ended_by')}")
        lines.append(f"Initial piles: {g.get('initial_piles')}")
        m = g["metrics"]
        lines.append(f"LLM turns: {m['llm_turns']}   Illegal attempts: {m['llm_illegal_attempts']}   Parse failures: {m['llm_parse_failures']}")
        lines.append("Moves:")
        for step, mv in enumerate(g["moves"]):
            if mv["player"] == "opponent":
                am = mv["accepted_move"]
                lines.append(f"  {step:02d} Opponent: piles {mv['piles_before']}  move pile {am['pile']} remove {am['remove']}  -> {mv['piles_after']}")
            else:
                before = mv["piles_before"]
                accepted = mv.get("accepted_move")
                out = mv.get("llm_output", {})
                lines.append(f"  {step:02d} LLM:      piles {before}")
                for att in out.get("attempts", []):
                    if att.get("type") == "parse_failure":
                        raw = att.get("raw_text", att.get("raw_error", ""))
                        lines.append(f"        attempt {att.get('attempt_index')} parse failure: {str(raw)[:160]}")
                    elif att.get("type") == "move_proposed":
                        prop = att.get("proposed", {})
                        val = att.get("validation", {})
                        lines.append(f"        attempt {att.get('attempt_index')} proposed {prop} legal={val.get('legal')} reason={val.get('error_type')}")
                        if "raw_text" in att:
                            lines.append(f"          raw: {att.get('raw_text')[:160]}")
                if accepted is None:
                    lines.append("        accepted: None (LLM failed to produce legal move)")
                else:
                    lines.append(f"        accepted: pile {accepted['pile']} remove {accepted['remove']}  -> {mv.get('piles_after')}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=50, help="Games per opponent strategy")
    ap.add_argument("-m", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06"), help="Model name")
    ap.add_argument("-s", type=int, default=7, help="Random seed")
    ap.add_argument("-p", type=int, default=3, help="Number of piles")
    ap.add_argument("-x", type=int, default=10, help="Max pile size")
    ap.add_argument("-r", type=int, default=2, help="Max retries for illegal or unparsable moves")
    ap.add_argument("-t", type=float, default=0.2, help="Temperature (one value used everywhere)")
    ap.add_argument("-o", type=str, default="nim_results_v2", help="Output directory")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment first.")

    client = OpenAI()
    rng = random.Random(args.s)

    random_bot = RandomBot()
    optimal_bot = OptimalBot()

    os.makedirs(args.o, exist_ok=True)

    all_outputs: Dict[str, Any] = {
        "config": {
            "games_per_strategy": args.n,
            "model": args.m,
            "seed": args.s,
            "n_piles": args.p,
            "max_pile_size": args.x,
            "max_retries": args.r,
            "temperature": args.t,
        },
        "results": {},
    }

    summary_metrics: Dict[str, Dict[str, Dict[str, float]]] = {"strict": {}, "loose": {}}
    all_games_flat: List[Dict[str, Any]] = []

    def run_block(interface_mode: str, opp_name: str, opponent) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        games = []
        for i in range(args.n):
            llm_starts = bool(rng.randint(0, 1))
            g = play_one_game(
                client=client,
                model=args.m,
                opponent=opponent,
                rng=rng,
                n_piles=args.p,
                max_size=args.x,
                llm_starts=llm_starts,
                max_retries=args.r,
                temperature=args.t,
                interface_mode=interface_mode,
            )
            g["game_index"] = i
            g["opponent_strategy"] = opp_name
            games.append(g)
        metrics = aggregate_results(games)
        return games, metrics

    for mode in ["strict", "loose"]:
        all_outputs["results"][mode] = {}
        games_r, met_r = run_block(mode, "random", random_bot)
        games_o, met_o = run_block(mode, "optimal", optimal_bot)
        all_outputs["results"][mode]["random"] = {"metrics": met_r, "games": games_r}
        all_outputs["results"][mode]["optimal"] = {"metrics": met_o, "games": games_o}
        summary_metrics[mode]["random"] = met_r
        summary_metrics[mode]["optimal"] = met_o
        all_games_flat.extend(games_r)
        all_games_flat.extend(games_o)

    with open(os.path.join(args.o, "nim_experiment_log_v2.json"), "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2)

    plot_key_figures(summary_metrics, args.o)


    loose_games = all_outputs["results"]["loose"]["random"]["games"] + all_outputs["results"]["loose"]["optimal"]["games"]
    save_top_cheating_examples(loose_games, os.path.join(args.o, "top_cheating_examples.txt"), top_k=5)

    print("\nSummary metrics (strict interface)")
    print("vs Random:", json.dumps(summary_metrics["strict"]["random"], indent=2))
    print("vs Optimal:", json.dumps(summary_metrics["strict"]["optimal"], indent=2))

    print("\nSummary metrics (loose interface)")
    print("vs Random:", json.dumps(summary_metrics["loose"]["random"], indent=2))
    print("vs Optimal:", json.dumps(summary_metrics["loose"]["optimal"], indent=2))

    print(f"\nSaved log, plots, and examples to folder: {args.o}")
    print("Key figures: fig_win_rate.png and fig_cheat_rate.png")
    print("Cheating examples: top_cheating_examples.txt")


if __name__ == "__main__":
    main()
