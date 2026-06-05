#!/usr/bin/env python3
"""
Generate thesis tables and plots from Priority 0 validation JSON results.

Usage:
    python3 generate_thesis_tables.py

Outputs:
    - thesis_tables.tex   : LaTeX tables for Chapter 5
    - scatter_plot.png    : Estimated vs Actual step time scatter
    - ranking_stats.json  : Numerical stats for inline text
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "thesis_outputs"


def load_results() -> List[Dict]:
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def group_by_world_size(results: List[Dict]) -> Dict[int, List[Dict]]:
    groups = {}
    for r in results:
        ws = r["plan"]["world_size"]
        groups.setdefault(ws, []).append(r)
    return groups


def compute_winner_accuracy(group: List[Dict]) -> bool:
    """True if the plan with lowest estimated time also has lowest actual time."""
    est_winner = min(group, key=lambda r: r["estimated_step_time_ms"])
    act_winner = min(group, key=lambda r: r["actual"]["avg_step_time_ms"])
    return est_winner["plan"] == act_winner["plan"]


def compute_pairwise_accuracy(group: List[Dict]) -> float:
    """Fraction of plan pairs where estimated ranking matches actual ranking."""
    n = len(group)
    if n < 2:
        return 1.0
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            est_i = group[i]["estimated_step_time_ms"]
            est_j = group[j]["estimated_step_time_ms"]
            act_i = group[i]["actual"]["avg_step_time_ms"]
            act_j = group[j]["actual"]["avg_step_time_ms"]
            if (est_i < est_j) == (act_i < act_j):
                correct += 1
            total += 1
    return correct / total if total > 0 else 1.0


def compute_spearman(group: List[Dict]) -> Tuple[float, float]:
    est = [r["estimated_step_time_ms"] for r in group]
    act = [r["actual"]["avg_step_time_ms"] for r in group]
    if len(est) < 2:
        return 1.0, 0.0
    rho, pval = stats.spearmanr(est, act)
    return rho, pval


def compute_mape(group: List[Dict]) -> float:
    """Mean absolute percentage error (relative to actual)."""
    errors = []
    for r in group:
        est = r["estimated_step_time_ms"]
        act = r["actual"]["avg_step_time_ms"]
        errors.append(abs(est - act) / act)
    return sum(errors) / len(errors)


def compute_ratio_stats(group: List[Dict]) -> Dict[str, float]:
    ratios = [r["estimated_step_time_ms"] / r["actual"]["avg_step_time_ms"] for r in group]
    return {
        "mean_ratio": sum(ratios) / len(ratios),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
    }


def generate_latex_table(groups: Dict[int, List[Dict]]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cost Model Ranking Accuracy by Cluster Size}")
    lines.append(r"\label{tab:ranking_accuracy}")
    lines.append(r"\begin{tabular}{c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"GPUs & Plans & Winner Acc. & Pairwise Acc. & Spearman $\rho$ & MAPE \\")
    lines.append(r"\midrule")

    overall_est = []
    overall_act = []
    for ws in sorted(groups.keys()):
        group = groups[ws]
        winner_acc = compute_winner_accuracy(group)
        pairwise_acc = compute_pairwise_accuracy(group)
        rho, _ = compute_spearman(group)
        mape = compute_mape(group)
        lines.append(
            f"{ws} & {len(group)} & "
            f"{'\checkmark' if winner_acc else '---'} & "
            f"{pairwise_acc*100:.0f}\\% & "
            f"{rho:.3f} & "
            f"{mape*100:.1f}\\% \\"
        )
        overall_est.extend([r["estimated_step_time_ms"] for r in group])
        overall_act.extend([r["actual"]["avg_step_time_ms"] for r in group])

    # Overall row
    overall_group = []
    for g in groups.values():
        overall_group.extend(g)
    winner_acc = compute_winner_accuracy(overall_group)
    pairwise_acc = compute_pairwise_accuracy(overall_group)
    rho, _ = compute_spearman(overall_group)
    mape = compute_mape(overall_group)
    lines.append(r"\midrule")
    lines.append(
        f"Overall & {len(overall_group)} & "
        f"{'\checkmark' if winner_acc else '---'} & "
        f"{pairwise_acc*100:.0f}\\% & "
        f"{rho:.3f} & "
        f"{mape*100:.1f}\\% \\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_detailed_table(groups: Dict[int, List[Dict]]) -> str:
    """Per-plan detailed table for 8-GPU cluster."""
    lines = []
    group = groups.get(8, [])
    if not group:
        return ""

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-Plan Cost Model Accuracy (8 GPUs, hidden=1024)}")
    lines.append(r"\label{tab:per_plan_8gpu}")
    lines.append(r"\begin{tabular}{c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Plan (pp,tp,dp) & $\hat{T}$ (ms) & $T_{actual}$ (ms) & Ratio & Rank$_e$ & Rank$_a$ & Error \\")
    lines.append(r"\midrule")

    sorted_by_actual = sorted(group, key=lambda r: r["actual"]["avg_step_time_ms"])
    for rank_a, r in enumerate(sorted_by_actual, 1):
        plan = r["plan"]
        est = r["estimated_step_time_ms"]
        act = r["actual"]["avg_step_time_ms"]
        ratio = est / act
        # estimated rank
        est_sorted = sorted(group, key=lambda x: x["estimated_step_time_ms"])
        rank_e = next(i for i, x in enumerate(est_sorted, 1) if x["plan"] == plan)
        error = abs(est - act) / act * 100
        lines.append(
            f"({plan['pp']},{plan['tp']},{plan['dp']}) & "
            f"{est:.1f} & {act:.1f} & "
            f"{ratio:.2f} & {rank_e} & {rank_a} & {error:.1f}\\% \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_scatter_plot(groups: Dict[int, List[Dict]], output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {4: "C0", 6: "C1", 8: "C2"}
    markers = {4: "o", 6: "s", 8: "^"}

    all_est = []
    all_act = []
    for ws in sorted(groups.keys()):
        group = groups[ws]
        est = [r["estimated_step_time_ms"] for r in group]
        act = [r["actual"]["avg_step_time_ms"] for r in group]
        all_est.extend(est)
        all_act.extend(act)
        ax.scatter(est, act, c=colors[ws], marker=markers[ws], label=f"{ws} GPUs", alpha=0.7, s=80)

    # y = x reference line
    max_val = max(max(all_est), max(all_act)) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="Perfect prediction")

    ax.set_xlabel("Estimated Step Time (ms)")
    ax.set_ylabel("Actual Step Time (ms)")
    ax.set_title("Cost Model: Estimated vs. Actual Step Time")
    ax.legend(loc="upper left")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Scatter plot saved to {output_path}")


def generate_stats_json(groups: Dict[int, List[Dict]], output_path: Path):
    stats_out = {}
    for ws in sorted(groups.keys()):
        group = groups[ws]
        stats_out[ws] = {
            "num_plans": len(group),
            "winner_accuracy": compute_winner_accuracy(group),
            "pairwise_accuracy": compute_pairwise_accuracy(group),
            "spearman_rho": compute_spearman(group)[0],
            "mape": compute_mape(group),
            **compute_ratio_stats(group),
        }
    with open(output_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Stats saved to {output_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = load_results()
    groups = group_by_world_size(results)

    # LaTeX tables
    table_tex = generate_latex_table(groups)
    detailed_tex = generate_detailed_table(groups)
    with open(OUTPUT_DIR / "thesis_tables.tex", "w") as f:
        f.write(table_tex + "\n\n%\n% Detailed 8-GPU table\n%\n\n")
        f.write(detailed_tex + "\n")
    print(f"LaTeX tables written to {OUTPUT_DIR / 'thesis_tables.tex'}")

    # Scatter plot
    generate_scatter_plot(groups, OUTPUT_DIR / "scatter_plot.png")

    # Stats JSON
    generate_stats_json(groups, OUTPUT_DIR / "ranking_stats.json")

    # Print summary to console
    print("\n" + "=" * 60)
    print("PRIORITY 0 VALIDATION SUMMARY")
    print("=" * 60)
    for ws in sorted(groups.keys()):
        group = groups[ws]
        print(f"\n{ws} GPUs ({len(group)} plans):")
        print(f"  Winner accuracy:   {compute_winner_accuracy(group)}")
        print(f"  Pairwise accuracy: {compute_pairwise_accuracy(group)*100:.1f}%")
        print(f"  Spearman rho:      {compute_spearman(group)[0]:.3f}")
        print(f"  MAPE:              {compute_mape(group)*100:.1f}%")
        rs = compute_ratio_stats(group)
        print(f"  Ratio est/act:     {rs['mean_ratio']:.2f} (range {rs['min_ratio']:.2f}–{rs['max_ratio']:.2f})")


if __name__ == "__main__":
    main()
