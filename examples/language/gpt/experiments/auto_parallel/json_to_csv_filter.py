#!/usr/bin/env python3
"""
Convert JSON result files in results/ (root only, no subdirs) into 2 CSVs
with duplicate-filtering for dp_outside variants that are actually identical.

Logic:
  - Group files by (model, world_size, pp, tp, dp, layers, hidden, heads,
    seq, batch, microbatches).
  - If a group has >=2 entries differing only by dp_outside:
      * If pp==1 or dp==1: the two topologies are identical. Keep only the
        row whose |ratio_actual_estimate - 1| is smallest (closest to 1).
      * If pp>1 and dp>1: the topologies are truly different — keep both.
  - All other files are kept as-is.

Outputs:
  - results/gpt2_results_filtered.csv
  - results/qwen25_results_filtered.csv

Run from the auto_parallel directory:
    python json_to_csv_filter.py
"""

import csv
import json
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def parse_json_file(path: Path) -> dict:
    with open(path, "r") as f:
        data = json.load(f)

    plan = data.get("plan", {})
    model = data.get("model", {})
    profile = data.get("profile", {})
    est = data.get("estimated_breakdown_ms", {})
    actual = data.get("actual", {})

    est_total = data.get("estimated_step_time_ms", 0.0)
    actual_avg = actual.get("avg_step_time_ms", 0.0)
    ratio = actual_avg / est_total if est_total > 0 else None

    return {
        "filename": path.name,
        "model": "gpt2" if path.name.startswith("gpt2_") else ("qwen25" if path.name.startswith("qwen25_") else "unknown"),
        "world_size": plan.get("world_size"),
        "pp": plan.get("pp"),
        "tp": plan.get("tp"),
        "dp": plan.get("dp"),
        "dp_outside": data.get("dp_outside"),
        "layers": model.get("layers"),
        "hidden": model.get("hidden"),
        "heads": model.get("heads"),
        "seq": model.get("seq"),
        "batch": model.get("batch"),
        "microbatches": model.get("microbatches"),
        "steps": model.get("steps"),
        "T_block_ms": profile.get("T_block_ms"),
        "estimated_total_ms": est_total,
        "compute_ms": est.get("compute"),
        "bubble_ms": est.get("bubble"),
        "tp_comm_ms": est.get("tp_comm"),
        "pp_comm_ms": est.get("pp_comm"),
        "dp_comm_ms": est.get("dp_comm"),
        "execution_overhead_ms": est.get("execution_overhead"),
        "embedding_ms": est.get("embedding"),
        "lm_head_ms": est.get("lm_head"),
        "actual_avg_ms": actual_avg,
        "ratio_actual_estimate": round(ratio, 3) if ratio is not None else None,
    }


def main():
    # Only files directly in RESULTS_DIR, not in subdirectories
    json_files = [f for f in RESULTS_DIR.iterdir() if f.is_file() and f.suffix == ".json"]

    all_rows = []
    for jf in sorted(json_files):
        try:
            row = parse_json_file(jf)
        except Exception as e:
            print(f"Warning: failed to parse {jf.name}: {e}")
            continue
        all_rows.append(row)

    # Group by configuration key (excluding filename, dp_outside, ratio)
    group_key_fields = [
        "model", "world_size", "pp", "tp", "dp",
        "layers", "hidden", "heads", "seq", "batch", "microbatches",
    ]

    groups = {}
    for row in all_rows:
        key = tuple(row[f] for f in group_key_fields)
        groups.setdefault(key, []).append(row)

    kept_rows = []
    filtered_rows = []

    for key, rows in groups.items():
        pp = key[2]   # index of 'pp' in group_key_fields
        tp = key[3]   # index of 'tp' in group_key_fields
        dp = key[4]   # index of 'dp' in group_key_fields

        if len(rows) >= 2 and (pp == 1 or dp == 1):
            # These are topology-identical duplicates; keep the one closest to ratio=1
            best = min(rows, key=lambda r: abs((r["ratio_actual_estimate"] or float('inf')) - 1))
            kept_rows.append(best)
            for r in rows:
                if r["filename"] != best["filename"]:
                    filtered_rows.append(r)
            print(
                f"[FILTER] {key[0]} ws={key[1]} pp={pp} tp={tp} dp={dp}: "
                f"kept {best['filename']} (ratio={best['ratio_actual_estimate']}), "
                f"dropped {[r['filename'] for r in rows if r['filename'] != best['filename']]}")
        else:
            kept_rows.extend(rows)

    # Split by model
    gpt2_rows = [r for r in kept_rows if r["model"] == "gpt2"]
    qwen_rows = [r for r in kept_rows if r["model"] == "qwen25"]

    columns = [
        "filename", "model", "world_size", "pp", "tp", "dp", "dp_outside",
        "layers", "hidden", "heads", "seq", "batch", "microbatches", "steps",
        "T_block_ms", "estimated_total_ms", "compute_ms", "bubble_ms",
        "tp_comm_ms", "pp_comm_ms", "dp_comm_ms", "execution_overhead_ms",
        "embedding_ms", "lm_head_ms", "actual_avg_ms", "ratio_actual_estimate",
    ]

    def write_csv(rows, out_name):
        out_path = RESULTS_DIR / out_name
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")

    write_csv(gpt2_rows, "gpt2_results_filtered.csv")
    write_csv(qwen_rows, "qwen25_results_filtered.csv")

    if filtered_rows:
        print(f"\nTotal filtered out: {len(filtered_rows)} duplicate(s)")
    else:
        print("\nNo duplicates were filtered.")


if __name__ == "__main__":
    main()
