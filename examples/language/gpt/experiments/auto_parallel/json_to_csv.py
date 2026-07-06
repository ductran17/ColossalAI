#!/usr/bin/env python3
"""
Convert JSON result files in results/ (root only, no subdirs) into 2 CSVs:
  - gpt2_results.csv
  - qwen25_results.csv

Run from the auto_parallel directory:
    python json_to_csv.py
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

    gpt2_rows = []
    qwen_rows = []

    for jf in sorted(json_files):
        try:
            row = parse_json_file(jf)
        except Exception as e:
            print(f"Warning: failed to parse {jf.name}: {e}")
            continue

        if row["model"] == "gpt2":
            gpt2_rows.append(row)
        elif row["model"] == "qwen25":
            qwen_rows.append(row)
        else:
            print(f"Warning: unknown model prefix for {jf.name}, skipping.")

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

    write_csv(gpt2_rows, "gpt2_results.csv")
    write_csv(qwen_rows, "qwen25_results.csv")


if __name__ == "__main__":
    main()
