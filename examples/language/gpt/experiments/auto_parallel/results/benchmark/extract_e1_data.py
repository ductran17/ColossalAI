#!/usr/bin/env python3
"""
Extract profiler validation data (E1) from all JSON result files in results/ (root only).

Outputs:
  - results/e1_alpha_beta.csv        : E1.1  alpha/beta per run
  - results/e1_tblock.csv            : E1.2  T_block isolated vs representative

Run from the auto_parallel directory:
    python extract_e1_data.py
"""

import json
import csv
import os
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"


def load_json_files():
    files = [f for f in RESULTS_DIR.iterdir() if f.is_file() and f.suffix == ".json"]
    rows = []
    for jf in sorted(files):
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: skip {jf.name}: {e}")
            continue
        rows.append((jf.name, data))
    return rows


def extract_e11(data_rows):
    """E1.1 — Communication parameters alpha/beta."""
    records = []
    for fname, data in data_rows:
        model = "gpt2" if fname.startswith("gpt2_") else ("qwen25" if fname.startswith("qwen25_") else "unknown")
        prof = data.get("profile", {})
        plan = data.get("plan", {})

        # Each JSON is treated as one independent run
        records.append({
            "filename": fname,
            "model": model,
            "world_size": plan.get("world_size"),
            "pp": plan.get("pp"),
            "tp": plan.get("tp"),
            "dp": plan.get("dp"),
            "connection": "intra",
            "alpha_us": prof.get("alpha_intra_us"),
            "beta_ns_per_B": prof.get("beta_intra_ns_per_B"),
        })
        records.append({
            "filename": fname,
            "model": model,
            "world_size": plan.get("world_size"),
            "pp": plan.get("pp"),
            "tp": plan.get("tp"),
            "dp": plan.get("dp"),
            "connection": "inter",
            "alpha_us": prof.get("alpha_cross_us"),
            "beta_ns_per_B": prof.get("beta_cross_ns_per_B"),
        })

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "e1_alpha_beta.csv"
    df.to_csv(out_path, index=False)
    print(f"[E1.1] Wrote {len(df)} rows to {out_path}")
    return df


def extract_e12(data_rows):
    """E1.2 — T_block isolated vs representative."""
    records = []
    for fname, data in data_rows:
        model = "gpt2" if fname.startswith("gpt2_") else ("qwen25" if fname.startswith("qwen25_") else "unknown")
        prof = data.get("profile", {})
        plan = data.get("plan", {})
        model_cfg = data.get("model", {})

        t_iso = prof.get("T_block_ms")
        t_repr = prof.get("T_block_with_microbatches_ms")

        records.append({
            "filename": fname,
            "model": model,
            "world_size": plan.get("world_size"),
            "pp": plan.get("pp"),
            "tp": plan.get("tp"),
            "dp": plan.get("dp"),
            "microbatches": model_cfg.get("microbatches"),
            "T_block_isolated_ms": t_iso,
            "T_block_repr_ms": t_repr if t_repr and t_repr > 0 else None,
        })

    df = pd.DataFrame(records)
    # Only keep rows where repr is available (new JSONs after export fix)
    df_repr = df[df["T_block_repr_ms"].notna()].copy()
    out_path = RESULTS_DIR / "e1_tblock.csv"
    df.to_csv(out_path, index=False)
    print(f"[E1.2] Wrote {len(df)} rows to {out_path} ({len(df_repr)} have repr data)")
    return df


def compute_e11_stats(df):
    """Print summary statistics table for E1.1."""
    print("\n=== E1.1 Alpha/Beta Summary ===\n")
    for conn in ["intra", "inter"]:
        sub = df[df["connection"] == conn]
        if len(sub) == 0:
            continue
        print(f"{conn.upper()}-NODE (n={len(sub)}):")
        print(f"  alpha  mean={sub['alpha_us'].mean():.2f} us  std={sub['alpha_us'].std():.2f}  CV={sub['alpha_us'].std()/sub['alpha_us'].mean()*100:.1f}%")
        print(f"  beta   mean={sub['beta_ns_per_B'].mean():.4f} ns/B  std={sub['beta_ns_per_B'].std():.4f}  CV={sub['beta_ns_per_B'].std()/sub['beta_ns_per_B'].mean()*100:.1f}%")
        print()

    # Per-model stats
    print("Per-model breakdown:")
    summary = df.groupby(["model", "connection"]).agg(
        alpha_mean=("alpha_us", "mean"),
        alpha_std=("alpha_us", "std"),
        beta_mean=("beta_ns_per_B", "mean"),
        beta_std=("beta_ns_per_B", "std"),
        n=("alpha_us", "count"),
    ).round({"alpha_mean": 2, "alpha_std": 2, "beta_mean": 4, "beta_std": 4})
    print(summary)
    print()


def compute_e12_stats(df):
    """Print summary for E1.2."""
    df_repr = df[df["T_block_repr_ms"].notna()].copy()
    if len(df_repr) == 0:
        print("[E1.2] No T_block_with_microbatches data found in existing JSONs.")
        print("       This field was added to the export recently. Re-run experiments to populate it.")
        return

    df_repr["ratio"] = df_repr["T_block_repr_ms"] / df_repr["T_block_isolated_ms"]
    print("\n=== E1.2 T_block_isolated vs T_block_representative ===\n")
    print(df_repr[["model", "world_size", "microbatches", "T_block_isolated_ms", "T_block_repr_ms", "ratio"]].to_string(index=False))
    print(f"\nMean ratio (repr/isolated): {df_repr['ratio'].mean():.3f}")
    print(f"Range: [{df_repr['ratio'].min():.3f}, {df_repr['ratio'].max():.3f}]")


def main():
    data_rows = load_json_files()
    print(f"Loaded {len(data_rows)} JSON files from {RESULTS_DIR}\n")

    df_ab = extract_e11(data_rows)
    compute_e11_stats(df_ab)

    df_tb = extract_e12(data_rows)
    compute_e12_stats(df_tb)


if __name__ == "__main__":
    main()
