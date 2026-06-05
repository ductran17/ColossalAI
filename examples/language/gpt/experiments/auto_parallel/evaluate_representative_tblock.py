#!/usr/bin/env python3
"""
Evaluate the new representative T_block results vs old baseline.

Compares:
  - Absolute accuracy (MAPE, ratio est/act)
  - Ranking accuracy (winner ID, pairwise, Spearman)
  - Per-plan breakdown
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/ductm27/ColossalAI")

OLD_DIR = Path(__file__).parent / "results_before_estimate"
NEW_DIR = Path(__file__).parent / "results"

def load_results(dir_path, filter_config=None):
    """Load all JSON results from a directory. Optionally filter by config."""
    results = []
    for p in sorted(dir_path.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        if filter_config:
            model = r.get("model", {})
            if not all(model.get(k) == v for k, v in filter_config.items()):
                continue
        results.append(r)
    return results

def group_by_world_size(results):
    groups = {}
    for r in results:
        ws = r["plan"]["world_size"]
        groups.setdefault(ws, []).append(r)
    return groups

def compute_winner_accuracy(group):
    est_winner = min(group, key=lambda r: r["estimated_step_time_ms"])
    act_winner = min(group, key=lambda r: r["actual"]["avg_step_time_ms"])
    return est_winner["plan"] == act_winner["plan"]

def compute_pairwise_accuracy(group):
    n = len(group)
    if n < 2:
        return 1.0
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            est_i = group[i]["estimated_step_time_ms"]
            est_j = group[j]["estimated_step_time_ms"]
            act_i = group[i]["actual"]["avg_step_time_ms"]
            act_j = group[j]["actual"]["avg_step_time_ms"]
            if (est_i < est_j) == (act_i < act_j):
                correct += 1
            total += 1
    return correct / total if total > 0 else 1.0

def compute_spearman(group):
    try:
        from scipy import stats
        est = [r["estimated_step_time_ms"] for r in group]
        act = [r["actual"]["avg_step_time_ms"] for r in group]
        if len(est) < 2:
            return 1.0, 0.0
        rho, pval = stats.spearmanr(est, act)
        return rho, pval
    except ImportError:
        return None, None

def compute_mape(group):
    errors = []
    for r in group:
        est = r["estimated_step_time_ms"]
        act = r["actual"]["avg_step_time_ms"]
        errors.append(abs(est - act) / act)
    return sum(errors) / len(errors)

def compute_ratio_stats(group):
    ratios = [r["estimated_step_time_ms"] / r["actual"]["avg_step_time_ms"] for r in group]
    return {
        "mean": sum(ratios) / len(ratios),
        "min": min(ratios),
        "max": max(ratios),
    }

def summarize_group(group, label):
    if not group:
        return None
    ws = group[0]["plan"]["world_size"]
    n = len(group)
    winner = compute_winner_accuracy(group)
    pairwise = compute_pairwise_accuracy(group)
    rho, _ = compute_spearman(group)
    mape = compute_mape(group)
    ratios = compute_ratio_stats(group)
    
    return {
        "label": label,
        "ws": ws,
        "n": n,
        "winner": winner,
        "pairwise": pairwise,
        "rho": rho,
        "mape": mape,
        "ratios": ratios,
    }

def print_summary(summary):
    if summary is None:
        return
    s = summary
    print(f"\n{s['label']} ({s['ws']} GPUs, {s['n']} plans):")
    print(f"  Winner correct:    {s['winner']}")
    print(f"  Pairwise accuracy: {s['pairwise']*100:.1f}%")
    print(f"  Spearman rho:      {s['rho']:.3f}" if s['rho'] is not None else "  Spearman rho:      N/A")
    print(f"  MAPE:              {s['mape']*100:.1f}%")
    r = s['ratios']
    print(f"  Est/Act ratio:     {r['mean']:.2f} (range {r['min']:.2f}–{r['max']:.2f})")

def find_matching_plan(old_group, new_group):
    """Match plans between old and new by (pp, tp, dp)"""
    matches = []
    for old in old_group:
        old_plan = old["plan"]
        key = (old_plan["pp"], old_plan["tp"], old_plan["dp"])
        for new in new_group:
            new_plan = new["plan"]
            if (new_plan["pp"], new_plan["tp"], new_plan["dp"]) == key:
                matches.append((old, new))
                break
    return matches

def main():
    # Filter for hidden=1024, layers=24, batch=16 config (the main validation config)
    filter_cfg = {"layers": 24, "hidden": 1024, "batch": 16}
    
    print("="*70)
    print("EVALUATION: Representative T_block vs Old Baseline")
    print("Config: layers=24, hidden=1024, batch=16, seq=256")
    print("="*70)
    
    old_results = load_results(OLD_DIR, filter_cfg)
    new_results = load_results(NEW_DIR, filter_cfg)
    
    print(f"\nOld results: {len(old_results)} files")
    print(f"New results: {len(new_results)} files")
    
    old_groups = group_by_world_size(old_results)
    new_groups = group_by_world_size(new_results)
    
    # Compare per-world-size
    all_ws = sorted(set(list(old_groups.keys()) + list(new_groups.keys())))
    
    print("\n" + "="*70)
    print("SUMMARY BY CLUSTER SIZE")
    print("="*70)
    
    for ws in all_ws:
        old_g = old_groups.get(ws, [])
        new_g = new_groups.get(ws, [])
        
        old_s = summarize_group(old_g, "OLD")
        new_s = summarize_group(new_g, "NEW")
        
        print(f"\n{'='*70}")
        print(f"{ws} GPUs")
        print(f"{'='*70}")
        
        if old_s:
            print_summary(old_s)
        else:
            print(f"\n  OLD: No data")
            
        if new_s:
            print_summary(new_s)
        else:
            print(f"\n  NEW: No data")
        
        # Per-plan comparison for matched plans
        if old_g and new_g:
            matches = find_matching_plan(old_g, new_g)
            if matches:
                print(f"\n  Per-plan comparison ({len(matches)} matched plans):")
                print(f"  {'Plan':<12} {'Old Est':>10} {'New Est':>10} {'Actual':>10} {'Old Ratio':>10} {'New Ratio':>10} {'ΔRatio':>10}")
                print(f"  {'-'*80}")
                
                old_ratios = []
                new_ratios = []
                
                for old, new in matches:
                    plan = old["plan"]
                    plan_str = f"({plan['pp']},{plan['tp']},{plan['dp']})"
                    old_est = old["estimated_step_time_ms"]
                    new_est = new["estimated_step_time_ms"]
                    actual = new["actual"]["avg_step_time_ms"]
                    old_ratio = old_est / actual
                    new_ratio = new_est / actual
                    delta = new_ratio - old_ratio
                    
                    old_ratios.append(old_ratio)
                    new_ratios.append(new_ratio)
                    
                    print(f"  {plan_str:<12} {old_est:>10.1f} {new_est:>10.1f} {actual:>10.1f} {old_ratio:>10.2f} {new_ratio:>10.2f} {delta:>+10.2f}")
                
                # Average improvement
                avg_old = sum(old_ratios) / len(old_ratios)
                avg_new = sum(new_ratios) / len(new_ratios)
                improvement = abs(1 - avg_new) - abs(1 - avg_old)
                
                print(f"  {'-'*80}")
                print(f"  {'Average':<12} {'':>10} {'':>10} {'':>10} {avg_old:>10.2f} {avg_new:>10.2f} {improvement:>+10.2f}")
                
                # Closer to 1.0 is better
                if abs(1 - avg_new) < abs(1 - avg_old):
                    print(f"\n  => NEW estimates are CLOSER to actual (better absolute accuracy)")
                else:
                    print(f"\n  => OLD estimates were closer to actual")
    
    # Overall comparison
    print("\n" + "="*70)
    print("OVERALL COMPARISON")
    print("="*70)
    
    old_all = summarize_group(old_results, "OLD Overall")
    new_all = summarize_group(new_results, "NEW Overall")
    
    if old_all and new_all:
        print(f"\n  Winner accuracy:   OLD={old_all['winner']}  NEW={new_all['winner']}")
        print(f"  Pairwise accuracy: OLD={old_all['pairwise']*100:.1f}%  NEW={new_all['pairwise']*100:.1f}%")
        print(f"  Spearman rho:      OLD={old_all['rho']:.3f}  NEW={new_all['rho']:.3f}")
        print(f"  MAPE:              OLD={old_all['mape']*100:.1f}%  NEW={new_all['mape']*100:.1f}%")
        print(f"  Avg ratio:         OLD={old_all['ratios']['mean']:.2f}  NEW={new_all['ratios']['mean']:.2f}")
        
        # Determine if improved
        old_dist = abs(1 - old_all['ratios']['mean'])
        new_dist = abs(1 - new_all['ratios']['mean'])
        if new_dist < old_dist:
            print(f"\n  CONCLUSION: Representative T_block IMPROVED absolute accuracy")
            print(f"  (distance from 1.0: {old_dist:.2f} -> {new_dist:.2f})")
        else:
            print(f"\n  CONCLUSION: Representative T_block did NOT improve absolute accuracy")
            print(f"  (distance from 1.0: {old_dist:.2f} -> {new_dist:.2f})")
        
        if new_all['pairwise'] >= old_all['pairwise']:
            print(f"  Ranking accuracy maintained or improved")
        else:
            print(f"  WARNING: Ranking accuracy degraded")

if __name__ == "__main__":
    main()
