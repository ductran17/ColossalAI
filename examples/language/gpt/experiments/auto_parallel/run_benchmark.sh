#!/usr/bin/env bash
# Benchmark runner for Auto 3D Parallel experiments.
#
# Automates running multiple configurations, collecting JSON results,
# and generating a summary table.
#
# Usage:
#   bash run_benchmark.sh [config_file]
#
# Default config: benchmark_config.yaml in the same directory.
#
# Example benchmark_config.yaml:
#   ---
#   # Node sets to test (each is a list of node names from nodes_config.yaml)
#   node_sets:
#     - [node18, node20]
#     - [node18, node15, node16]
#     - [node18, node15, node16, node19]
#     - [node18, node15, node16, node19, node20]
#
#   # Model configurations
#   models:
#     - { layers: 8,  hidden: 256,  heads: 4,  seq: 64,  batch: 4,  microbatches: 4,  steps: 3 }
#     - { layers: 12, hidden: 512,  heads: 8,  seq: 128, batch: 8,  microbatches: 8,  steps: 5 }
#     - { layers: 24, hidden: 1024, heads: 16, seq: 512, batch: 16, microbatches: 8,  steps: 5 }
#
#   # Baselines to compare against auto-plan
#   baselines:
#     - name: auto
#       args: "--auto"
#     - name: manual_pp2tp2
#       args: "--hybrid --pp 2 --tp 2"
#     - name: manual_pp4tp2
#       args: "--hybrid --pp 4 --tp 2"
#     - name: manual_pp6tp2
#       args: "--hybrid --pp 6 --tp 2"
#     - name: manual_pp12tp1
#       args: "--hybrid --pp 12 --tp 1"
#
#   # Output directory for JSON results
#   results_dir: ./results
#
#   # Timeout per run (seconds)
#   timeout: 600

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-$SCRIPT_DIR/benchmark_config.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$SCRIPT_DIR/results/benchmark_summary_${TIMESTAMP}.csv"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    echo "Creating default config..."
    cat > "$CONFIG_FILE" <<'EOF'
# Default benchmark config
node_sets:
  - [node18, node20]
  - [node18, node15, node16, node19, node20]

models:
  - { layers: 12, hidden: 256, heads: 4, seq: 64, batch: 8, microbatches: 8, steps: 3 }

baselines:
  - name: auto
    args: "--auto"
  - name: manual_pp2tp2
    args: "--hybrid --pp 2 --tp 2 --layers 12 --batch 8 --microbatches 8 --steps 3"
  - name: manual_pp6tp2
    args: "--hybrid --pp 6 --tp 2 --layers 12 --batch 8 --microbatches 8 --steps 3"
  - name: manual_pp12tp1
    args: "--hybrid --pp 12 --tp 1 --layers 12 --batch 8 --microbatches 8 --steps 3"

results_dir: ./results
timeout: 600
EOF
    echo "Created default config: $CONFIG_FILE"
    echo "Please edit it and re-run."
    exit 0
fi

# Parse YAML using Python (requires pyyaml)
python3 -c "import yaml" 2>/dev/null || {
    echo "Error: PyYAML not installed. Run: pip install pyyaml"
    exit 1
}

parse_yaml() {
    python3 -c "
import yaml, sys, json
cfg = yaml.safe_load(open('$CONFIG_FILE'))
print(json.dumps(cfg))
"
}

CONFIG_JSON=$(parse_yaml)
RESULTS_DIR=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('results_dir','./results'))")
TIMEOUT=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('timeout',600))")

mkdir -p "$RESULTS_DIR"

# Write CSV header
echo "run_id,nodes,gpus,baseline,model,pp,tp,dp,status,estimated_ms,actual_avg_ms,profile_time_s,T_block_ms,min_free_gb,json_path" > "$SUMMARY_FILE"

run_id=0

# Iterate over node sets
node_sets_count=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['node_sets']))")
models_count=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models']))")
baselines_count=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['baselines']))")

echo "============================================"
echo " Benchmark Run Started: $TIMESTAMP"
echo " Config: $CONFIG_FILE"
echo " Results: $RESULTS_DIR"
echo " Node sets: $node_sets_count"
echo " Models: $models_count"
echo " Baselines per model: $baselines_count"
echo " Total runs: $((node_sets_count * models_count * baselines_count))"
echo "============================================"
echo ""

for i in $(seq 0 $((node_sets_count - 1))); do
    nodes=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); n=d['node_sets'][$i]; print(' '.join(n))")
    gpus=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); n=d['node_sets'][$i]; print(sum(len(yaml.safe_load(open('$SCRIPT_DIR/nodes_config.yaml'))[ni]['GPU_enabled']) for ni in n))")

    for j in $(seq 0 $((models_count - 1))); do
        layers=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['models'][$j]['layers'])")
        hidden=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['models'][$j]['hidden'])")
        heads=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['models'][$j]['heads'])")
        seq=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['models'][$j]['seq'])")
        batch=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['models'][$j]['batch'])")
        microbatches=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['models'][$j]['microbatches'])")
        steps=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['models'][$j]['steps'])")
        model_desc="L${layers}H${hidden}B${batch}"

        for k in $(seq 0 $((baselines_count - 1))); do
            baseline_name=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['baselines'][$k]['name'])")
            baseline_args=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['baselines'][$k]['args'])")

            run_id=$((run_id + 1))
            echo "[$run_id] nodes=[$nodes] gpus=$gpus baseline=$baseline_name model=$model_desc"

            # Build command
            if [[ "$baseline_name" == "auto" ]]; then
                cmd="bash launch_nodes.sh $nodes --auto --layers $layers --hidden $hidden --heads $heads --seq $seq --batch $batch --microbatches $microbatches --steps $steps"
            else
                cmd="bash launch_nodes.sh $nodes $baseline_args --layers $layers --hidden $hidden --heads $heads --seq $seq --batch $batch --microbatches $microbatches --steps $steps"
            fi

            # Run with timeout
            log_file="$RESULTS_DIR/run_${run_id}_${baseline_name}_${gpus}gpu_${model_desc}.log"
            t_start=$(date +%s)
            if timeout "$TIMEOUT" bash -c "cd '$SCRIPT_DIR' && $cmd" > "$log_file" 2>&1; then
                status="SUCCESS"
            else
                status="FAILED"
            fi
            t_end=$(date +%s)
            elapsed=$((t_end - t_start))

            # Extract metrics from JSON result file (most recent matching file)
            json_file=$(ls -t "$RESULTS_DIR"/auto_parallel_${gpus}gpu_${layers}L_${hidden}H_${batch}B_pp*_*_*.json 2>/dev/null | head -n 1 || echo "")
            if [[ -n "$json_file" && -f "$json_file" ]]; then
                pp=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['plan']['pp'])")
                tp=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['plan']['tp'])")
                dp=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['plan']['dp'])")
                est=$(python3 -c "import json; d=json.load(open('$json_file')); print(d.get('estimated_step_time_ms',''))")
                act=$(python3 -c "import json; d=json.load(open('$json_file')); print(d.get('actual',{}).get('avg_step_time_ms',''))")
                tblock=$(python3 -c "import json; d=json.load(open('$json_file')); print(d.get('profile',{}).get('T_block_ms',''))")
                minfree=$(python3 -c "import json; d=json.load(open('$json_file')); print(d.get('profile',{}).get('min_free_memory_gb',''))")
            else
                pp=""; tp=""; dp=""; est=""; act=""; tblock=""; minfree=""
            fi

            echo "$run_id,$nodes,$gpus,$baseline_name,$model_desc,$pp,$tp,$dp,$status,$est,$act,$elapsed,$tblock,$minfree,$json_file" >> "$SUMMARY_FILE"

            echo "      -> $status (${elapsed}s)  json=$json_file"
            echo ""
        done
    done
done

echo "============================================"
echo " Benchmark Complete"
echo " Summary: $SUMMARY_FILE"
echo " JSON results: $RESULTS_DIR"
echo "============================================"
echo ""

# Print quick summary table
echo "Quick Summary:"
python3 -c "
import csv, sys
with open('$SUMMARY_FILE') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'Total runs: {len(rows)}')
print(f'Successful: {sum(1 for r in rows if r[\"status\"]==\"SUCCESS\")}')
print(f'Failed:     {sum(1 for r in rows if r[\"status\"]==\"FAILED\")}')
print()
print(f'{\"Run\":>4} {\"Baseline\":>16} {\"GPUs\":>4} {\"Model\":>12} {\"PP\":>3} {\"TP\":>3} {\"DP\":>3} {\"Est(ms)\":>8} {\"Act(ms)\":>8} {\"Status\":>8}')
print('-' * 80)
for r in rows:
    print(f'{r[\"run_id\"]:>4} {r[\"baseline\"]:>16} {r[\"gpus\"]:>4} {r[\"model\"]:>12} {r[\"pp\"]:>3} {r[\"tp\"]:>3} {r[\"dp\"]:>3} {r[\"estimated_ms\"]:>8} {r[\"actual_avg_ms\"]:>8} {r[\"status\"]:>8}')
"
