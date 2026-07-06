#!/usr/bin/env bash
# Launch alpha/beta profiler across multiple nodes via ssh + torchrun.
# Usage:
#   ./launch_profiler.sh node18 node19
#   ./launch_profiler.sh node18 node19 node20
#   ./launch_profiler.sh node18 node19 -- --warmup 20 --repeat 100
#
# Extra args after -- are forwarded to profile_alpha_beta.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILER_SCRIPT="${SCRIPT_DIR}/profile_alpha_beta.py"
MASTER_ADDR="${1}"
MASTER_PORT=29500
shift

# Collect nodes and extra args
NODES=()
EXTRA_ARGS=()
passed_dash_dash=false
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        passed_dash_dash=true
        continue
    fi
    if $passed_dash_dash; then
        EXTRA_ARGS+=("$arg")
    else
        NODES+=("$arg")
    fi
done

NNODES=${#NODES[@]}
if [[ $NNODES -lt 1 ]]; then
    echo "Usage: $0 <master_node> [worker_node ...] [-- <profiler_args>]"
    exit 1
fi

OUTPUT_DIR="${SCRIPT_DIR}/results/profiler"
echo "[launch_profiler] Nodes: ${NODES[*]}"
echo "[launch_profiler] nnodes=${NNODES} nproc_per_node=2 master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[launch_profiler] Extra args: ${EXTRA_ARGS[*]}"
echo "[launch_profiler] Output will be on ${MASTER_ADDR}:${OUTPUT_DIR}"
echo ""

# Sync profiler script to all nodes first
for node in "${NODES[@]}"; do
    echo "[launch_profiler] Syncing to ${node}..."
    rsync -az --exclude='*.pyc' --exclude='__pycache__' \
        "${SCRIPT_DIR}/" "${node}:${SCRIPT_DIR}/" >/dev/null 2>&1
done

# Launch on all nodes
PIDS=()
NODE_RANK=0
for node in "${NODES[@]}"; do
    CMD="cd ${SCRIPT_DIR} && torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=2 \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        --node_rank=${NODE_RANK} \
        profile_alpha_beta.py \
        --output-dir ${OUTPUT_DIR} \
        ${EXTRA_ARGS[*]}"
    
    echo "[launch_profiler] Starting node_rank=${NODE_RANK} on ${node}..."
    if [[ "$node" == "$(hostname -s)" || "$node" == "localhost" ]]; then
        eval "$CMD" &
    else
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            "$node" "$CMD" &
    fi
    PIDS+=($!)
    ((NODE_RANK++))
done

# Wait for all
echo ""
echo "[launch_profiler] Waiting for all nodes to finish..."
wait "${PIDS[@]}"

# Pull results from master if we're not on it
if [[ "$(hostname -s)" != "${MASTER_ADDR}" ]]; then
    echo "[launch_profiler] Pulling results from ${MASTER_ADDR}..."
    rsync -az "${MASTER_ADDR}:${OUTPUT_DIR}/" "${OUTPUT_DIR}/"
fi

echo "[launch_profiler] Done. Results in ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"
