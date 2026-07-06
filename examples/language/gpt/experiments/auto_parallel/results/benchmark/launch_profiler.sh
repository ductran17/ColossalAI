#!/usr/bin/env bash
# Launch alpha/beta profiler across multiple nodes from node18.
#
# Usage:
#   ./launch_profiler.sh <master_ip> [worker_ip ...] [-- <profiler_args>]
#
# Examples:
#   ./launch_profiler.sh 10.10.10.18 10.10.10.19
#   ./launch_profiler.sh 10.10.10.18 10.10.10.19 -- --warmup 20 --repeat 100 --sizes 256,512,1024
#
# Extra args after -- are forwarded to profile_alpha_beta.py.

set -euo pipefail

MASTER_ADDR="${1:-}"
if [[ -n "$MASTER_ADDR" ]]; then
    shift
fi

WORKERS=()
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
        WORKERS+=("$arg")
    fi
done

if [[ -z "$MASTER_ADDR" ]]; then
    echo "Usage: $0 <MASTER_IP> [WORKER_IP ...] [-- <profiler_args>]"
    echo ""
    echo "Examples:"
    echo "  $0 10.10.10.18 10.10.10.19"
    echo "  $0 10.10.10.18 10.10.10.19 -- --warmup 20 --repeat 100"
    exit 1
fi

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_PORT=29500

# NCCL environment variables (same network policy as launch_nodes.sh).  Keep
# CUDA peer-to-peer enabled so the intra-node benchmark measures the real link.
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond-local}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_ENVS="NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} NCCL_IB_DISABLE=${NCCL_IB_DISABLE} TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

ALL_NODES=("$MASTER_ADDR" "${WORKERS[@]}")
NNODES=${#ALL_NODES[@]}

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    NPROC="${NPROC_PER_NODE}"
elif command -v nvidia-smi &> /dev/null; then
    NPROC=$(nvidia-smi -L | wc -l)
else
    NPROC=2
fi

echo "[launch_profiler] Nodes: ${ALL_NODES[*]}"
echo "[launch_profiler] nnodes=${NNODES} nproc_per_node=${NPROC} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[launch_profiler] Profiler args: ${EXTRA_ARGS[*]}"
echo ""

export PYTHONPATH="/home/ductm27/ColossalAI:${PYTHONPATH:-}"

PIDS=()
NODE_RANK=0
LOCAL_IPS=" $(hostname -I 2>/dev/null || true) "
for node in "${ALL_NODES[@]}"; do
    LOG_FILE="${BENCH_DIR}/profiler_node${NODE_RANK}.log"
    CMD_FULL="cd ${BENCH_DIR} && PYTHONPATH=/home/ductm27/ColossalAI:\${PYTHONPATH} ${NCCL_ENVS} torchrun \\
        --nnodes=${NNODES} \\
        --nproc_per_node=${NPROC} \\
        --master_addr=${MASTER_ADDR} \\
        --master_port=${MASTER_PORT} \\
        --node_rank=${NODE_RANK} \\
        profile_alpha_beta.py \\
        ${EXTRA_ARGS[*]} 2>&1 | tee -a ${LOG_FILE}"

    echo "[launch_profiler] Starting node_rank=${NODE_RANK} on ${node} (log: ${LOG_FILE})..."
    if [[ "$LOCAL_IPS" == *" $node "* || "$node" == "127.0.0.1" || "$node" == "localhost" ]]; then
        PYTHONUNBUFFERED=1 eval "$CMD_FULL" &
    else
        ssh -t -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            "root@${node}" "$CMD_FULL" &
    fi
    PIDS+=($!)
    NODE_RANK=$((NODE_RANK + 1))
done

echo ""
echo "[launch_profiler] Waiting for all ${NNODES} nodes..."
wait "${PIDS[@]}"

echo "[launch_profiler] Done."
ls -la "${BENCH_DIR}"/e1_alpha_beta_raw.csv 2>/dev/null || true
