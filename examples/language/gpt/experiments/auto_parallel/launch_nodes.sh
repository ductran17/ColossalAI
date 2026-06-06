#!/usr/bin/env bash
# Launch distributed training on user-specified nodes with selected GPUs.
#
# Usage:
#   bash launch_nodes.sh <node_name> [<node_name> ...] [<training_args>]
#
# Examples:
#   bash launch_nodes.sh node18 node20 node16 --auto
#   bash launch_nodes.sh node12 node14 --hybrid --pp 2 --tp 2
#   bash launch_nodes.sh node12 --nccl-test
#   bash launch_nodes.sh node18 node20            # default 3D auto-parallel args
#
# Node definitions are read from nodes_config.yaml in the same directory.
# The first node in the argument list becomes the master (node_rank=0).
# Only the GPUs listed in GPU_enabled will be used on each node.
# GPU indices follow the order shown by nvidia-smi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/nodes_config.yaml"
TORCHRUN=/root/miniconda3/bin/torchrun

MASTER_PORT=29500
NCCL_ENVS="NCCL_SOCKET_IFNAME=bond-local NCCL_IB_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO"

# ── Helper: read YAML via Python ─────────────────────────────────────────────
_get_ip() {
    python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['$1']['IP'])"
}
_get_gpus() {
    python3 -c "import yaml; print(','.join(str(g) for g in yaml.safe_load(open('$CONFIG_FILE'))['$1']['GPU_enabled']))"
}
_get_gpu_count() {
    python3 -c "import yaml; print(len(yaml.safe_load(open('$CONFIG_FILE'))['$1']['GPU_enabled']))"
}

# ── Parse arguments ──────────────────────────────────────────────────────────
NODE_NAMES=()
TRAIN_ARGS=()
mode_found=false

for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        mode_found=true
    fi
    if [[ "$mode_found" == false ]]; then
        NODE_NAMES+=("$arg")
    else
        TRAIN_ARGS+=("$arg")
    fi
done

if [[ ${#NODE_NAMES[@]} -eq 0 ]]; then
    echo "Error: No node names provided."
    echo "Usage: bash launch_nodes.sh <node_name> [<node_name> ...] [<training_args>]"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ── Validate nodes and compute layout ────────────────────────────────────────
declare -a NODE_IPS
declare -a NODE_GPUS
declare -a NODE_NPROC
TOTAL_GPUS=0

for name in "${NODE_NAMES[@]}"; do
    if ! python3 -c "import yaml, sys; d=yaml.safe_load(open('$CONFIG_FILE')); sys.exit(0 if '$name' in d else 1)" 2>/dev/null; then
        echo "Error: Node '$name' not found in $CONFIG_FILE"
        exit 1
    fi
    ip=$(_get_ip "$name")
    gpus=$(_get_gpus "$name")
    nproc=$(_get_gpu_count "$name")
    if [[ "$nproc" -eq 0 ]]; then
        echo "Error: Node '$name' has no GPUs enabled in config."
        exit 1
    fi
    NODE_IPS+=("$ip")
    NODE_GPUS+=("$gpus")
    NODE_NPROC+=("$nproc")
    TOTAL_GPUS=$((TOTAL_GPUS + nproc))
done

MASTER_ADDR=${NODE_IPS[0]}
NNODES=${#NODE_NAMES[@]}

# ── Select training script and defaults ──────────────────────────────────────
if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    first_arg="${TRAIN_ARGS[0]}"
else
    first_arg=""
fi

if [[ "$first_arg" == "--nccl-test" ]]; then
    SCRIPT="$SCRIPT_DIR/test_nccl_allreduce.py"
    FINAL_ARGS=""
elif [[ "$first_arg" == "--profile-test" ]]; then
    SCRIPT="$SCRIPT_DIR/test_profiler.py"
    FINAL_ARGS=""
elif [[ "$first_arg" == "--hybrid" ]]; then
    SCRIPT="$SCRIPT_DIR/run_hybrid_parallel.py"
    if [[ ${#TRAIN_ARGS[@]} -ge 2 ]]; then
        FINAL_ARGS="${TRAIN_ARGS[*]:1}"
    else
        FINAL_ARGS="--pp 2 --tp 2 --layers 4 --batch 2 --steps 3"
    fi
elif [[ "$first_arg" == "--auto" ]]; then
    SCRIPT="$SCRIPT_DIR/run_auto_hybrid_parallel.py"
    if [[ ${#TRAIN_ARGS[@]} -ge 2 ]]; then
        FINAL_ARGS="${TRAIN_ARGS[*]:1}"
    else
        FINAL_ARGS="--layers 8 --hidden 256 --heads 4 --seq 64 --batch 4 --microbatches 4 --steps 3"
    fi
else
    SCRIPT="$SCRIPT_DIR/run_3d_auto_parallel.py"
    if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
        FINAL_ARGS="${TRAIN_ARGS[*]}"
    else
        FINAL_ARGS="--hetero --layers 4 --batch 2 --steps 3"
    fi
fi

# ── Logging setup ────────────────────────────────────────────────────────────
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "=== Dynamic Distributed Launch ==="
echo "  Nodes  : ${NODE_NAMES[*]}"
echo "  Master : $MASTER_ADDR:$MASTER_PORT"
echo "  NNODES : $NNODES"
echo "  World  : $TOTAL_GPUS GPUs"
echo "  Script : $SCRIPT"
echo "  Args   : $FINAL_ARGS"
echo "  Logs   : $LOG_DIR/<node>_$TS.log"
echo ""

# ── Build torchrun command ───────────────────────────────────────────────────
_torchrun_cmd() {
    local node_rank=$1
    local nproc=$2
    local cuda_devices=$3
    echo "env CUDA_VISIBLE_DEVICES=$cuda_devices $NCCL_ENVS MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
$TORCHRUN \
  --nnodes=$NNODES \
  --nproc_per_node=$nproc \
  --node_rank=$node_rank \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $SCRIPT $FINAL_ARGS"
}

# ── Detect local machine IPs ─────────────────────────────────────────────────
LOCAL_IPS=$(hostname -I 2>/dev/null | tr ' ' '\n' || true)

_is_local() {
    local target_ip=$1
    for ip in $LOCAL_IPS; do
        if [[ "$ip" == "$target_ip" ]]; then
            return 0
        fi
    done
    return 1
}

# ── Launch remote workers (all nodes except master) ──────────────────────────
declare -a REMOTE_PIDS=()

for i in $(seq 1 $((NNODES - 1))); do
    rank=$i
    name=${NODE_NAMES[$i]}
    ip=${NODE_IPS[$i]}
    nproc=${NODE_NPROC[$i]}
    gpus=${NODE_GPUS[$i]}

    echo "[$i/$((NNODES - 1))] Starting $name ($nproc GPUs, CUDA_VISIBLE_DEVICES=$gpus) via SSH ..."
    CMD=$(_torchrun_cmd "$rank" "$nproc" "$gpus")
    ssh "$ip" "$CMD" > "$LOG_DIR/${name}_$TS.log" 2>&1 &
    REMOTE_PIDS+=($!)
done

# ── Launch master ────────────────────────────────────────────────────────────
MASTER_NAME=${NODE_NAMES[0]}
MASTER_NPROC=${NODE_NPROC[0]}
MASTER_GPUS=${NODE_GPUS[0]}
MASTER_CMD=$(_torchrun_cmd 0 "$MASTER_NPROC" "$MASTER_GPUS")

sleep 1

if _is_local "$MASTER_ADDR"; then
    echo "[MASTER] Starting $MASTER_NAME ($MASTER_NPROC GPUs, CUDA_VISIBLE_DEVICES=$MASTER_GPUS) locally ..."
    eval "$MASTER_CMD" 2>&1 | tee "$LOG_DIR/${MASTER_NAME}_$TS.log"
    EXIT_MASTER=$?
else
    echo "[MASTER] Starting $MASTER_NAME ($MASTER_NPROC GPUs, CUDA_VISIBLE_DEVICES=$MASTER_GPUS) via SSH ..."
    ssh "$MASTER_ADDR" "$MASTER_CMD" 2>&1 | tee "$LOG_DIR/${MASTER_NAME}_$TS.log"
    EXIT_MASTER=$?
fi

# ── Wait for remotes and collect exit codes ──────────────────────────────────
EXIT_REMOTES=0
for pid in "${REMOTE_PIDS[@]}"; do
    if ! wait "$pid"; then
        EXIT_REMOTES=1
    fi
done

echo ""
echo "=== Exit code: master=$EXIT_MASTER  remotes=$EXIT_REMOTES ==="

# ── Auto-relaunch with fewer nodes if planner found better subset ───────────
RELAUNCH_FILE="$SCRIPT_DIR/RELAUNCH.txt"
if [[ -f "$RELAUNCH_FILE" ]]; then
    N_OPTIMAL=$(cat "$RELAUNCH_FILE" | tr -d '[:space:]')
    rm -f "$RELAUNCH_FILE"
    if [[ "$N_OPTIMAL" =~ ^[0-9]+$ ]] && [[ "$N_OPTIMAL" -lt "$NNODES" ]]; then
        OPTIMAL_NODES=("${NODE_NAMES[@]:0:$N_OPTIMAL}")
        echo ""
        echo "=== AUTO-RELAY: Planner found optimal with $N_OPTIMAL node(s) ==="
        echo "  Optimal nodes: ${OPTIMAL_NODES[*]}"
        echo "  Previous nodes: ${NODE_NAMES[*]}"
        echo "  Re-launching with optimal nodes ..."
        echo ""
        exec bash "$0" "${OPTIMAL_NODES[@]}" "${TRAIN_ARGS[@]}"
    fi
fi

if [[ $EXIT_MASTER -ne 0 || $EXIT_REMOTES -ne 0 ]]; then
    echo "One or more nodes failed. Check logs in $LOG_DIR/"
    exit 1
fi
echo "Done."
