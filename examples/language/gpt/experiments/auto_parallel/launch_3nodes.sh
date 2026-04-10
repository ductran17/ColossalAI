#!/usr/bin/env bash
# Launch auto-3D-parallel training across 3 nodes from node 18 only.
#
# Usage (run only on 10.10.10.18):
#   bash launch_3nodes.sh [extra args forwarded to run_3d_auto_parallel.py]
#
# Examples:
#   bash launch_3nodes.sh                          # Phase 1 auto-search
#   bash launch_3nodes.sh --hetero                 # Phase 2: hetero-TP
#   bash launch_3nodes.sh --var-stages             # Phase 2b: variable-size stages
#   bash launch_3nodes.sh --hetero --layers 8 --batch 4
#
# Requires passwordless SSH from node 18 → 20 and 18 → 16.
# Verify with: ssh 10.10.10.20 hostname && ssh 10.10.10.16 hostname
#
# Node layout:
#   10.10.10.18  node_rank=0  2 GPUs   (this machine, master)
#   10.10.10.20  node_rank=1  4 GPUs
#   10.10.10.16  node_rank=2  2 GPUs
#   ─────────────────────────────────
#   Total                     8 GPUs   world_size=8

set -euo pipefail

# ── configurable ────────────────────────────────────────────────────────────
MASTER_ADDR=10.10.10.18
MASTER_PORT=29500
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full path to torchrun in the conda environment.
# Adjust if conda is installed elsewhere on the remote nodes.
TORCHRUN=/root/miniconda3/bin/torchrun

# Pass --nccl-test to run the minimal all_reduce test instead of training.
if [[ "${1:-}" == "--nccl-test" ]]; then
    SCRIPT="$SCRIPT_DIR/test_nccl_allreduce.py"
    TRAIN_ARGS=""
else
    SCRIPT="$SCRIPT_DIR/run_3d_auto_parallel.py"
    TRAIN_ARGS="${*:---hetero --layers 4 --batch 2 --steps 3}"   # default if no args given
fi

NODE20=10.10.10.20
NODE16=10.10.10.16

# NCCL must use the Ethernet interface that connects the nodes.
# NCCL_ASYNC_ERROR_HANDLING=1 : turn hangs into hard errors with messages.
# NCCL_DEBUG=WARN              : print NCCL errors/warnings (set to INFO for full trace).
# NCCL_DEBUG_FILE              : per-rank debug log (only written when NCCL_DEBUG is set).
NCCL_ENVS="NCCL_SOCKET_IFNAME=bond-local NCCL_IB_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO"
# Note: no NCCL_DEBUG_FILE — debug output goes to stderr which torchrun captures in log files.
# ── end configurable ─────────────────────────────────────────────────────────

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "=== Auto 3D Parallel — 3-node launch ==="
echo "  Master : $MASTER_ADDR:$MASTER_PORT"
echo "  Script : $SCRIPT"
echo "  Args   : $TRAIN_ARGS"
echo "  Logs   : $LOG_DIR/node{18,20,16}_$TS.log"
echo ""

# Common torchrun base command (substitutions happen per node).
_torchrun_cmd() {
    local node_rank=$1
    local nproc=$2
    echo "env $NCCL_ENVS MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
$TORCHRUN \
  --nnodes=3 \
  --nproc_per_node=$nproc \
  --node_rank=$node_rank \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $SCRIPT $TRAIN_ARGS"
}

# ── Start remote workers first (they wait for the master's TCP store) ───────
echo "[1/3] Starting node 20 (4 GPUs) via SSH ..."
ssh "$NODE20" "$(_torchrun_cmd 1 4)" \
    > "$LOG_DIR/node20_$TS.log" 2>&1 &
PID_NODE20=$!

echo "[2/3] Starting node 16 (2 GPUs) via SSH ..."
ssh "$NODE16" "$(_torchrun_cmd 2 2)" \
    > "$LOG_DIR/node16_$TS.log" 2>&1 &
PID_NODE16=$!

# Small pause so SSH connections are established before the master starts.
sleep 1

# ── Start local master (node 18, node_rank=0) ───────────────────────────────
echo "[3/3] Starting node 18 (2 GPUs) locally — this is the master ..."
CMD_NODE18="$(_torchrun_cmd 0 2)"
eval "env $NCCL_ENVS MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
$TORCHRUN \
  --nnodes=3 \
  --nproc_per_node=2 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $SCRIPT $TRAIN_ARGS" \
    2>&1 | tee "$LOG_DIR/node18_$TS.log"
EXIT_LOCAL=$?

# ── Wait for remotes and collect exit codes ──────────────────────────────────
wait "$PID_NODE20" || EXIT_NODE20=$? && EXIT_NODE20=${EXIT_NODE20:-0}
wait "$PID_NODE16" || EXIT_NODE16=$? && EXIT_NODE16=${EXIT_NODE16:-0}

echo ""
echo "=== Exit codes: node18=$EXIT_LOCAL  node20=$EXIT_NODE20  node16=$EXIT_NODE16 ==="

if [[ $EXIT_LOCAL -ne 0 || $EXIT_NODE20 -ne 0 || $EXIT_NODE16 -ne 0 ]]; then
    echo "One or more nodes failed. Check logs in $LOG_DIR/"
    exit 1
fi
echo "Done."
