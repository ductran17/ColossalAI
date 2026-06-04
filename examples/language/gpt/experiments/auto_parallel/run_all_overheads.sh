#!/usr/bin/env bash
# Run overhead measurement scripts across node18 and node19
# Measures: framework, optimizer, TP sync, DP sync (intra + cross), PP dispatch

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHRUN=/root/miniconda3/bin/torchrun
MASTER_ADDR=10.10.10.18
MASTER_PORT=29600
NCCL_ENVS="NCCL_SOCKET_IFNAME=bond-local NCCL_IB_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
export PYTHONPATH=/home/ductm27/ColossalAI:${PYTHONPATH:-}

echo "=========================================="
echo "OVERHEAD MEASUREMENT CAMPAIGN"
echo "Cluster: node18 (2xL40) + node19 (2xL40)"
echo "=========================================="

# 1. Adam optimizer (1 GPU, local)
echo ""
echo "[1/6] Measuring Adam optimizer step..."
python3 $SCRIPT_DIR/debug_overhead_2_optimizer.py

# 2. Framework per-microbatch (1 GPU, local)
echo ""
echo "[2/6] Measuring framework per-microbatch overhead..."
python3 $SCRIPT_DIR/debug_overhead_1_framework.py

# 3. TP AllReduce sync (intra-node, 2 GPUs on node18)
echo ""
echo "[3/6] Measuring TP AllReduce sync (intra-node, 2 GPUs)..."
NCCL_ENVS="$NCCL_ENVS" PYTHONPATH=$PYTHONPATH \
  $TORCHRUN --nproc_per_node=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  $SCRIPT_DIR/debug_overhead_3_tp_sync.py

# 4. DP AllReduce sync (intra-node, 2 GPUs on node18)
echo ""
echo "[4/6] Measuring DP AllReduce sync (intra-node, 2 GPUs)..."
NCCL_ENVS="$NCCL_ENVS" PYTHONPATH=$PYTHONPATH \
  $TORCHRUN --nproc_per_node=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  $SCRIPT_DIR/debug_overhead_4b_dp_intra.py

# 5. DP AllReduce sync (cross-node, node18 + node19)
echo ""
echo "[5/6] Measuring DP AllReduce sync (cross-node, 2 nodes)..."
# Start node19 first
ssh 10.10.10.19 "cd $SCRIPT_DIR && NCCL_ENVS=\"$NCCL_ENVS\" PYTHONPATH=$PYTHONPATH \
  $TORCHRUN --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  $SCRIPT_DIR/debug_overhead_4b_dp_intra.py" &
REMOTE_PID=$!

sleep 2

# Then start node18 (master)
NCCL_ENVS="$NCCL_ENVS" PYTHONPATH=$PYTHONPATH \
  $TORCHRUN --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  $SCRIPT_DIR/debug_overhead_4b_dp_intra.py

wait $REMOTE_PID || true

# 6. PP dispatch overhead (actual ColossalAI execute_pipeline)
echo ""
echo "[6/6] Measuring PP stage manager dispatch overhead..."
# Run pp=2,tp=1 on 2 GPUs node18
NCCL_ENVS="$NCCL_ENVS" PYTHONPATH=$PYTHONPATH \
  $TORCHRUN --nproc_per_node=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  $SCRIPT_DIR/debug_overhead_5_pp_dispatch.py

echo ""
echo "=========================================="
echo "All measurements complete."
echo "Results saved to /tmp/overhead_results.json"
echo "=========================================="
