#!/bin/bash
# Experiment 1: enwik8 three-way comparison
#
# Compute-matched comparison of:
#   - RegularTransformer: 8 distinct layers, each applied once
#   - LoopedTransformer:  1 shared layer, looped 8 times
#   - ChoosyTransformer:  8 pool layers, 8 routing steps
#   - ChoosyTransformer:  same, but no load-balancing loss (ablation)
#
# Runs 3 seeds for error bars. Fits in ~12 hours on 4 GPUs.
#
# Usage:
#   bash scripts/experiments/exp1-enwik8-comparison.sh
#
set -e

unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,4,5

# Shared training config
STEPS=10000
EVAL_EVERY=500
SAVE_EVERY=2000
BATCH_SIZE=32
SEQ_LEN=256
SEEDS=(42 137 271)

export WANDB_PROJECT="choosy-experiments"

echo "============================================"
echo "  Experiment 1: enwik8 three-way comparison"
echo "  Steps: $STEPS | Batch: $BATCH_SIZE | Seq: $SEQ_LEN"
echo "  Seeds: ${SEEDS[*]}"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

run_model() {
    local name=$1
    local model_cfg=$2
    local seed=$3
    local extra_args="${4:-}"

    local run_name="exp1-${name}-s${seed}"
    local ckpt_dir="checkpoints/exp1-${name}-s${seed}"

    echo "=== ${run_name} ==="
    WANDB_NAME="$run_name" \
    WANDB_TAGS="exp1,enwik8,compute-matched,seed-${seed},${name}" \
    python -m choosy.train \
        model=$model_cfg \
        training.num_steps=$STEPS \
        training.eval_every=$EVAL_EVERY \
        training.save_every=$SAVE_EVERY \
        training.data_parallel=true \
        training.checkpoint_dir=$ckpt_dir \
        training.seed=$seed \
        data.batch_size=$BATCH_SIZE \
        data.seq_len=$SEQ_LEN \
        $extra_args
    echo ""
}

for SEED in "${SEEDS[@]}"; do
    echo "========== Seed: $SEED =========="
    echo ""

    run_model "regular"           "regular-toy" $SEED
    run_model "looped"            "looped-toy"  $SEED
    run_model "choosy"            "choosy-toy"  $SEED
    run_model "choosy-no-balance" "choosy-toy"  $SEED "model.routing_loss_weight=0.0"
done

echo "============================================"
echo "  Experiment 1 complete!"
echo "  Results in: checkpoints/exp1-*/"
echo "  WandB: $WANDB_PROJECT"
echo "============================================"
