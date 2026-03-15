#!/bin/bash
# Experiment 1.1: Fix routing collapse in ChoosyTransformer
#
# Exp1 showed total routing collapse (entropy=0, layer 0 gets 100%).
# This experiment sweeps balancing loss weight, all with temperature annealing.
#
# Balancing loss weights: 0.005, 0.05, 0.1, 0.5, 1.0
#   - 0.005: below exp1's 0.01, test if annealing alone fixes collapse
#   - 0.05:  mild increase
#   - 0.1:   moderate (10× exp1)
#   - 0.5:   strong
#   - 1.0:   very strong, may hurt main loss
#
# All runs use temperature annealing 2.0 → 0.5.
# 5 configs × 3 seeds = 15 runs × ~7 min = ~2 hours.
#
set -e

unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,4,5

STEPS=10000
EVAL_EVERY=500
SAVE_EVERY=2000
BATCH_SIZE=32
SEQ_LEN=256
SEEDS=(42 137 271)
BALANCE_WEIGHTS=(0.01 0.005 0.05 0.1 0.25 0.5 1.0)

export WANDB_PROJECT="choosy-experiments"

echo "============================================"
echo "  Experiment 1.1: Balancing loss sweep"
echo "  Steps: $STEPS | Seeds: ${SEEDS[*]}"
echo "  Weights: ${BALANCE_WEIGHTS[*]}"
echo "  All with temp annealing 2.0 → 0.5"
echo "============================================"
echo ""

run_model() {
    local name=$1
    local seed=$2
    local extra_args="${3:-}"

    local run_name="exp1.1-${name}-s${seed}"
    local ckpt_dir="checkpoints/exp1.1-${name}-s${seed}"

    echo "=== ${run_name} ==="
    WANDB_NAME="$run_name" \
    WANDB_TAGS="exp1.1,enwik8,routing-fix,seed-${seed},${name}" \
    python -m choosy.train \
        model=choosy-toy \
        training.num_steps=$STEPS \
        training.eval_every=$EVAL_EVERY \
        training.save_every=$SAVE_EVERY \
        training.data_parallel=true \
        training.checkpoint_dir=$ckpt_dir \
        training.seed=$seed \
        data.batch_size=$BATCH_SIZE \
        data.seq_len=$SEQ_LEN \
        model.router_temp_start=2.0 \
        model.router_temp_end=0.5 \
        $extra_args
    echo ""
}

for SEED in "${SEEDS[@]}"; do
    echo "========== Seed: $SEED =========="
    echo ""

    for W in "${BALANCE_WEIGHTS[@]}"; do
        run_model "bal${W}" $SEED "model.routing_loss_weight=${W}"
    done
done

echo "============================================"
echo "  Experiment 1.1 complete!"
echo "  Results in: checkpoints/exp1.1-*/"
echo "============================================"
