#!/bin/bash
# Experiment 1: enwik8 three-way comparison
#
# Compute-matched comparison of:
#   - RegularTransformer: 8 distinct layers, each applied once
#   - LoopedTransformer:  1 shared layer, looped 8 times
#   - ChoosyTransformer:  8 pool layers, 8 routing steps
#
# All use toy-sized models (d_model=128) for fast iteration.
# To scale up, replace model configs with the full-sized variants.
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

export WANDB_PROJECT="choosy-experiments"
export WANDB_TAGS="exp1,enwik8,compute-matched"

echo "============================================"
echo "  Experiment 1: enwik8 three-way comparison"
echo "  Steps: $STEPS | Batch: $BATCH_SIZE | Seq: $SEQ_LEN"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

# --- RegularTransformer ---
echo "=== [1/3] RegularTransformer ==="
WANDB_NAME="exp1-regular" python -m choosy.train \
    model=regular-toy \
    training.num_steps=$STEPS \
    training.eval_every=$EVAL_EVERY \
    training.save_every=$SAVE_EVERY \
    training.data_parallel=true \
    training.checkpoint_dir=checkpoints/exp1-regular \
    data.batch_size=$BATCH_SIZE \
    data.seq_len=$SEQ_LEN

echo ""

# --- LoopedTransformer ---
echo "=== [2/3] LoopedTransformer ==="
WANDB_NAME="exp1-looped" python -m choosy.train \
    model=looped-toy \
    training.num_steps=$STEPS \
    training.eval_every=$EVAL_EVERY \
    training.save_every=$SAVE_EVERY \
    training.data_parallel=true \
    training.checkpoint_dir=checkpoints/exp1-looped \
    data.batch_size=$BATCH_SIZE \
    data.seq_len=$SEQ_LEN

echo ""

# --- ChoosyTransformer ---
echo "=== [3/3] ChoosyTransformer ==="
WANDB_NAME="exp1-choosy" python -m choosy.train \
    model=choosy-toy \
    training.num_steps=$STEPS \
    training.eval_every=$EVAL_EVERY \
    training.save_every=$SAVE_EVERY \
    training.data_parallel=true \
    training.checkpoint_dir=checkpoints/exp1-choosy \
    data.batch_size=$BATCH_SIZE \
    data.seq_len=$SEQ_LEN

echo ""

# --- ChoosyTransformer (no balancing loss ablation) ---
echo "=== [bonus] ChoosyTransformer (no balancing loss) ==="
WANDB_NAME="exp1-choosy-no-balance" python -m choosy.train \
    model=choosy-toy \
    model.routing_loss_weight=0.0 \
    training.num_steps=$STEPS \
    training.eval_every=$EVAL_EVERY \
    training.save_every=$SAVE_EVERY \
    training.data_parallel=true \
    training.checkpoint_dir=checkpoints/exp1-choosy-no-balance \
    data.batch_size=$BATCH_SIZE \
    data.seq_len=$SEQ_LEN

echo ""
echo "============================================"
echo "  Experiment 1 complete!"
echo "  Results in: checkpoints/exp1-*/"
echo "  WandB: $WANDB_PROJECT"
echo "============================================"
