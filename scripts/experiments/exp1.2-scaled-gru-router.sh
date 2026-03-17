#!/bin/bash
# Experiment 1.2: Scaled-up ChoosyTransformer with GRU routing
#
# Full-size models (d=512, 8 heads, d_ff=2048) with the recurrent
# content vector (GRU) for routing context. Tests whether the GRU
# gives the router enough signal to meaningfully specialize layers.
#
# Grid:
#   Balance weights: {0.01, 0.1, 0.25, 1.0}
#   Learning rates:  {3e-4, 6e-4, 1e-3}
#   LR schedules:    {constant, cosine w/ 500-step warmup}
#   Seeds:           {42, 137}
#
# Also runs baselines (Regular, Looped) at full scale for comparison.
#
# 4 × 3 × 2 = 24 choosy configs × 2 seeds = 48 choosy runs (~12h)
# + 2 baselines × 3 LR × 2 sched × 2 seeds = 24 baseline runs (~4h)
# Baselines are faster, so overlap is fine. Total ~12h.
#
set -e

unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,4,5

STEPS=15000
EVAL_EVERY=500
SAVE_EVERY=5000
BATCH_SIZE=32
SEQ_LEN=256
SEEDS=(42 137)
BALANCE_WEIGHTS=(0.01 0.1 0.25 1.0)
LRS=(3e-4 6e-4 1e-3)
SCHEDULES=(constant cosine)
WARMUP=500

export WANDB_PROJECT="choosy-experiments"

echo "============================================"
echo "  Experiment 1.2: Scaled GRU Router"
echo "  d_model=512, 8 heads, 15K steps"
echo "  Seeds: ${SEEDS[*]}"
echo "  Balance: ${BALANCE_WEIGHTS[*]}"
echo "  LRs: ${LRS[*]}"
echo "  Schedules: ${SCHEDULES[*]}"
echo "============================================"
echo ""

run_model() {
    local name=$1
    local model_cfg=$2
    local seed=$3
    local lr=$4
    local sched=$5
    local extra_args="${6:-}"

    local sched_tag=$sched
    local warmup_arg=""
    if [ "$sched" = "cosine" ]; then
        warmup_arg="training.warmup_steps=$WARMUP"
        sched_tag="cos"
    fi

    local run_name="exp1.2-${name}-lr${lr}-${sched_tag}-s${seed}"
    local ckpt_dir="checkpoints/exp1.2-${name}-lr${lr}-${sched_tag}-s${seed}"

    echo "=== ${run_name} ==="
    WANDB_NAME="$run_name" \
    WANDB_TAGS="exp1.2,enwik8,full-scale,gru-router,seed-${seed},${name},lr-${lr},${sched_tag}" \
    python -m choosy.train \
        model=$model_cfg \
        training.num_steps=$STEPS \
        training.eval_every=$EVAL_EVERY \
        training.save_every=$SAVE_EVERY \
        training.data_parallel=true \
        training.checkpoint_dir=$ckpt_dir \
        training.seed=$seed \
        training.lr_schedule=$sched \
        $warmup_arg \
        data.batch_size=$BATCH_SIZE \
        data.seq_len=$SEQ_LEN \
        optimizer.learning_rate=$lr \
        $extra_args
    echo ""
}

# --- Part A: Baselines at full scale ---
echo "########## BASELINES ##########"
echo ""
for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
        for SCHED in "${SCHEDULES[@]}"; do
            run_model "regular" "regular" $SEED $LR $SCHED
            run_model "looped"  "looped"  $SEED $LR $SCHED
        done
    done
done

# --- Part B: Choosy sweep ---
echo "########## CHOOSY SWEEP ##########"
echo ""
for SEED in "${SEEDS[@]}"; do
    for W in "${BALANCE_WEIGHTS[@]}"; do
        for LR in "${LRS[@]}"; do
            for SCHED in "${SCHEDULES[@]}"; do
                local_warmup=""
                if [ "$SCHED" = "cosine" ]; then
                    local_warmup="training.warmup_steps=$WARMUP"
                fi

                run_model "choosy-bal${W}" "choosy" $SEED $LR $SCHED \
                    "model.routing_loss_weight=${W} model.router_temp_start=2.0 model.router_temp_end=0.5"
            done
        done
    done
done

echo "============================================"
echo "  Experiment 1.2 complete!"
echo "  Results in: checkpoints/exp1.2-*/"
echo "============================================"
