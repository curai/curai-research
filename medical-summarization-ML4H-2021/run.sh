PSEUDO_LABEL_PATH="<path to pseudolabel file>"
NUM_SAMPLES=1000
EPOCHS=6
SEED=0
BATCH_SIZE=8
ACCUMULATE_GRAD_BATCHES=16
PRETRAINED_MODEL=google/pegasus-cnn_dailymail
ITER=0
FROM_PSEUDO_LABEL="n"
PSEUDO_LABEL_STRATEGY="sum_log_logits"
SUM_LOG_LOGITS_THRESHOLD=-0.1
THRESHOLD_WINDOW_LOW=0.95
THRESHOLD_WINDOW_HIGH=1.0
DROPOUT=0.1

function _run_exp() {
    python train_pegasus.py \
    --pretrained-model ${PRETRAINED_MODEL} --num-samples ${NUM_SAMPLES} --epochs ${EPOCHS} \
    --exp-name ${EXPERIMENT_NAME} --seed ${SEED} --batch-size ${BATCH_SIZE} --accumulate-grad-batches ${ACCUMULATE_GRAD_BATCHES} \
    --from-pseudo-label ${FROM_PSEUDO_LABEL} --pseudo-label-strategy ${PSEUDO_LABEL_STRATEGY} \
    --sum-log-logits-threshold ${SUM_LOG_LOGITS_THRESHOLD} --iteration ${ITER} --sll-human-threshold-window ${THRESHOLD_WINDOW_LOW} ${THRESHOLD_WINDOW_HIGH} \
    --dropout $DROPOUT --sparse-training
}

function _run_exp_w_unl() {
    python train_pegasus.py \
    --pretrained-model ${PRETRAINED_MODEL} --num-samples ${NUM_SAMPLES} --epochs ${EPOCHS} --track \
    --exp-name ${EXPERIMENT_NAME} --seed ${SEED} --batch-size ${BATCH_SIZE} --accumulate-grad-batches ${ACCUMULATE_GRAD_BATCHES} \
    --from-pseudo-label ${FROM_PSEUDO_LABEL} --pseudo-label-strategy ${PSEUDO_LABEL_STRATEGY} \
    --sum-log-logits-threshold ${SUM_LOG_LOGITS_THRESHOLD} --iteration ${ITER} --label-unlabeled-set --filter-unlabeled-by-ratio 4 \
    --pseudo-label-path "<path to pseudolabel file>" \
    --dropout $DROPOUT
}

function run_exp() {
    SEED=$1
    NUM_SAMPLES=$2
    EXPERIMENT_NAME="Baseline-Sparse-Training-N-${2}"
    FROM_PSEUDO_LABEL=$3
    _run_exp
}

function run_unl_exp() {
    SEED=$1
    EXPERIMENT_NAME="PL-w-unlabeled-percentage&ratio-thresholding-${SUM_LOG_LOGITS_THRESHOLD}-N-${2}"
    NUM_SAMPLES=$2
    FROM_PSEUDO_LABEL=$3
    _run_exp_w_unl
}

## Baseline Experiments

# SUM_LOG_LOGITS_THRESHOLD=0.01
# PSEUDO_LABEL_STRATEGY="sum_log_logits"
# ITER=0
# seed=0
# # samples=100
# # run_exp $seed $samples "n"

# for samples in 6400; do
#     for threshold in 0.0; do
#         SUM_LOG_LOGITS_THRESHOLD=$threshold
#         PSEUDO_LABEL_STRATEGY="sum_log_logits"
#         ITER=0
#         run_exp 0 $samples "n"
#         # ITER=1
#         # run_unl_exp 0 $samples "y"
#         # # ITER=2
#         # # run_unl_exp 0 $samples "y"
#         # # ITER=3
#         # # run_unl_exp 0 $samples "y"
#     done
# done



for checkpoint_strategy in "google/pegasus-cnn_dailymail"; do
    for samples in 100 500 750 1000 1250; do 
        for do in 0.1 0.5; do

            DROPOUT=$do
            
            for pl_thresholds in 0.0,"0%" 0.01,"top-1%"; do
                
                IFS=',' read pl_threshold pl_region <<< "${pl_thresholds}"

                SUM_LOG_LOGITS_THRESHOLD=$pl_threshold

                for thresholds in 0.99,1.0,"bottom" 0.495,0.505,"middle" 0.01,0.01,"random"; do
                    IFS=',' read low high hl_region <<< "${thresholds}"

                    THRESHOLD_WINDOW_LOW=$low
                    THRESHOLD_WINDOW_HIGH=$high
                    PRETRAINED_MODEL=google/pegasus-cnn_dailymail
                    PSEUDO_LABEL_STRATEGY="sum_log_logits"
                    ITER=0
                    seed=0

                    EXPERIMENT_NAME="PL-${pl_region}-HL-${hl_region}-1%-N-${samples}-ITER-${ITER}"
                    run_exp $seed $samples "n"

                    for i in 1 2 3; do
                        ITER=$i
                        PRETRAINED_MODEL=$checkpoint_strategy
                        EXPERIMENT_NAME="PL-${pl_region}-HL-${hl_region}-1%-N-${samples}-ITER-${ITER}"
                        run_exp $seed $samples "y"
                    done
                done
            done
        done
    done
done

