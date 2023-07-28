#WANDB_PROJECT=history_taking_reranking WANDB_JOB_TYPE=Pretrain_S4 accelerate launch --mixed_precision=fp16 --num_processes=4 train_S4_lm.py 
#WANDB_PROJECT=history_taking_reranking WANDB_JOB_TYPE=S4CrossEncoder HYDRA_FULL_ERROR=1 python S4_autoregressive_ranker.py --train --eval --track --cuda
WANDB_PROJECT=history_taking_reranking WANDB_JOB_TYPE=S4Ranker WANDB_CACHE_DIR=./cache/ python S4_global_reranker.py --track --cuda --eval --train
#WANDB_PROJECT=history_taking_reranking WANDB_JOB_TYPE=S4CrossEncoder WANDB_CACHE_DIR=./cache/ python S4_cross_encoder.py --track --cuda --eval --train