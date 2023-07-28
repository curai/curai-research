#WANDB_PROJECT=history_taking_reranking WANDB_JOB_TYPE=Pretrain_S4 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true python train_S4_lm.py
#WANDB_PROJECT=history_taking_reranking WANDB_JOB_TYPE=Pretrain_S4 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true python s4_lm_test.py
WANDB_PROJECT=history_taking_reranking WANDB_JOB_TYPE=S4CrossEncoder HYDRA_FULL_ERROR=1 python train_S4CrossEncoder.py --eval --track --cuda
#WANDB_PROJECT=history_taking_reranking WANDB_JOB_TYPE=S4Ranker WANDB_CACHE_DIR=./cache/ python S4_autoregressive_ranker.py --track --cuda --eval --train