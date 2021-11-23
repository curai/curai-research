WANDB_PROJECT=medical-conversations python dialogpt.py \
    --per_gpu_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --track \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --eval_steps 0 \
    --dataset_artifact medical-conversations/dialogpt:baseline_short
