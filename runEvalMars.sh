#!/bin/bash
#KIND=mobileV2Cascade
KIND=modFull

python train_mars.py \
    --mode=eval \
    --dataset_dir=../MARS \
    --loss_mode=cosine-softmax \
    --log_dir=./${KIND}Net/mars/ \
    --run_id=cosine-softmax \
    --eval_log_dir=./${KIND}Eval/mars
