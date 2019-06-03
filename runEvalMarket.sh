#!/bin/bash
KIND=mobileV2

python train_market1501.py \
    --mode=eval \
    --dataset_dir=../Market/Market-1501-v15.09.15/ \
    --loss_mode=cosine-softmax \
    --log_dir=./${KIND}Net/market1501/ \
    --run_id=cosine-softmax \
    --eval_log_dir=./${KIND}Eval/market1501/
