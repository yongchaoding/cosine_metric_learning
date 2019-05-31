#!/bin/bash
python train_mars.py \
    --mode=eval \
    --dataset_dir=../MARS \
    --loss_mode=cosine-softmax \
    --log_dir=./netOut/mars/ \
    --run_id=cosine-softmax \
    --eval_log_dir=./eval_output/mars
