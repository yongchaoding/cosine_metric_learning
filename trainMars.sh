#!/bin/bash
python train_mars.py \
    --dataset_dir=../MARS/ \
    --loss_mode=cosine-softmax \
    --log_dir=./modNet/mars \
    --run_id=cosine-softmax \
    --learning_rate=1e-3
    --restore_path=./modNet/mars/cosine-softmax/model.ckpt-3116
