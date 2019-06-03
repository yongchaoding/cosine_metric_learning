#!/bin/bash
python train_mars.py \
    --dataset_dir=../MARS/ \
    --loss_mode=cosine-softmax \
    --log_dir=./mobileV2CascadeNet/mars \
    --run_id=cosine-softmax \
    --learning_rate=1e-3
#    --restore_path=./mobileV2Net/mars/cosine-softmax/model.ckpt-52883
