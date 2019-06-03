#!/bin/bash
python train_mars.py \
    --dataset_dir=../MARS/ \
    --loss_mode=cosine-softmax \
    --log_dir=./mobileV2CascadeFixedNet/mars \
    --run_id=cosine-softmax \
    --learning_rate=1e-3
    --restore_path=./nets/mobilenet/model/mobilenet_v2_1.0_96.ckpt
