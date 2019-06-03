#!/bin/bash
python train_market1501.py \
    --dataset_dir=../Market/Market-1501-v15.09.15/ \
    --loss_mode=cosine-softmax \
    --log_dir=./mobileV2Net/market1501 \
    --run_id=cosine-softmax \
    --learning_rate=1e-3
#    --restore_path=./mobileV2Net/mars/cosine-softmax/model.ckpt-52883
