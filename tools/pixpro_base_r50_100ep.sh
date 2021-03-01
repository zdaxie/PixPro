#!/bin/bash

set -e
set -x

data_dir="./data/imagenet/"
output_dir="./output/pixpro_base_r50_100ep"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 12348 --nproc_per_node=8 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --zip --cache-mode no \
    --crop 0.08 \
    --aug BYOL \
    --dataset ImageNet \
    --batch-size 128 \
    \
    --model PixPro \
    --arch resnet50 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 100 \
    --amp-opt-level O1 \
    \
    --save-freq 10 \
    --auto-resume \
    \
    --pixpro-p 2 \
    --pixpro-momentum 0.99 \
    --pixpro-pos-ratio 0.7 \
    --pixpro-transform-layer 1 \
    --pixpro-ins-loss-weight 0. \
