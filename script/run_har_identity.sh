#!/bin/bash


python3 run.py \
    --ret_hist_path=data/sample.csv \
    --train_start_step=0 \
    --train_steps=300 \
    --eval_steps=60 \
    --test_steps=120 \
    --n_iters=0 \
    --predmodel=har \
    --predmodel_dist=normal \
    --flow=identity \
    --dim=1 \
    --phi_dim=0 \
    --lag=22 \
    --log_leve=INFO \