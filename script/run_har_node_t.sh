#!/bin/bash

python3 run.py \
    --ret_hist_path=data/sample.csv  \
    --train_start_step=0 \
    --train_steps=300 \
    --eval_steps=60 \
    --test_steps=120 \
    --n_iters=200 \
    --predmodel=har \
    --eval_interval=5 \
    --predmodel_dist=t \
    --flow=node \
    --dim=1 \
    --divergence_fn=approximate \
    --d_hidden=4,4 \
    --lr=0.005 \
    --nonlinearity=swish \
    --optimizer=adam \
    --time_length=1.0 \
    --weight_decay=1e-05 \
    --lag=22 \
    --seed=0 \
    --phi_dim=0 \
    #--use_wandb \
    #--save_dir=model/ \

    