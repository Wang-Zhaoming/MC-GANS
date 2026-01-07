#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--dataset CUB \
--class_embedding '10' \
--attSize 1024 \
--nz0 1024 \
--lr 1e-4 \
--syn_num 50 \
--lambda_1 2 \


