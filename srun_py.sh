#!/usr/bin/env sh

# srun \
#     -p llmeval \
#     --quotatype=auto \
#     --job-name=sh \
#     --gres=gpu:8 \
#     --ntasks=8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=8 \
#     --kill-on-bad-exit=1 \
#     --pty bash

srun \
    --partition=llmeval \
    --quotatype=auto \
    --job-name=sh \
    --gres=gpu:4 \
    --ntasks=4 \
    --ntasks-per-node=4 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    $1
