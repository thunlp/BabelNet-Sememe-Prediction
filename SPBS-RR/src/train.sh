#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 \
python main.py \
--data_dir ../../data-noun/ \
--embedding_dim 800 \
--margin_value 4 \
--batch_size 16384 \
--learning_rate 0.01 \
--n_generator 24 \
--n_rank_calculator 24 \
--eval_freq 200 \
--max_epoch 20000
