#!/usr/bin/env bash
c=0
#for d in oos stackoverflow banking
nohup python run.py \
--task_name clinc150 \
--shot 0.1 \
--known_ratio 0.75 \
--seed 9 \
--rec_drop 0.3 \
--rec_num 15 \
--batch_size 8 \
--val_batch_size 32 \
--num_train_epochs 1000 \
--learning_rate 1e-5 \
--data_dir "pathTodata" \
--convex \
--train_rec \
1>logs 2>&1 &

tail -f logs
