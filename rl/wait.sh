#!/bin/bash

sleep 40m

python main_rpm.py --history_dim 4 --client_nums 25 --participant_nums 5 --seed 4 --dataset CIFAR10 --arch CNN --partition iid --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > res-6_redo-seed4.log
