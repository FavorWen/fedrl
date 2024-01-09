#!/bin/bash

sleep 35m

python main_rpm.py --history_dim 8 --client_nums 100 --participant_nums 10 --seed 3 --dataset CIFAR10 --arch CNN --partition iid --optimizer SGD  --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > dev-100-3-seed3.log

python main_rpm.py --history_dim 8 --client_nums 100 --participant_nums 10 --seed 2 --dataset CIFAR10 --arch CNN --partition iid --optimizer SGD  --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > dev-100-3-seed2.log