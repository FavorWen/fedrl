#!/bin/bash
seed=$1

echo seed=$seed

# datasets=('CIFAR10' 'MNIST' 'FMNIST')
# arches=('CNN' 'RESNET18' 'MOBILENETV2')
# partitions=('iid' 'label-skew' 'quantity-skew')
datasets=('CIFAR10')
arches=('CNN')
partitions=('iid')
nums=25
partin=5
# 25 - 5
# 100 - 20

for dataset in "${datasets[@]}"; do
    for arch in "${arches[@]}"; do
        for partition in "${partitions[@]}"; do
            log_path="train_log/seed${seed}_arch${arch}_dataset${dataset}_nums${nums}_part${partition}.run.clip.log"
            python main_rpm.py --history_dim=4 --client_nums $nums --participant_nums $partin --seed $seed --dataset $dataset --arch $arch --partition $partition --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > $log_path 2>&1

            python main_rpm.py --history_dim=4 --client_nums 50 --participant_nums 10 --seed $seed --dataset $dataset --arch $arch --partition $partition --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > $log_path 2>&1

            python main_rpm.py --history_dim=4 --client_nums $nums --participant_nums $partin --seed 4 --dataset $dataset --arch $arch --partition $partition --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > $log_path 2>&1

            python main_rpm.py --history_dim=4 --client_nums 50 --participant_nums 10 --seed 4 --dataset $dataset --arch $arch --partition $partition --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > $log_path 2>&1

            python main_rpm.py --history_dim=4 --client_nums $nums --participant_nums $partin --seed 5 --dataset $dataset --arch $arch --partition $partition --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > $log_path 2>&1

            python main_rpm.py --history_dim=4 --client_nums 50 --participant_nums 10 --seed 5 --dataset $dataset --arch $arch --partition $partition --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > $log_path 2>&1



            # python main_rpm.py --history_dim=8 --client_nums $nums --participant_nums $partin --seed $seed --dataset $dataset --arch $arch --partition $partition --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > $log_path 2>&1
        done
    done
done


# python main_rpm.py --history_dim 10 --client_nums 25 --participant_nums 5 --seed 5 --dataset CIFAR10 --arch RESNET18 --partition iid --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > out_rpm_seed5_1.log