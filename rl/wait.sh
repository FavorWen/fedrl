#!/bin/bash


partition=(iid)
datasets=(CIFAR10 CIFAR100)
archs=(CNN RESNET18)
hdims=(8)
cn=(100)
pn=(10)
seeds=(5 4 3 2)
# partition=(iid label-skew)
# datasets=(FMNIST MNIST CIFAR10 CIFAR100)
# archs=(MLP MLP CNN RESNET18)
# hdims=(8 4 4)
# cn=(100 25 5)
# pn=(10 5 3)
# seeds=(5 4 3 2)

for part in "${partition[@]}"; do
    for((i=0;i<2;i++))
    do
        dataset=${datasets[$i]}
        arch=${archs[$i]}
        for((j=0;j<2;j++))
        do
            hdim=${hdims[$j]}
            c=${cn[$j]}
            p=${pn[$j]}
            for seed in "${seeds[@]}"; do
                python main_rpm.py --history_dim $hdim --client_nums $c --participant_nums $p --seed $seed --dataset $dataset --arch $arch --partition $part --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > res-${arch}-${c}-seed${seed}.log &
                echo "nohup python main_rpm.py --history_dim $hdim --client_nums $c --participant_nums $p --seed $seed --dataset $dataset --arch $arch --partition $part --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > res-${arch}-${c}-seed${seed}.log &"
            done
            wait
            echo ""
        done
    done
done


# sleep 40m

# python main_rpm.py --history_dim 4 --client_nums 25 --participant_nums 5 --seed 4 --dataset CIFAR10 --arch CNN --partition iid --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > res-6_redo-seed4.log
