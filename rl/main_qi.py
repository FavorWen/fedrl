from train import *
from helper import criterion, setup_seed, ReplayMemory
from rl_model import *
import argparse
import logging
import os
import pickle

parser = argparse.ArgumentParser()
# parser.add_argument("--gpu",type=int,default='1')
parser.add_argument("--client_nums",type=int)
parser.add_argument("--participant_nums",type=int)
parser.add_argument("--seed",type=int)
parser.add_argument("--device",type=str, default='cuda')
parser.add_argument("--dataset",type=str)
parser.add_argument("--arch",type=str)
parser.add_argument("--partition",type=str, default='iid')
parser.add_argument("--optimizer",type=str, default='SGD')
parser.add_argument("--lr",type=float, default=0.01)
parser.add_argument("--epoch",type=int, default=1)
parser.add_argument("--rl_ddl",type=int, default=100)
parser.add_argument("--batch_size",type=int, default=32)
args = parser.parse_args()

client_nums = args.client_nums
participant_nums = args.participant_nums
seed = args.seed
device = args.device
dataset_name = args.dataset
arch_name = args.arch
partition = args.partition
optimizer = args.optimizer
lr = args.lr
epoch = args.epoch
rl_ddl = args.rl_ddl
batch_size = args.batch_size


logger = logging.getLogger('')
logger.setLevel(logging.INFO)

setup_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger.info('Settings: {}'.format(args))
# 创建环境
env = Env(client_nums+1, arch_name, client_nums, participant_nums, dataset_name, partition, seed, device, criterion, optimizer="SGD", lr=lr, epoch=epoch, rl_ddl = rl_ddl, batch_size=batch_size)



env.reset()

result = {
    'seed': seed,
    'dataset': dataset_name,
    'arch': arch_name,
    'partition': partition,
    'lr': lr,
    'epoch': epoch,
    'optimizer': optimizer,
    'batch_size': batch_size,
    'rank': env.rank,
    'client_nums': client_nums,
    'participant_nums':participant_nums,
    'train_round': 1,
    'rl_ddl': rl_ddl,
    'logs' : [],
}

for i in range(1):
    logger.info("Train Round {}".format(i+1))
    obs = env.reset_light()
    # obs_list, action_list, reward_list = [], [], []
    while True:
        next_obs, reward, done, participants, acc  = env.step()
        result['logs'].append((participants, acc))
        
        logger.info('Tick {} Accuracy: {}, Participants: {}'.format(env.tick, acc, participants))

        if env.tick % 10 == 0:
            acc, loss = env.validate(env.testset)
            logger.info('Tick {} Validate Acc: {}, Validate Loss: {}'.format(env.tick, acc, loss))

        if done:
            break

logfile = 'train_data/qi_data_seed{}_arch{}_dataset{}_nums{}_part{}'.format(seed, arch_name, dataset_name, client_nums, partition)
with open(logfile, 'wb') as f:
    pickle.dump(result, f)



# 保存模型到文件 ./model.ckpt
# agent.save('./model.ckpt')
# nohup python main_qi.py --client_nums 100  --participant_nums 10 --dataset MNIST --arch RESNET18 --partition iid --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 --seed 1 > train_data/qi_data_seed1.log 2>&1 &
# nohup python main_qi.py --client_nums 25 --participant_nums 5 --seed 2 --dataset CIFAR10 --arch CNN --partition iid --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > train_data/qi_data_seed2.log 2>&1 &
# python main_qi.py --client_nums {100, 25} --participant_nums {10, 5} --seed {int} --dataset {CIFAR10, MNIST, FMNIST} --arch {CNN, RESNET18, MOBILENETV2} --partition {iid, quantity-skew, label-skew} 