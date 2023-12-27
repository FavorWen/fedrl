from train import *
from helper import criterion, setup_seed
from rl_model import *
import argparse
import logging
import os

# nohup python main.py --history_dim 2 --client_nums 100 --participant_nums 10 --seed 0 --dataset CIFAR10 --arch CNN --partition iid --optimizer SGD --lr 0.1 --epoch 1 --rl_ddl 200 --batch_size 32 > out2.log 2>&1 &

parser = argparse.ArgumentParser()
# parser.add_argument("--gpu",type=int,default='1')
parser.add_argument("--history_dim",type=int, default=5)
parser.add_argument("--client_nums",type=int)
parser.add_argument("--participant_nums",type=int)
parser.add_argument("--seed",type=int)
parser.add_argument("--device",type=str, default='cuda')
parser.add_argument("--dataset",type=str)
parser.add_argument("--arch",type=str)
parser.add_argument("--partition",type=str, default='iid')
parser.add_argument("--optimizer",type=str, default='SGD')
parser.add_argument("--lr",type=float, default=0.1)
parser.add_argument("--epoch",type=int, default=1)
parser.add_argument("--rl_ddl",type=int, default=100)
parser.add_argument("--batch_size",type=int, default=32)
args = parser.parse_args()

client_nums = args.client_nums
participant_nums = args.participant_nums
obs_dim = args.history_dim * (args.client_nums + 1)
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


act_dim = client_nums

logger = logging.getLogger('')
logger.setLevel(logging.INFO)

setup_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 创建环境
env = Env(obs_dim, arch_name, client_nums, participant_nums, dataset_name, partition, seed, device, criterion, optimizer="SGD", lr=0.01, epoch=1, rl_ddl = rl_ddl, batch_size=32)

logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

# 根据parl框架构建agent
model = Model(obs_dim, act_dim).to(device)
alg = PolicyGradient(model, lr)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim, participant_nums=participant_nums, lr=lr)

# 加载模型
# if os.path.exists('./model.ckpt'):
#     agent.restore('./model.ckpt')
#     run_episode(env, agent, train_or_test='test', render=True)
#     exit()

for i in range(50):
    logger.info("Train Round {}".format(i+1))
    run_episode_no_rl(env)

# 保存模型到文件 ./model.ckpt
# agent.save('./model.ckpt')

# nohup python main_no_rl.py --history_dim 2 --client_nums 100 --participant_nums 10 --seed 0 --dataset CIFAR10 --arch RESNET18 --partition iid --optimizer SGD --lr 0.1 --epoch 1 --rl_ddl 200 --batch_size 32 > out_no_rl2-resnet-1.5.log 2>&1 &

