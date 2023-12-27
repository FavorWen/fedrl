from train import *
from helper import criterion, setup_seed
from rl_model import *
import argparse
import logging
import os

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
alg = PolicyGradient(model, lr, device)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim, participant_nums=participant_nums, lr=lr)

# 加载模型
# if os.path.exists('./model.ckpt'):
#     agent.restore('./model.ckpt')
#     run_episode(env, agent, train_or_test='test', render=True)
#     exit()

env.reset()
for i in range(50):
    logger.info("Train Round {}".format(i+1))
    obs = env.reset_light()
    # obs_list, action_list, reward_list = [], [], []
    while True:
        obs_list, action_list, reward_list = [], [], []

        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done = env.step(action)
        reward_list.append(reward)

        spearman_co = spearman(env, agent)
        if env.tick % 10 == 0:
            acc, loss = env.validate(env.testset)
            logger.info('Tick {} Spearman co: {}, Test Acc: {}, Test Loss: {}'.format(env.tick, spearman_co, acc, loss))
        logger.info('Tick {} Spearman co: {}'.format(env.tick, spearman_co))

        batch_obs = obs_list
        batch_action = action_list
        batch_reward = calc_reward_to_go(reward_list, gamma=0.0)
        agent.learn(batch_obs, batch_action, batch_reward)
        if done:
            break

    spearman_co = spearman(env, agent)
    if (i + 1) % 100 == 0:
        total_reward = evaluate(env, agent, render=False) # render=True 查看渲染效果，需要在本地运行，AIStudio无法显示
        logger.info('Test reward: {}, Spearman co: {}'.format(total_reward, spearman_co))


# 保存模型到文件 ./model.ckpt
# agent.save('./model.ckpt')
# nohup python main_step_by_step_rl.py --history_dim 2 --client_nums 100 --participant_nums 10 --seed 0 --dataset CIFAR10 --arch CNN --partition iid --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > out_s_by_s_100.log 2>&1 &

