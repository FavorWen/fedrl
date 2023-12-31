from train import *
from helper import criterion, setup_seed, ReplayMemory, LogSaver
from rl_model import *
import argparse
import logging
import os
import pickle

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
parser.add_argument("--lr",type=float, default=0.01)
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger.info('Settings: {}'.format(args))
log_saver = LogSaver()

# 创建环境
env = Env(obs_dim, arch_name, client_nums, participant_nums, dataset_name, partition, seed, device, criterion, log_saver=log_saver, optimizer="SGD", lr=lr, epoch=epoch, rl_ddl = rl_ddl, batch_size=batch_size)


logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))


# MEMORY_SIZE = 16
# MEMORY_BATCHSIZE = 8
# LEARN_FREQ = 5
# MEMORY_WARMUP_SIZE = 6
# MULTI_LEARN_ = 2

# MEMORY_SIZE = 32
# MEMORY_BATCHSIZE = 16
# LEARN_FREQ = 5
# MEMORY_WARMUP_SIZE = 6
# MULTI_LEARN_ = 2

MEMORY_SIZE = 64
MEMORY_BATCHSIZE = 32
LEARN_FREQ = 5
MEMORY_WARMUP_SIZE = 6
MULTI_LEARN_ = 2
rpm = ReplayMemory(MEMORY_SIZE)
logger.info('Settings: {}'.format(args))
logger.info('Settings: MEMORY_SIZE {}, MEMORY_BATCHSIZE {}, LEARN_FREQ {}, MEMORY_WARMUP_SIZE {}, MULTI_LEARN_ {}'.format(MEMORY_SIZE,MEMORY_BATCHSIZE, LEARN_FREQ, MEMORY_WARMUP_SIZE, MULTI_LEARN_))

EXPERIENCE_INFO = """
out_rpm_seed_2_1.log为基准，测试seed=4时，multi_learn为3，网络隐藏层20倍，优化器为SGD,学习率为0.001，weight_decay=10e-4,观察结果作为其他实验基准
网络宽度固定为1024 * 3，1024 * 10 * 3， 1024 * 10 * 3
采用batch方法进行训练，cost计算方式先sum再除sample个数
noise thread为标准情况
"""

logger.info(EXPERIENCE_INFO)

# nohup python main_rpm.py --history_dim 2 --client_nums 25 --participant_nums 5 --seed 4 --dataset CIFAR10 --arch CNN --partition iid --optimizer SGD --lr 0.001 --epoch 1 --rl_ddl 200 --batch_size 32 > out_rpm_seed4_8.log 2>&1 &


# 根据parl框架构建agent
model = Model(obs_dim, act_dim).to(device)
alg = PolicyGradient(model, lr, device)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim, participant_nums=participant_nums, client_nums=client_nums, lr=lr)

# 加载模型
# if os.path.exists('./model.ckpt'):
#     agent.restore('./model.ckpt')
#     run_episode(env, agent, train_or_test='test', render=True)
#     exit()

experience_path = "train_data/seed{}_arch{}_dataset{}_nums{}_select{}_part{}.elog".format(seed, arch_name, dataset_name, client_nums, participant_nums, partition)

env.reset()


for i in range(5):
    logger.info("Train Round {}".format(i+1))
    obs = env.reset_light()
    # obs_list, action_list, reward_list = [], [], []
    while True:
        action, act_prob = agent.sample(obs) # 采样动作
        next_obs, reward, done, _, acc  = env.step(action)
        log_saver.updateLog((action, act_prob, reward, acc))
        rpm.append((obs, action, reward, next_obs, done))

        if (len(rpm) >= MEMORY_WARMUP_SIZE) and ((env.tick+1) % LEARN_FREQ == 0):
            for k in range(MULTI_LEARN_):
                (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(MEMORY_BATCHSIZE)
                # agent.learn(batch_obs, batch_action, batch_reward)
                agent.learn_by_batch(batch_obs, batch_action, batch_reward)
            spearman_co = spearman(env, agent)
            logger.info('Tick {} Spearman co: {}'.format(env.tick, spearman_co))
        if len(rpm) < MEMORY_WARMUP_SIZE:
            logger.info('Tick {} Accumulating memory'.format(env.tick))

        if env.tick % 10 == 0:
            acc, loss = env.validate(env.testset)
            spearman_co = spearman(env, agent)
            logger.info('{} Tick {} Spearman co: {}, Test Acc: {}, Test Loss: {}'.format(i+1, env.tick, spearman_co, acc, loss))

        if done:
            break
    log_saver.flush()

    with open(experience_path, 'wb') as f:
        pickle.dump(log_saver, f)

    spearman_co = spearman(env, agent)
    if (i + 1) % 100 == 0:
        total_reward = evaluate(env, agent, render=False) # render=True 查看渲染效果，需要在本地运行，AIStudio无法显示
        logger.info('Test reward: {}, Spearman co: {}'.format(total_reward, spearman_co))




# 保存模型到文件 ./model.ckpt
# agent.save('./model.ckpt')
# nohup python main_rpm.py --history_dim 10 --client_nums 25 --participant_nums 5 --seed 5 --dataset CIFAR10 --arch RESNET18 --partition iid --optimizer SGD --lr 0.01 --epoch 1 --rl_ddl 200 --batch_size 32 > out_rpm_seed5_1.log 2>&1 &
