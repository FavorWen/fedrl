from env import Env
from rl_model import Agent
import numpy as np
import scipy
import logging

logger = logging.getLogger('')
logger.setLevel(logging.INFO)

def run_episode_no_rl(env:Env):
    env.reset()
    while True:
        env.step(None)
        obs, reward, done = env.step(None)
        if env.tick % 10 == 0:
            acc, loss = env.validate(env.testset)
            logger.info('Tick {} Test Acc: {}, Test Loss: {}'.format(env.tick, acc, loss))
        if done:
            break
        
def run_step_by_step(env:Env, agent:Agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset_light()
    obs_list.append(obs)
    action, _ = agent.sample(obs) # 采样动作
    action_list.append(action)

def run_episode(env:Env, agent:Agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action, _  = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done = env.step(action)
        reward_list.append(reward)

        spearman_co = spearman(env, agent)
        if env.tick % 10 == 0:
            acc, loss = env.validate(env.testset)
            logger.info('Tick {} Spearman co: {}, Test Acc: {}, Test Loss: {}'.format(env.tick, spearman_co, acc, loss))
        logger.info('Tick {} Spearman co: {}'.format(env.tick, spearman_co))
        if done:
            break
    return obs_list, action_list, reward_list

def evaluate(env:Env, agent:Agent, render=False):
    eval_reward = []
    for i in range(2):
        obs = env.reset_light()
        episode_reward = 0
        while True:
            action, _ = agent.sample(obs) # 选取最优动作
            obs, reward, isOver = env.step(action)
            episode_reward += reward
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def spearman(obs, env:Env, agent:Agent):
    agent.eval()
    act_prob = agent.predict(obs).squeeze().detach().cpu()
    x = [act_prob[i] for i in env.rank]
    y = [c for c in range(env.client_nums)]
    return scipy.stats.spearmanr(x,y)[0]

def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)