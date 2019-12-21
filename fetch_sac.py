"""
env: FetchReach-v1
fetch.old_m: old folder with model at 0.65 success rate
fetch.fetch_env: dummy wrapper for "FetchReach-v1", concatenate obs
expert_trajs1, expert_trajs2: 20 trajs collected by SAC (0.99 success rate)
                              each traj is 50 time steps
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as L
import copy
from envs.cyl import DummyCylEnv, CylEnv
from baselines.common.tf_util import function
from kick_ass_utils import euclidean_loss, MSE, eval_class_acc, init_tf
from policies import BCPol, BCPolWithEncoder, RandomPol, GaussianTrajectorEncoder, SACPol
from buffers import UnsupervisedBuffer
from envs.point import PointEnv
from envs.points_plus_distractors import PointDistractorEnv

import gym
from fetch.fetch_env import FetchEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

import os
from stable_baselines.bench import Monitor

def fetch_sac():
    # SAC on FetchReach
    from stable_baselines import PPO2
    from stable_baselines import SAC
    from stable_baselines.sac.policies import MlpPolicy as Spol

#     env = gym.make("FetchReach-v1")
    env = FetchEnv()
#     n_cpu = 8
#     state_size = 16
#     ation_size = 4
#     env = SubprocVecEnv([lambda: FetchEnv() for i in range(n_cpu)])
    log_dir = "tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir, allow_early_resets=True)

    model = SAC(Spol, env, verbose=1, learning_starts=700)
    model.learn(total_timesteps=int(6e5))
    model.save('fetch_sac')

    env.no_auto_reset()
    save_trajs = True
    if save_trajs:
        all_states = []
        all_actions = []
        n_trajs_to_save = 20
        for traj_idx in range(n_trajs_to_save):
            this_traj_states, this_traj_actions = [], []
            obs = env.reset()
            score = 0
            for i in range(50):
                action, _states = model.predict(obs)
                this_traj_actions.append(copy.deepcopy(action))
                this_traj_states.append(copy.deepcopy(obs))
                obs, rewards, dones, info = env.step(action)
                score += rewards
                # env.render()
            print(F"Traj score: {score}")
            this_traj_states = np.array(this_traj_states)
            this_traj_actions = np.array(this_traj_actions)
            all_states.append(this_traj_states)
            all_actions.append(this_traj_actions)

        d = dict()
        d['expert_obs'] = all_states
        d['expert_acs'] = all_actions
        np.savez("fetch_sac.npz", **d)

def continue_train():
    from stable_baselines import PPO2
    from stable_baselines import SAC
    from stable_baselines.sac.policies import MlpPolicy as Spol

    log_dir = "tmp/gym_cont/"
    os.makedirs(log_dir, exist_ok=True)

    env = FetchEnv()
    env = Monitor(env, log_dir, allow_early_resets=True)

    model = SAC.load("fetch_sac")
    model.set_env(env)
    model.learn(total_timesteps=int(1.5e5))
    model.save('fetch_sac_cont')

def fetch_eval():
    # FetchReach demo / collecting trajs
    from stable_baselines import PPO2
    from stable_baselines import SAC
    from stable_baselines.sac.policies import MlpPolicy as Spol

    env = FetchEnv()
    env.no_auto_reset()
    model = SAC.load("fetch_sac")

    # Evaluate the agent
    save_trajs = True
    if save_trajs:
        all_states = []
        all_actions = []
        n_trajs_to_save = 20
        for traj_idx in range(n_trajs_to_save):
            this_traj_states, this_traj_actions = [], []
            obs = env.reset()
            score = 0
            for i in range(50):
                action, _states = model.predict(obs)
                this_traj_actions.append(copy.deepcopy(action))
                this_traj_states.append(copy.deepcopy(obs))
                obs, rewards, dones, info = env.step(action)
                score += rewards
                # env.render()
            print(F"Traj score: {score}", "Success?", info.get('is_success', False))
            this_traj_states = np.array(this_traj_states)
            this_traj_actions = np.array(this_traj_actions)
            all_states.append(this_traj_states)
            all_actions.append(this_traj_actions)

        d = dict()
        d['expert_obs'] = all_states
        d['expert_acs'] = all_actions
        np.savez("fetch.npz", **d)


def main():
    sess = tf.Session()
    with sess.as_default():
        fetch_sac()
        # fetch_eval()
        # continue_train()



if __name__ == "__main__":
    main()
