
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
from prettytable import PrettyTable


def rollout(pol, env, horizon=700, render=False):
    states, actions, rews = [], [], []
    s = env.reset()
    for i in range(horizon):
        a, rescaled_a = pol.act(s)
        states.append(s)
        actions.append(a)
        s, rew, done, _ = env.step(rescaled_a)
        rews.append(rew)
        if render:
            env.render()
    return states, actions, rews


def rollout_eval(pol, env, horizon=700, render=False):
    states, actions, rews = [], [], []
    s = env.reset()
    for i in range(horizon):
        a, rescaled_a = pol.act(s, deterministic=True)
        states.append(s)
        actions.append(a)
        s, rew, done, _ = env.step(rescaled_a)
        rews.append(rew)
        if render:
            env.render()
    return rew


def run():
    traj_horizon = 700
    n_trajs_per_iteration = 1
    env = PointEnv(horizon=traj_horizon)
    #env = DummyCylEnv()
    buffer = UnsupervisedBuffer(buffer_size=15, obs_dim=env.obs_dim, action_dim=env.action_dim,
                                time_dim=traj_horizon)
    random_pol = RandomPol(env=env)

    outer_iters = 40000 // (n_trajs_per_iteration*traj_horizon)
    print(F"Outer Iters: {outer_iters}")

    sac_pol = SACPol(env=env, total_training_steps=outer_iters*traj_horizon)
    exploration_pol = random_pol

    init_tf()

    traj_states, traj_actions, traj_rews = rollout(pol=exploration_pol, env=env, horizon=traj_horizon)
    buffer.add_traj(traj_states, traj_actions)
    buffer.overwrite_rewards(traj_rews)

    exploration_pol = sac_pol

    for outer_iter_idx in range(outer_iters):
        for i in range(n_trajs_per_iteration):
            traj_states, traj_actions, traj_rews = rollout(pol=exploration_pol, env=env, horizon=traj_horizon)
            buffer.add_traj(traj_states, traj_actions)
            buffer.overwrite_rewards(traj_rews)
        buffer.make_rl_data()

        losses = exploration_pol.train(num_timesteps=traj_horizon, replay_buffer=buffer)

        val_rews = []
        for i in range(3):
            env.reset()
            rew = rollout_eval(pol=exploration_pol, env=env, render=False)
            val_rews.append(rew)
        table = PrettyTable(["Iteration", "Validation Rewards"])
        table.add_row([outer_iter_idx, np.mean(val_rews)])
        print(table)
        #print(F"Reward for val: {np.mean(val_rews)}")

    save = False
    if save:
        n_expert_trajs_to_save = 10
        all_states = []
        all_actions = []
        for i in range(10):
            traj_states, traj_actions, traj_rews = rollout(pol=exploration_pol, env=env, horizon=traj_horizon, render=True)
            all_states.append(traj_states)
            all_actions.append(traj_actions)

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        d = dict()
        d['expert_obs'] = all_states
        d['expert_acs'] = all_actions
        np.savez("expert_trajs_point.npz", **d)

    while True:
        env.reset()
        rew = rollout_eval(pol=exploration_pol, env=env, render=True)



def main():
    sess = tf.Session()
    with sess.as_default():
        run()



if __name__ == "__main__":
    main()
