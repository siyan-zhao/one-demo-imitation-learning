import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as L
import copy
from envs.cyl import CylEnv, DummyCylEnv
from baselines.common.tf_util import function
from kick_ass_utils import euclidean_loss, MSE, eval_class_acc, init_tf
from policies import BCPol, BCPolWithEncoder, RandomPol, GaussianTrajectorEncoder, SACPol
from buffers import UnsupervisedBuffer, BonnieUnsupervisedBuffer
from envs.point import PointEnv
from prettytable import PrettyTable
from envs.points_plus_distractors import PointDistractorEnv
from kick_ass_alg_v2 import get_env_and_expert_traj

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

    env_name = 'hopper'

    batch_size = 10
    eval_mod = 20
    #traj_horizon = 200
    #n_trajs_per_iteration = 1
    #outer_iters = 2*80000 // (n_unsupervised_trajs * traj_horizon) + 1
    outer_iters = 25*eval_mod
    print(F"Total Outer Iters: {outer_iters}")
    #env = CylEnv()
    #env = DummyCylEnv(obs_size=8, action_size=2)
    #env = PointEnv()
    env, expert_traj_dir, traj_horizon = get_env_and_expert_traj(env_name)
    traj_horizon = env.horizon
    a_mb = np.zeros((batch_size, env.action_dim))
    s_mb = np.zeros((batch_size, env.obs_dim))
    baseline_imitation_pol = BCPol(s_mb=s_mb, a_mb=a_mb, discrete=False)

    init_tf()

    test_table = PrettyTable(["Iteration", "Test Set Reward"])
    test_set_rews = []

    buffer = UnsupervisedBuffer(buffer_size=2, obs_dim=env.obs_dim, action_dim=env.action_dim,
                                time_dim=traj_horizon)

    expert_trajs = np.load(expert_traj_dir)
    expert_obs = expert_trajs['expert_obs']
    expert_acts = expert_trajs['expert_acs']
    #(expert_acts.shape)

    buffer.add_traj(expert_obs[0], expert_acts[0])
    buffer.add_traj(expert_obs[0], expert_acts[0])
    buffer.make_rl_data()

    # uncomment to actually train the sac pol. Right now it's just random.
    for outer_iter_idx in range(outer_iters):
        mb_state, mb_act, _, _ = buffer.get_training_mb(batch_size=batch_size)
        baseline_imitation_pol.train(mb_state, mb_act)

        #eval_rewards = []
        # evaluate on test set.
        if outer_iter_idx % eval_mod == 0:
            this_eval_rewards = []
            for eval_iter in range(10):
                obs = env.reset()
                for time in range(traj_horizon):
                    action = baseline_imitation_pol.act(obs)[0]
                    obs, rew, done, _ = env.step(action)
                    if done:
                        pass
                        #print(time)
                    if eval_iter == 9:
                        #env.render()
                        pass
                this_eval_rewards.append(rew)
            #rrrr
            this_eval_rewards = np.array(this_eval_rewards)
            #eval_rewards.append(this_eval_rewards)
            test_set_rews.append(this_eval_rewards)
            test_table.add_row([outer_iter_idx, np.mean(this_eval_rewards)])
            print(test_table)
    d = dict()
    d['test_rews'] = np.array(test_set_rews)
    np.savez("results/" + env_name + "_bc_baseline.npz", **d)







        # collect new trajs and add to the buffer
        # train encoder + BC pol jointly. The encoder is trained in PEARL style over the buffer. -- Siyan style
        # Update all buffer rewards. This requires access to everything in the encoder to get a distance metric on Zs
        # Train Exploration pol using the buffer that now has updated rewards.





def main():
    sess = tf.Session()
    with sess.as_default():
        run()



if __name__ == "__main__":
    main()
