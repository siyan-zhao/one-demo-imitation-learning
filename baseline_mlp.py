import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as L
import copy
from envs.cyl import CylEnv, DummyCylEnv
from baselines.common.tf_util import function
from kick_ass_utils import euclidean_loss, MSE, eval_class_acc, init_tf
from policies import BCPol, BCPolWithEncoder, RandomPol, GaussianTrajectorEncoder, SACPol, MLPPolWithLastStateEncoder
from buffers import UnsupervisedBuffer, BonnieUnsupervisedBuffer
from envs.point import PointEnv
from prettytable import PrettyTable
from envs.points_plus_distractors import PointDistractorEnv
from envs.reacher import ReacherEnv
from envs.reacher_plus_distractors import ReacherDistractorEnv
from sawyer_envs.actual_sawyer_env import get_other_sawyer_envs
from kick_ass_utils import get_env_and_expert_traj

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
    env, expert_traj_dir, traj_horizon = get_env_and_expert_traj(env_name)


    batch_size = 10
    eval_mod = 1
    buffer_size = batch_size*20
    n_unsupervised_trajs = batch_size
    n_imitation_train_iters = traj_horizon // 20
    sac_train_iters = traj_horizon // 5
    outer_iters = 25

    print(F"Total Outer Iters: {outer_iters}")

    print(F"Obs Dim: {env.obs_dim}")
    print(F"Action Dim: {env.action_dim}")

    buffer = UnsupervisedBuffer(buffer_size=buffer_size, obs_dim=env.obs_dim, action_dim=env.action_dim,
                                time_dim=traj_horizon)
    buffer.update_rewards = buffer.update_rewards_mlp_plus_last_frame_only
    # buffer = BonnieUnsupervisedBuffer(buffer_size=20, obs_dim=env.obs_dim, action_dim=env.action_dim,
    #                             time_dim=traj_horizon)
    random_pol = RandomPol(env=env)
    sac_pol = SACPol(env=env, total_training_steps=outer_iters*sac_train_iters)
    exploration_pol = random_pol

    state = env.reset()
    #encode_dim = encoder.encode_dim
    a_mb = np.zeros((batch_size, env.action_dim))
    s_mb = np.zeros((batch_size, env.obs_dim))
    encoder = MLPPolWithLastStateEncoder(s_mb=s_mb, a_mb=a_mb)
    imitation_pol = encoder #BCPolWithEncoder(a_mb=a_mb, s_mb=s_mb, gaussian_encoder=encoder, discrete=False)
    baseline_imitation_pol = BCPol(s_mb=s_mb, a_mb=a_mb, discrete=False)

    init_tf()

    test_table = PrettyTable(["Iteration", "Test Set Reward", "Test set reward baseline pol", "average action rew"])
    test_set_rews, test_set_rews_baseline_pol, test_set_rews_average_act = [], [], []
    valid_set_loss, valid_set_baseline_loss = [], []

    for i in range(buffer_size):
        # fill buffer with random trajectories. Not sure about this step.
        # SAC does have an exploration phase at the start.
        traj_states, traj_actions, _ = rollout(pol=exploration_pol, env=env, horizon=traj_horizon)
        buffer.add_traj(traj_states, traj_actions)
    buffer.update_rewards(encoder=encoder, batch_size=batch_size)
    buffer.make_rl_data()

    # uncomment to actually train the sac pol. Right now it's just random.
    exploration_pol = sac_pol

    for outer_iter_idx in range(outer_iters):

        if outer_iter_idx % 1 == 0 and outer_iter_idx > 0:
            for i in range(n_unsupervised_trajs):
                traj_states, traj_actions, _ = rollout(pol=exploration_pol, env=env, horizon=traj_horizon)
                buffer.add_traj(traj_states, traj_actions)


        print(F"Training Imitaiton Policy for iteration: {outer_iter_idx} / {outer_iters}")
        for i in range(n_imitation_train_iters):
            #  use the buffer to get individual states and whole trajs.
            #  need s_mb, a_mb, whole_state_traj, whole_action_traj
            mb_state, mb_act, mb_whole_obs, mb_whole_acts = buffer.get_training_mb(batch_size)
            loss_enc = imitation_pol.train(s_mb=mb_state, a_mb=mb_act,
                                           whole_traj_mb=mb_whole_obs,
                                           whole_act_mb=mb_whole_acts)
            loss_im = baseline_imitation_pol.train(s_mb=mb_state, a_mb=mb_act)
        #print(F"Loss IM: {loss_im}")
        #print(F"Loss Enc: {loss_enc}")

        # buffer.update_rewards_v5()
        print(F"Updating rewards for iteration: {outer_iter_idx} / {outer_iters}")
        buffer.update_rewards(encoder=encoder, batch_size=batch_size)
        buffer.make_rl_data()

        print(F"Running SAC for iteration: {outer_iter_idx} / {outer_iters}")
        #for i in range(sac_train_iters):
        exploration_pol.train(num_timesteps=sac_train_iters, replay_buffer=buffer)

        # evaluate the imitation policy
        for i in range(n_imitation_train_iters):
            mb_state, mb_act, mb_whole_obs, mb_whole_acts = buffer.get_training_mb(batch_size)
            encoder_loss = imitation_pol.eval(s_mb=mb_state, whole_traj_mb=mb_whole_obs, whole_act_mb=mb_whole_acts,
                                              a_mb=mb_act)
            baseline_loss = baseline_imitation_pol.eval(s_mb=mb_state, a_mb=mb_act)
        table = PrettyTable(["Iteration", "Encoded Model Loss", "Baseline Model Loss"])
        table.add_row([outer_iter_idx, np.mean(encoder_loss), np.mean(baseline_loss)])
        print(table)
        valid_set_loss.append(np.mean(encoder_loss))
        valid_set_baseline_loss.append(np.mean(baseline_loss))
        #print(F"Encoder Loss: {np.mean(encoder_loss)}")
        #print(F"Baseline Loss: {np.mean(baseline_loss)}")

        # evaluate on test set.
        evaluate = True
        if evaluate:
            if outer_iter_idx % eval_mod == 0:
                ret = run_evaluation(outer_iter_idx, outer_iters, expert_traj_dir, env_name, env, traj_horizon, imitation_pol,
                                     baseline_imitation_pol, batch_size, test_table, test_set_rews, test_set_rews_baseline_pol,
                                     test_set_rews_average_act, valid_set_loss, valid_set_baseline_loss)
                test_set_rews, test_set_rews_baseline_pol, test_set_rews_average_act, test_table = ret
                #print(len(test_set_rews))


def run_evaluation(outer_iter_idx, outer_iters, expert_traj_dir, env_name, env, traj_horizon, imitation_pol,
                   baseline_imitation_pol, batch_size, test_table, test_set_rews, test_set_rews_baseline_pol,
                   test_set_rews_average_act, valid_set_loss, valid_set_baseline_loss):
    print(F"Running evaluation for iteration: {outer_iter_idx} / {outer_iters}")
    this_test_set_rews = []
    this_test_set_rews_baseline_pol = []
    this_test_set_rews_avg_act = []
    expert_trajs = np.load(expert_traj_dir, allow_pickle=True)
    expert_obs = expert_trajs['expert_obs']
    expert_acts = expert_trajs['expert_acs']
    # print(len(expert_obs))
    for traj_idx in range(len(expert_obs)):
        one_obs_traj = expert_obs[traj_idx]  # time by features
        one_act_traj = expert_acts[traj_idx]  # time by features
        # print(len(one_act_traj))
        # print(one_obs_traj.shape)
        # print(one_act_traj.shape)
        #  get a new trajectory
        if env_name in ['point_distractor', "reacher_dis"]:
            obs = env.reset_fixed_goal(0)
        else:
            obs = env.reset()
        # obs = env.reset()
        for time in range(traj_horizon):
            action = imitation_pol.act(state=obs, whole_act_mb=one_act_traj, whole_traj_mb=one_obs_traj)
            rescaled_action = action * np.abs(env.action_space.low)
            obs, rew, _, _ = env.step(rescaled_action)
            if traj_idx == len(expert_obs) - 1:
                #env.render()
                pass
        this_test_set_rews.append(rew)

        obs = env.reset()
        for time in range(traj_horizon):
            action = baseline_imitation_pol.act(obs)[0]
            #print(action.shape)
            obs, rew, _, _ = env.step(action)
        this_test_set_rews_baseline_pol.append(rew)

    if env_name in ['point_distractor', "reacher_dis"]:
        obs = env.reset_fixed_goal(0)
    else:
        obs = env.reset()

    eval_average_act = False
    if eval_average_act:
        for time in range(traj_horizon):
            obs = np.repeat(obs[np.newaxis, ...], batch_size, axis=0)

            action = imitation_pol.act(state=obs, whole_act_mb=expert_acts, whole_traj_mb=expert_obs)
            action = np.mean(action, axis=0)
            rescaled_action = action * np.abs(env.action_space.low)
            obs, rew, _, _ = env.step(rescaled_action)
            # if traj_idx == len(expert_obs) - 1:
            #env.render()
            # pass
        this_test_set_rews_avg_act.append(rew)
    else:
        this_test_set_rews_avg_act.append(-10000.0)

    this_test_set_rews = np.array(this_test_set_rews)
    # print(this_test_set_rews)
    this_test_set_rews_baseline_pol = np.array(this_test_set_rews_baseline_pol)
    this_test_set_rews_avg_act = np.array(this_test_set_rews_avg_act)
    test_set_rews.append(this_test_set_rews)
    test_set_rews_baseline_pol.append(this_test_set_rews_baseline_pol)
    test_set_rews_average_act.append(this_test_set_rews_avg_act)
    test_table.add_row([outer_iter_idx, np.mean(this_test_set_rews), np.mean(this_test_set_rews_baseline_pol),
                        np.mean(this_test_set_rews_avg_act)])
    print(test_table)
    # rrrr
    # print(F"Test set reward: {test_set_rews}")


    test_set_rews_np = np.array(test_set_rews)
    test_set_rews_baseline_pol_np = np.array(test_set_rews_baseline_pol)
    test_set_rews_avg_action_np = np.array(test_set_rews_average_act)
    # print(test_set_rews.shape)
    # print(test_set_rews_baseline_pol.shape)
    d = dict()
    d['test_rews'] = test_set_rews_np
    d['test_rews_baseline'] = test_set_rews_baseline_pol_np
    d['test_rews_avg_action'] = test_set_rews_avg_action_np
    d['valid_set_loss'] = valid_set_loss
    d['valid_set_baseline_loss'] = valid_set_baseline_loss
    np.savez("results/" + env_name + "baseline_mlp_plus_last_state_encoding.npz", **d)

    return [test_set_rews, test_set_rews_baseline_pol, test_set_rews_average_act, test_table]







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
