import numpy as np
import tensorflow as tf
import copy
from kick_ass_utils import get_env

def run_v2():
    import gym
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines import PPO2
    from stable_baselines import SAC
    from stable_baselines.sac.policies import MlpPolicy as Spol

    #horizon = 200

    env_name = 'sawyer_reach'
    reward_threshold = -0.05

    env = get_env(env_name)
    horizon = env.horizon
    print(horizon)
    #env.reset = env.reset_fixed_goal
    #kkkk
    model = SAC(Spol, env, verbose=1, learning_starts=500)
    model.learn(total_timesteps=3*40000)

    save_trajs = True
    if save_trajs:
        all_states = []
        all_actions = []
        n_trajs_to_save = 10
        while len(all_states) < n_trajs_to_save:
            this_traj_states, this_traj_actions = [], []
            if env_name in ['point_disctractor', 'reacher_distractor']:
                obs = env.reset_fixed_goal(0)
            else:
                obs = env.reset()
            for i in range(horizon):
                action, _states = model.predict(obs)
                this_traj_actions.append(copy.deepcopy(action))
                this_traj_states.append(copy.deepcopy(obs))
                obs, rewards, dones, info = env.step(action)
                #env.render()
            print(F"Final Rew: {rewards}")
            if rewards > reward_threshold:
                print("Saving good traj")
                this_traj_states = np.array(this_traj_states)
                this_traj_actions = np.array(this_traj_actions)
                all_states.append(this_traj_states)
                all_actions.append(this_traj_actions)

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        d = dict()
        d['expert_obs'] = all_states
        d['expert_acs'] = all_actions
        np.savez("expert_trajs_dir/expert_trajs_" + env_name + ".npz", **d)

    while True:
        obs = env.reset()
        for i in range(horizon):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #print(obs.shape)
            #print(action.shape)
            #kkkkk
            env.render()
        print(F"Final Rew: {rewards}")


def main():
    sess = tf.Session()
    with sess.as_default():
        run_v2()



if __name__ == "__main__":
    main()
