import numpy as np
import tensorflow as tf
from envs.point import PointEnv
from envs.points_plus_distractors import PointDistractorEnv
from envs.reacher import ReacherEnv
from envs.cyl import CylEnv
from sawyer_envs.actual_sawyer_env import get_other_sawyer_envs
from envs.reacher_plus_distractors import ReacherDistractorEnv

def euclidean_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.pow(y_true - y_pred, 2))) / tf.cast(tf.shape(y_true)[0], 'float')


def MSE(a, b):
    return np.mean((a - b) ** 2)


def eval_class_acc(y_hat_logits, y_true_labels):
    logit_arg_max = np.argmax(y_hat_logits, axis=1)
    y_true_arg_max = np.argmax(y_true_labels, axis=1)
    acc = np.equal(logit_arg_max, y_true_arg_max)
    return np.mean(acc)


def init_tf():
    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess = tf.get_default_session()
    # Run the initializer
    sess.run(init)
    sess.run(init_l)



def compute_kl_two_normals(n1_mean, n1_var, n2_mean, n2_var):
    #  this is the formula for the KL between two normal distributions.
    n1_var = n1_var + 0.01
    n2_var = n2_var + 0.01
    n1_std = np.sqrt(n1_var)
    n2_std = np.sqrt(n2_var)
    return np.log(np.divide(n2_std, n1_std)) + np.divide(n1_var + np.square(n1_mean - n2_mean), 2*n2_var) - 0.5
    #return tf.log(tf.div(n2_var, n1_var)) + tf.div(n1_var*n1_var + tf.square(n1_mean - n2_mean), 2*n2_var*n2_var) - 0.5


def get_env(env_name):
    if env_name == 'point':
        return PointEnv(horizon=200)
    if env_name == 'reacher':
        return ReacherEnv(horizon=400)
    if env_name == 'point_distractor':
        return PointDistractorEnv(fixed_reset=True, horizon=200)
    if env_name == 'sawyer_reach':
        return get_other_sawyer_envs('reach')
    if env_name == 'sawyer_touch':
        return get_other_sawyer_envs('touch')
    if env_name == 'sawyer_pick':
        return get_other_sawyer_envs('pick')
    if env_name == 'reacher_dis':
        return ReacherDistractorEnv(horizon=400)
    if env_name == 'cyl':
        return CylEnv(horizon=350)
    if env_name == "hopper":
        import gym
        env = gym.make('Hopper-v3')
        env.horizon = 300
        env.action_dim = 3
        env.obs_dim = 11
        return env
    if env_name == "cheetah":
        import gym
        env = gym.make('HalfCheetah-v3')
        env.horizon = 300
        env.action_dim = 6
        env.obs_dim = 17
        return env


def get_env_and_expert_traj(env):
    if env == 'point':
        traj_horizon = 200
        expert_dir = 'expert_trajs_dir/expert_trajs_point.npz'
        #env = get_env()
        env = PointEnv(horizon=traj_horizon)
        #print(env.horizon)
    elif env == 'point_distractor':
        traj_horizon = 200
        env = PointDistractorEnv(horizon=traj_horizon, fixed_reset=True)
        expert_dir = 'expert_trajs_dir/expert_trajs_point_distractor.npz'
    elif env == 'dummy':
        traj_horizon = 200
        env = DummyCylEnv(obs_size=8, action_size=2)
        expert_dir = 'expert_trajs_dir/expert_trajs_point.npz'
    elif env == 'reacher':
        traj_horizon = 400
        env = ReacherEnv(horizon=traj_horizon)
        expert_dir = 'expert_trajs_dir/expert_trajs_reacher.npz'
    elif env == "cyl":
        traj_horizon = 350
        env = CylEnv(horizon=traj_horizon)
        expert_dir = 'expert_trajs_dir/expert_trajs_cyl.npz'
        #return env, expert_dir
    elif env == 'reacher_dis':
        traj_horizon = 400
        env = ReacherDistractorEnv(horizon=traj_horizon)
        expert_dir = 'expert_trajs_dir/expert_trajs_reacher_dis_200.npz'
    elif env == 'sawyer':
        traj_horizon = 200
        env = get_other_sawyer_envs('pick')
        env.obs_dim = 28
        env.action_dim = 4
        expert_dir = 'expert_trajs_dir/expert_trajs_sawyer_pick.npz'
    elif env == 'sawyer_reach':
        traj_horizon = 200
        env = get_other_sawyer_envs('reach')
        env.obs_dim = 28
        env.action_dim = 4
        expert_dir = 'expert_trajs_dir/expert_trajs_sawyer_reach.npz'
    elif env == 'sawyer_pick':
        traj_horizon = 200
        env = get_other_sawyer_envs('pick')
        env.obs_dim = 28
        env.action_dim = 4
        expert_dir = 'expert_trajs_dir/expert_trajs_sawyer_pick.npz'
    elif env == "hopper":
        traj_horizon = 300
        import gym
        env = gym.make('Hopper-v3')
        env.horizon = 300
        env.action_dim = 3
        env.obs_dim = 11
        expert_dir = 'expert_trajs_dir/expert_trajs_hopper.npz'
    elif env == "cheetah":
        traj_horizon = 300
        import gym
        env = gym.make('HalfCheetah-v3')
        ob = env.reset()
        #print(ob.shape)
        #kkkk
        env.horizon = 300
        env.action_dim = 6
        env.obs_dim = 17
        expert_dir = 'expert_trajs_dir/expert_trajs_cheetah.npz'

    return env, expert_dir, traj_horizon



if __name__ == "__main__":
    pass

