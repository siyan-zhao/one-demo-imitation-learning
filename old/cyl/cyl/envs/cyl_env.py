import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
import glfw
import os


class CylEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mj_path, _ = mujoco_py.utils.discover_mujoco()
        xml_path = os.path.join(mj_path, 'model', 'cyl.xml')
        model = mujoco_py.load_model_from_path(xml_path)
        mujoco_env.MujocoEnv.__init__(self, 'cyl.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        #ctrl_cost_coeff = 0.0001
        #xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        reward = self.get_reward()
        ob = self._get_obs()
        return ob, reward, False, dict()  # dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def get_reward(self):
        x = self.sim.data.qpos
        kkk
        #xposafter = self.sim.data.qpos[0]
        #reward_fwd = (xposafter - xposbefore) / self.dt
        #reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        #reward = reward_fwd + reward_ctrl


    def get_body_com(self, body_name):
        return self.sim.data.get_body_xpos(body_name)

    def action_from_key(self, key):
        lb, ub = self.action_bounds
        if key == glfw.KEY_LEFT:
            return np.array([0, ub[0] * 0.3])
        elif key == glfw.KEY_RIGHT:
            return np.array([0, lb[0] * 0.3])
        elif key == glfw.KEY_UP:
            return np.array([ub[1], 0])
        elif key == glfw.KEY_DOWN:
            return np.array([lb[1], 0])
        else:
            return np.array([0, 0])

    def get_keys_to_action(self):
        lb, ub = self.action_bounds
        if key == glfw.KEY_LEFT:
            return np.array([0, ub[0] * 0.3])
        elif key == glfw.KEY_RIGHT:
            return np.array([0, lb[0] * 0.3])
        elif key == glfw.KEY_UP:
            return np.array([ub[1], 0])
        elif key == glfw.KEY_DOWN:
            return np.array([lb[1], 0])
        else:
            return np.array([0, 0])
        keys_a = {(ord('t'),): np.array([0, ub[0] * 0.3]),
                  (ord('g'),): np.array([0, lb[0] * 0.3])}
        return keys_a

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()


class SparseCylEnv(CylEnv):
    def __init__(self):
        CylEnv.__init__(self)

    def get_reward(self):
        return 0.0
