from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np
import mujoco_py
from pynput.keyboard import Key, Listener
import time
import copy
from gym import spaces


class ReacherEnv:
    def __init__(self, sparse_reward=False, horizon=700, fixed_reset=True):
        self.sparse_reward = sparse_reward
        model = load_model_from_path("envs/reacher.xml")
        sim = MjSim(model)
        self.sim = sim
        viewer = MjViewer(sim)
        self.init_state = sim.get_state()
        self.model = model
        self.viewer = viewer
        self.action_dim = len(self.sim.data.ctrl)
        #print(self.action_dim)
        self.obs_dim = len(self.get_obs())
        self.fixed_reset = fixed_reset

        high = np.array([np.inf] * self.obs_dim)
        # self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([2.0] * self.action_dim)
        self.action_space = spaces.Box(-high, high, dtype=np.float32)
        self.metadata = None
        self.horizon = horizon
        self.cur_step = 0

    def step(self, a):
        done = False
        self.sim.data.ctrl[0:2] = a
        #k = self.sim.data.ctrl
        #self.sim.data.ctrl[0:2] = a
        #for _ in range(4):
        #    self.sim.step()
        self.sim.step()
        rew = self.get_reward(a)
        obs = self.get_obs()
        #print(self.sim.data.qpos)
        #print(self.sim.get_state())
        self.cur_step += 1
        if self.cur_step > self.horizon:
            done = True
            self.cur_step = 0
            #self.reset()
        return obs, rew, done, dict()

    def get_body_com(self, body_name):
        return self.sim.data.get_body_xpos(body_name)

    def get_obs(self):
        #qpos = self.sim.data.qpos
        #qvel = self.sim.data.qvel
        theta = self.sim.data.qpos.flat[:2]
        theta_dot = self.sim.data.qvel.flat[:2]
        base = [np.cos(theta), np.sin(theta), theta_dot, ]
        target = self.get_body_com("target")
        tip = self.get_body_com("fingertip")[:2]
        delta = self.get_body_com("fingertip") - self.get_body_com("target")
        obs = np.concatenate([*base, target, tip, delta])
        #obs = np.concatenate([qpos, qvel])
        return obs

    def reset(self):
        qpos, qvel = self.get_random_state()
        #print(qpos)
        if self.fixed_reset:
            qpos_ground = [-0.1, 0.1,  -0.1, 0.1] #np.zeros_like(qpos)
            qpos_ground[0:2] = qpos[0:2]
            qpos = qpos_ground
        self.set_state(qpos, qvel)
        #print("Resetting")
        self.cur_step = 0
        return self.get_obs()
        #self.sim.set_state(self.init_state)


    def get_random_state(self):
        qpos = 3.0*np.random.randn(self.model.nq)
        qpos = np.clip(qpos, -3.1, 3.1)
        qvel = np.zeros_like(qpos)
        return qpos, qvel


    def close(self):
        self.viewer

    def set_state(self, qpos, qvel):
        #assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def get_reward(self, a):
        #  This shouldn't even be used.
        #cyl_xy = self.get_body_com('cylinder')
        #point_xy = self.get_body_com('point')
        #print(F"Cyl XY: {cyl_xy}")
        #print(F"Point XY: {point_xy}")
        #cyl_obs_dist = self.get_body_com("object") - self.get_body_com("goal")
        cyl_obs_dist = self.get_body_com("fingertip") - self.get_body_com("target")
        dist = np.linalg.norm(cyl_obs_dist)
        ctrl = np.square(a).sum()
        # print(F"Dist is: {dist}")
        if self.sparse_reward is False:
            rew = -1.0 * (dist + ctrl)  # negative cost is reward
            return rew
        else:
            return float(dist < 0.01)

    def render(self):
        self.viewer.render()


def test_reacher_env():
    import time
    env = ReacherEnv()
    i = 0
    while True:
        #action_array = np.array([0, 1])
        action_array = 0.1*np.random.randn(2)
        obs, rew, false, dct = env.step(action_array)
        time.sleep(0.001)
        env.render()
        i += 1
        if i % 500 == 0 and i > 0:
            print(F"resetting: {i//500}")
            # pass
            env.reset()

if __name__ == "__main__":
    test_reacher_env()

