from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np
import mujoco_py
from pynput.keyboard import Key, Listener
import time
import copy
from gym import spaces


class ReacherDistractorEnv:
    def __init__(self, sparse_reward=False, horizon=200):
        self.goal_idx = 0 #np.random.randint(0, 4)
        self.goal_one_hot = np.array([0, 0, 0, 0])
        self.goal_one_hot[self.goal_idx] = 1
        self.sparse_reward = sparse_reward
        model = load_model_from_path("envs/ReacherDistractor.xml")
        sim = MjSim(model)
        self.sim = sim
        viewer = MjViewer(sim)
        self.init_state = sim.get_state()
        self.model = model
        self.viewer = viewer
        self.action_dim = len(self.sim.data.ctrl)
        self.obs_dim = len(self.get_obs())

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
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        obs = np.concatenate([qpos, qvel, self.goal_one_hot])
        #print(obs.shape)
        return obs

    def reset(self):
        self.goal_idx = np.random.randint(0, 4)
        self.goal_one_hot = np.array([0, 0, 0, 0])
        self.goal_one_hot[self.goal_idx] = 1
        #print(self.goal_idx)
        qpos, qvel = self.get_random_state()
        qpos_ground = np.array([0.44896555, -0.0152224,   0.07119617, -0.47178262,  0.07728941,  0.5,
                0.05299965, -0.5, 0.5, 0.10195087])
        qpos_ground[0:2] = qpos[0:2]
        qpos = qpos_ground
        qvel = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #print(qpos)
        #print(qvel)
        #print(F"Qpos: {qpos}")
        #print(F"Qvel: {qvel}")
        self.set_state(qpos, qvel)
        #print("Resetting")
        self.cur_step = 0
        return self.get_obs()
        #self.sim.set_state(self.init_state)

    def reset_fixed_goal(self, goal_idx=0):
        self.goal_idx = goal_idx
        self.goal_one_hot = np.array([0, 0, 0, 0])
        self.goal_one_hot[self.goal_idx] = 1
        # print(self.goal_idx)
        qpos, qvel = self.get_random_state()
        qpos_ground = np.array([0.44896555, -0.0152224, 0.07119617, -0.47178262, 0.07728941, 0.5,
                                0.05299965, -0.5, 0.5, 0.10195087])
        qpos_ground[0:2] = qpos[0:2]
        #print("new reacher state:", qpos[0:2])
        qpos = qpos_ground
        qvel = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(F"Qpos: {qpos}")
        # print(F"Qvel: {qvel}")
        self.set_state(qpos, qvel)
        # print("Resetting")
        self.cur_step = 0
        return self.get_obs()
        # self.sim.set_state(self.init_state)


    def get_random_state(self):
        qpos = 5.2 * np.random.randn(self.model.nq)
        qpos = np.clip(qpos, -10, 10)
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
        cyl_obs_dist = self.get_body_com("fingertip") - self.get_body_com("target_" + str(self.goal_idx))
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


class KeyInputWrapper:
    def __init__(self, env):
        # Collect events until released
        self.last_key_time = time.time()
        self.env = env
        env.render()
        self.last_obs = env.reset()
        self.current_demo_acts = []
        self.current_demo_obs = []
        self.all_demo_acts = []
        self.all_demo_obs = []
        #while True:
        #    env.render()
        #with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
        #    listener.start()
        #    listener.join()
        listener = Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        while True:
            time.sleep(0.0001)
            env.step(np.array([0, 0]))
            #env.sim.step()
            env.render()



    def on_press(self, key):
        if str(key) in ["'h'", "'j'", "'k'", "'u'"]:
            #print("key pressed")
            key = str(key)
            #self.env.step(np.random.randn(2))
            if time.time() - self.last_key_time > 1/30:  # 30 fps
                self.last_key_time = time.time()
                action = [0, 0]
                if key == "'u'":
                    action[1] = 1
                elif key == "'j'":
                    action[1] = -1
                elif key == "'k'":
                    action[0] = 1
                elif key == "'h'":
                    action[0] = -1
                #print(action)
                action = 0.01*np.array(action)
                self.current_demo_acts.append(action)
                self.current_demo_obs.append(copy.deepcopy(self.last_obs))
                self.last_obs, _, _, _ = self.env.step(action)

                #self.env.render()
        if key == Key.shift:
            print("Resetting")
            self.last_obs = self.env.reset()
            self.current_demo_obs = []
            self.current_demo_acts = []
        if str(key) == "'z'":
            print("Saving trajectory")
            self.all_demo_acts.append(copy.deepcopy(self.current_demo_acts))
            self.all_demo_obs.append(copy.deepcopy(self.current_demo_obs))
            self.last_obs = self.env.reset()
            self.current_demo_obs = []
            self.current_demo_acts = []
        if str(key) == "'x'":
            d = dict()
            for i in range(len(self.all_demo_obs)):
                d['expert_obs_' + str(i)] = self.all_demo_obs[i]
                d['expert_acs_' + str(i)] = self.all_demo_acts[i]
            np.savez("expert_trajs.npz", **d)

            #print(F"{key} pressed")

    def on_release(self, key):
        #print('{0} release'.format(
        #    key))
        if key == Key.esc:
            # Stop listener
            return False


def collect_demo():
    pass


def test_point_env():
    import time
    env = PointDistractorEnv()
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


def test_key_input():
    env = PointEnv()
    env.render()
    KeyInputWrapper(env)

def test_load_trajs():
    zz = np.load("expert_trajs.npz")
    a = zz['expert_acs_1']
    o = zz['expert_obs_1']
    print(1.0*a)

    kkkk

if __name__ == "__main__":
    test_point_env()

