import gym
from gym import spaces
import numpy as np

class FetchEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self, auto_reset = True):
        self.env = gym.make("FetchReach-v1", reward_type='dense')
        self.obs_dim = 16
        high = np.array([np.inf] * self.obs_dim)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        
        self.auto_reset = auto_reset
        
        self.env.reset()
     
    def wrap(self, obs):
        new_obs = np.concatenate((obs['observation'], obs['achieved_goal'], obs['desired_goal']), axis = None)
        return new_obs
    
    def no_auto_reset(self):
        self.auto_reset = False
        
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.auto_reset:
            if done: 
                self.env.reset()
        return self.wrap(obs), rew, done, info
        
    def reset(self):
        obs = self.env.reset()
        return self.wrap(obs)
        
    
    def render(self, mode='human', close=False):
        self.env.render()
    
def test_fetch_env():
    env = FetchEnv()
    print (env.observation_space)
    i = 0
    while i < int(1e3):
        action_array = 0.001*np.random.randn(4)
        obs, rew, done, dct = env.step(action_array)
        i += 1
        if done: 
            print ("final reward: ", rew)
            print ("observation: ", obs, "time:", i)

def test_env():
    env = gym.make("FetchReach-v1", reward_type='dense')
    env.reset()
    i = 0
    while i < int(1e2):
        action_array = 0.0001*np.random.randn(4)
        obs, rew, done, dct = env.step(action_array)
        print (done)
        i += 1
        if done: 
            print ("final reward: ", rew)
            print ("observation: ", obs, "time:", i)
            env.reset()
            
if __name__ == "__main__":
    # test_env()
    test_fetch_env()