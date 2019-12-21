import gym
import numpy as np
import cyl
env = gym.make('cyl-v0')
env.reset()
while(1):
   env.reset()
   print("reset")
   for i in range(50):
     env.render()
     
     x = env.step(env.sim.data.ctrl)
     x = env.step(np.array([0,0]))

    
   