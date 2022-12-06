import gym
import mujoco_py
import numpy as np
from stable_baselines3 import PPO
from walk_env import Walk

env = Walk()
viewer = mujoco_py.MjViewer(env.sim)
env.reset()

obs = env.reset()

model = PPO.load("ts_million", print_system_info = True)

while True:
    action, _states = model.predict(obs)
    obs, reward, _, _ = env.step(action)
    viewer.render()
