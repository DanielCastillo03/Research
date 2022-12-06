import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from walk_env import Walk

#comment to run TensorBoard
#tensorboard --logdir <tensorboard_log>

env = Walk()
env.reset()

model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log = "./PPO_Walk_tensorboard/")
model.learn(total_timesteps = 1_000_000, tb_log_name = "ts_million")
model.save("ts_million")

obs = env.reset()

episode = 1

while True:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    
    if done:
        episode += 1
        
    if episode % 10 == 0:
        evaluate_policy(model, env)
