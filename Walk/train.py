import gym
import numpy as np
from walk_env_mjcpy import HumanWalk
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


#comment to run TensorBoard
#tensorboard --logdir <tensorboard_log>

env = HumanWalk()
env.reset()
ts = 20_000_000
model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log = "./PPO_Walk_tensorboard/")
model.learn(total_timesteps = ts, tb_log_name = "12-7-2022_HumanWalk")
model.save("12-7-2022_HumanWalk")

del model  # delete trained model to demonstrate loading

model = PPO.load("12-7-2022_HumanWalk", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()