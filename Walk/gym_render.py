import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from walk_env_mjcpy import HumanWalk

env = HumanWalk()
# env = gym.make("Humanoid-v3")

model = PPO.load("/home/daniel/Desktop/Research/Walk/saves/01-27-2023_human.zip", env=env)


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()

obs = vec_env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)

    vec_env.render()