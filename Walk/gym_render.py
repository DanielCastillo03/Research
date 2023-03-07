import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from walk_env_mjcpy import HumanWalk

env = HumanWalk()
# env = gym.make("Humanoid-v3")

model = PPO.load("/home/daniel/Desktop/Research/Walk/checkpoints/02-28-2023 full body_2000000_steps.zip", env=env)


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()

obs = vec_env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print("torso", env.data.body_xpos[14][2])
    print("======")
    vec_env.render()