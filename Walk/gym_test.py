import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from walk_env_mjcpy import HumanWalk



# Create environment
env = HumanWalk()

save_name = "02-28-2023 full body"

checkpoint_callback = CheckpointCallback(
  save_freq=1_000_000,
  save_path="./checkpoints/",
  name_prefix=save_name,
  save_replay_buffer=True,
)

#params to test
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = "./PPO_Walk_tensorboard")


# Train the agent and display a progress bar
model.learn(total_timesteps=20_000_000, tb_log_name = save_name ,progress_bar = True, callback = checkpoint_callback)

# Save the agent
model.save("./saves/"+save_name)
