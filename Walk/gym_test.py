import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from walk_env_mjcpy import HumanWalk


# Create environment
env = HumanWalk()

save_name = "02-04-2023_human_lowerbody_ent_coef-0.02_batch_size-2048_clip-0.1"

checkpoint_callback = CheckpointCallback(
  save_freq=1_000_000,
  save_path="./checkpoints/",
  name_prefix=save_name,
  save_replay_buffer=True,
)

#params to test
# clip = 0.1, learning_rate = 1e-5, ent_coef = 0.01
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = "./PPO_Walk_tensorboard/", ent_coef = 0.2, batch_size = 2048, clip_range = 0.1)

# Train the agent and display a progress bar
model.learn(total_timesteps=20_000_000, tb_log_name = save_name ,progress_bar = True, callback = checkpoint_callback)

# Save the agent
model.save("./saves/"+save_name)
