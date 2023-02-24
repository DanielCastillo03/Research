import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from walk_env_mjcpy import HumanWalk



# Create environment
env = HumanWalk()

save_name = "02-22-2023v3"

checkpoint_callback = CheckpointCallback(
  save_freq=1_000_000,
  save_path="./checkpoints/",
  name_prefix=save_name,
  save_replay_buffer=True,
)

#params to test
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = "./PPO_Walk_tensorboard/",)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = "./PPO_Walk_tensorboard/",
            batch_size = 512,
            n_steps = 1024,
            gamma = 0.99,
            learning_rate = 2.929050476596153e-05,
            ent_coef = 2.5089277308447216e-08,
            clip_range = 0.1,
            gae_lambda = 0.95,
            n_epochs = 10,
            max_grad_norm = 0.5,
            vf_coef = 0.36889095328104893)


# Train the agent and display a progress bar
model.learn(total_timesteps=20_000_000, tb_log_name = save_name ,progress_bar = True, callback = checkpoint_callback)

# Save the agent
model.save("./saves/"+save_name)
