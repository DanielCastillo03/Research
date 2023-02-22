import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from walk_env_mjcpy import HumanWalk
from walk_env import Hman



# Create environment
env = Hman()

save_name = "t"

checkpoint_callback = CheckpointCallback(
  save_freq=1_000_000,
  save_path="./checkpoints/",
  name_prefix=save_name,
  save_replay_buffer=True,
)

#params to test
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = "./PPO_Walk_tensorboard/",)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = "./PPO_Walk_tensorboard/",
            batch_size = 16,
            n_steps = 512,
            gamma = 0.995,
            learning_rate = 7.103910717143798e-05,
            ent_coef = 3.789297601209816e-05,
            clip_range = 0.4,
            gae_lambda = 0.8,
            n_epochs = 5,
            max_grad_norm = 0.3,
            vf_coef = 0.7167836800871341)


# Train the agent and display a progress bar
model.learn(total_timesteps=20_000_000, tb_log_name = save_name ,progress_bar = True, callback = checkpoint_callback)

# Save the agent
model.save("./saves/"+save_name)
