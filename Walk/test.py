# hide all deprecation warnings from tensorflow
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import optuna
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

from walk_env_mjcpy import HumanWalk

def optimize_ppo2(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])),
        'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
        'gamma': trial.suggest_loguniform('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.2, 0.3),
        'gae_lambda': trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])    
        }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ppo2(trial)
    env = make_vec_env(lambda: HumanWalk(), seed=0)
    model = PPO('MlpPolicy', env, verbose=0, nminibatches=1, **model_params)
    model.learn(2_500_000)
    mean_reward, _ = evaluate_policy(model, HumanWalk(), n_eval_episodes=10)

    return -1 * mean_reward


if __name__ == '__main__':
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=1000, n_jobs=2)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')