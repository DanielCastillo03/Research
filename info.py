from mujoco_py import MjSim, load_model_from_path
import numpy as np

human = "/home/daniel/Desktop/Research/Rajagopal2015_converted/Human_test.xml"
humanoid = "/home/daniel/anaconda3/envs/gym-test/lib/python3.8/site-packages/gym/envs/mujoco/assets/humanoid.xml"

human_model = load_model_from_path(human)
hoid_model = load_model_from_path(humanoid)

sim_humanoid = MjSim(hoid_model)
sim_human = MjSim(human_model)

data_human = sim_human.data
data_hoid = sim_humanoid.data

print(human_model.body_names, len(human_model.body_names))
print(data_human.body_xpos)