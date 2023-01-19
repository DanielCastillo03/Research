from random import uniform
import gym
import numpy as np
from gym import utils
from gym import spaces
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym.envs.mujoco import MujocoEnv

class Walk(gym.Env):
    metadata = {
        "render_modes":[
            "human"
        ],
        "render_fps": 50
    }
    
    def __init__(       
        self,
        ):
        path = "/home/daniel/Desktop/Research/Rajagopal2015_converted/Rajagopal2015_converted.xml"
        
        self.model = load_model_from_path(path)
        self.sim = MjSim(self.model)

        self.data = self.sim.data
        self.state = self.sim.get_state()
        
        self.forward_weight = 1.25
        self.ctrl_weight  = 0.1
        self.healthy_reward = 5.0
        self.termin_when_unhealthy = True
        self.reset_noise = 1e-2
        self.done = False
        self.ep = 0
        self.range_pelvis_y_min = -0.8
        self.range_pelvis_y_max = 0.1
        self.range_pevis_tilt_min = -0.5
        self.range_pevis_tilt_max = 0.5
                
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(659,), dtype=np.float64)
        
        self.action_space = self.action_space = spaces.Box(low = -1, high = 1, shape = (97,), dtype = np.float64)


        # MujocoEnv.__init__(
        #     self,path, 5, self.observation_space,
        # )
        
    @property
    def is_healthy(self):
        min_pelvisy = self.range_pelvis_y_min
        max_pelvisy = self.range_pelvis_y_max
        min_tilt = self.range_pevis_tilt_min
        max_tilt = self.range_pevis_tilt_max
        
        is_healthy =  (min_pelvisy < self.data.get_joint_qpos("pelvis_ty") < max_pelvisy 
                       or 
                       min_tilt < self.data.get_joint_qpos("pelvis_tilt") < max_tilt)

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self.termin_when_unhealthy else False
        return terminated

        
    def _get_obs(self):
            return  np.concatenate(
                    [
                    #the qpos of each body in the model. len(51)
                    self.data.qpos.flat,
                    #the velocity of each body in the model len(51)
                    self.data.qvel.flat,
                    #mass and inertia of each body in the model len(230)
                    self.data.cinert.flat,
                    #center of mass based velocity len(138)
                    self.data.cvel.flat,
                    #Constraint force generated as the actuator force len(51)
                    self.data.qfrc_actuator.flat,
                    #center of mass based external force on the body len(138)
                    self.data.cfrc_ext.flat
                    ]
                )
            cd
    def step(self, a):
        self.done = False
        mass = np.expand_dims(self.model.body_mass, axis=1)
        xy_pos_before = (np.sum(mass * self.data.xipos, axis=0) / np.sum(mass))[0:2]
        
        self.data.ctrl[:] = a
        self.sim.step()

        self.ep += 1
        
        xy_pos_after = (np.sum(mass * self.data.xipos, axis=0) / np.sum(mass))[0:2]
        
        xy_vel = (xy_pos_after - xy_pos_before) / (self.model.opt.timestep * 5)
        
        x_vel, y_vel = xy_vel
        
        ctrl_cost = self.ctrl_weight * np.sum(np.square(self.data.ctrl))
        
        forward_reward = self.forward_weight * x_vel
        
        rewards = forward_reward + self.healthy_reward
        
        observation = self._get_obs()
        reward = rewards - ctrl_cost
        
        terminated = self.terminated
        
        if terminated or self.ep > 1000:
            self.reset_model()
            self.done = True
        
        info = {
            'reward_linvel': forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": self.healthy_reward,
            "x_position": xy_pos_after[0],
            "y_position": xy_pos_after[1],
            "distance_from_origin": np.linalg.norm(xy_pos_after, ord=2),
            "x_velocity": x_vel,
            "y_velocity": y_vel,
            "forward_reward": forward_reward,
        }
        
        
        return observation, reward, self.done, info
    
    def reset_model(self):
        self.ep = 0
        noise_low = -self.reset_noise
        noise_high = self.reset_noise
        qpos_init = np.zeros(len(self.data.qpos), dtype=np.float64)
        qvel_init = np.zeros(len(self.data.qvel), dtype=np.float64)
        
        qpos = qpos_init + np.random.uniform(noise_low, noise_high, self.model.nq)
        qvel = qvel_init + np.random.uniform(noise_low, noise_high, self.model.nv)

        
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)

        self.sim.forward()


    def reset(self):
        self.reset_model()
        return self._get_obs()
    
