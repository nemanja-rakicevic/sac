
import numpy as np
from gym import utils
import os

from pybullet_envs.gym_locomotion_envs import  WalkerBaseBulletEnv, Ant


REWARD_THRSH = 20
_VEL_THRSH = .0005



### TODO : Make camera from above!

# self.camera = Camera(self.unwrapped)

# class Camera:

#   def __init__(self, env):
#     self.env = env
#     pass

#   def move_and_look_at(self, i, j, k, x, y, z):
#     lookat = [x, y, z]
#     # camInfo = self.env._p.getDebugVisualizerCamera()
#     # distance = camInfo[10] + i
#     # pitch = camInfo[9] + j
#     # yaw = camInfo[8] + k
#     distance, yaw, pitch = 5, 0, 0
#     self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)



class QuadrupedEnv(WalkerBaseBulletEnv):
    """
    Quadruped Ant agent
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    """

    def __init__(self, render=False):
        self.init_body = np.zeros(2)
        self.robot = Ant()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

        self.param_ranges = np.vstack([self.action_space.low,
                                       self.action_space.high]).T
        self.env_info = dict(
            num_targets=1,
            num_obstacles=0,
            wall_geoms=None,
            ball_geom=None,
            target_info=[{'xy': (20, 0)}],
            striker_ranges=None,
            ball_ranges=None)


    def _get_info_dict(self, obs=None):
        hull_position = self.unwrapped.robot.body_xyz
        hull_angle = self.unwrapped.robot.body_rpy[1]
        contact_info = obs[np.array([24,25,26,27])] if obs is not None else []
        velocity_info = self.unwrapped.robot_body.speed()
        rel_angle = \
            self.unwrapped.jdict['hip_1'].current_relative_position()[0] \
            - self.unwrapped.jdict['hip_4'].current_relative_position()[0]

        info_dict = dict(position=np.append(hull_position, hull_angle),
                         position_aux=np.hstack([contact_info,
                                                 rel_angle, 
                                                 velocity_info]),
                         velocity=velocity_info,
                         angle=hull_angle)
        return info_dict


    def initialize(self, seed_task, **kwargs):
        # restart seed
        self.seed(seed_task)
        self.action_space.seed(seed_task)
        # standard reset
        obs = self.reset()
        info_dict = self._get_info_dict(obs)
        self.init_body = info_dict['position'][0:2]
        return obs, info_dict['position'], info_dict['position_aux']


    def finalize(self, rew_list, **kwargs):
        info_dict = self._get_info_dict()
        reward_len = np.linalg.norm(info_dict['position'][0:2]-self.init_body)
        outcome = -1  # 0 if reward_len >= REWARD_THRSH else -1
        return np.array([outcome, np.sum(rew_list)])


    def step(self, action):
        obs, rew, done, _ = super().step(action)
        info_dict = self._get_info_dict(obs)
        done = done or np.linalg.norm(info_dict['velocity'])<=_VEL_THRSH
        return obs, rew, done, info_dict
