
import pdb

import numpy as np
from gym import utils
import os

import pybullet
from pybullet_envs.gym_locomotion_envs import  WalkerBaseBulletEnv, Ant
from pybullet_envs.robot_locomotors import  WalkerBase


REWARD_THRSH = 20
_VEL_THRSH = .0005



class TopCamera:
    """ Overwriting the visualisation angle to make it birds-eye view """
    def __init__(self, env):
        self.env = env
        pass

    def move_and_look_at(self, i, j, k, x, y, z):
        # lookat = [x, y, z]
        lookat = self.env.camera_info['lookat']
        distance = self.env.camera_info['camera']['distance']-4.5
        yaw = self.env.camera_info['camera']['yaw']
        pitch = self.env.camera_info['camera']['pitch']
        # distance, yaw, pitch = 3, -90., -45.
        self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)



class FollowCamera:
    """ Overwriting the visualisation angle to make it birds-eye view """
    def __init__(self, env):
        self.env = env
        pass

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        distance = 4.5
        yaw = 0
        pitch = -30
        # distance, yaw, pitch = 3, -90., -45.
        self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)




class QuadrupedWalkerEnv(WalkerBaseBulletEnv):
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

        self.camera_info = {'camera': {'distance': 12, 'yaw': -0, 'pitch': -89},
                            'lookat': [0, 0, 0]}
        self.camera = TopCamera(self)


    def _get_info_dict(self, state=None):
        hull_position = self.unwrapped.robot.body_xyz
        hull_angle = self.unwrapped.robot.body_rpy[1]
        contact_info = state[np.array([24,25,26,27])] if state is not None else []
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
        state = self.reset()
        info_dict = self._get_info_dict(state)
        self.init_body = info_dict['position'][0:2]
        return state, info_dict['position'], info_dict['position_aux']


    def finalize(self, rew_list, **kwargs):
        info_dict = self._get_info_dict()
        reward_len = np.linalg.norm(info_dict['position'][0:2]-self.init_body)
        outcome = -1  # 0 if reward_len >= REWARD_THRSH else -1
        return np.array([outcome, np.sum(rew_list)])


    def step(self, action):
        state, rew, done, _ = super().step(action)
        info_dict = self._get_info_dict(state)
        done = done or np.linalg.norm(info_dict['velocity'])<=_VEL_THRSH
        return state, rew, done, info_dict


    def render(self, mode='human', close=False):
      
        if mode == "human":
            self.isRender = True
        if self.physicsClientId>=0:
            self.camera_adjust()

        if mode != "rgb_array":
            return np.array([])

        if (self.physicsClientId>=0):
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.camera_info['lookat'],
                roll=0,
                upAxisIndex=2,
                **self.camera_info['camera'])
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self._render_width) / self._render_height,
                nearVal=0.1,
                farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
            self._p.configureDebugVisualizer(
                self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        else:
            px = np.array([[[255,255,255,255]]*self._render_width]\
                            *self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), 
                               (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array



class AugmentedQuadruped(Ant):

    def __init__(self, scale=1):
        WalkerBase.__init__(self, "ant.xml", "torso", 
                            action_dim=8, obs_dim=30, power=2.5)
        self.walk_target_x = 0 
        self.walk_target_y = 0
        self.SCALE = scale

    def calc_state(self):
        standard_state = super().calc_state()
        augmented_state = np.concatenate(
            [standard_state, self.SCALE * self.robot_body.pose().xyz()[:2]])
        return augmented_state



class QuadrupedWalkerAugmentedEnv(QuadrupedWalkerEnv):

    def __init__(self, render=False):
        self.init_body = np.zeros(2)
        self.robot = AugmentedQuadruped(scale=1)
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

        self.param_ranges = np.vstack([self.action_space.low,
                                       self.action_space.high]).T
        self.env_info = dict(
            num_targets=1,
            num_statetacles=0,
            wall_geoms=None,
            ball_geom=None,
            target_info=[{'xy': (20, 0)}],
            striker_ranges=None,
            ball_ranges=None)

        self.camera_info = {'camera': {'distance': 12, 'yaw': -0, 'pitch': -89},
                            'lookat': [0, 0, 0]}
        self.camera = TopCamera(self)



class QuadrupedWalkerAugmentedMixScaleEnv(QuadrupedWalkerEnv):

    def __init__(self, render=False):
        self.init_body = np.zeros(2)
        self.robot = AugmentedQuadruped(scale=100)
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

        self.param_ranges = np.vstack([self.action_space.low,
                                       self.action_space.high]).T
        self.env_info = dict(
            num_targets=1,
            num_statetacles=0,
            wall_geoms=None,
            ball_geom=None,
            target_info=[{'xy': (20, 0)}],
            striker_ranges=None,
            ball_ranges=None)

        self.camera_info = {'camera': {'distance': 12, 'yaw': -0, 'pitch': -89},
                            'lookat': [0, 0, 0]}
        self.camera = TopCamera(self)
