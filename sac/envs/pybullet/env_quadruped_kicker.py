
import numpy as np
from gym import utils
import os


import pybullet
import pybullet_data

from pybullet_envs.gym_locomotion_envs import  WalkerBaseBulletEnv, Ant
from pybullet_envs.robot_locomotors import  WalkerBase
from pybullet_envs.scene_abstract import Scene

from robot_bases import BodyPart

import pdb

# with 1/120 max speed at contact is 5 for E-W
TIME_STEP_FIXED = 1/60 #1/120 # 0.0165
FRAME_SKIP = 4  # 4

_VEL_THRSH = .05
_REW_THRSH = 0.2
_EPS = 1e-06




def get_cube(_p, x, y, z):
    body = _p.loadURDF(os.path.join(os.path.join(os.path.dirname(__file__)), 
                                    "assets/wall.urdf"), [x, y, z])
    _p.changeDynamics(body, -1, mass=1.2)  #match Roboschool
    part_name, _ = _p.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    return BodyPart(_p, part_name, bodies, 0, -1)


def get_sphere(_p, x, y, z):
    body = _p.loadURDF(os.path.join(os.path.join(os.path.dirname(__file__)), 
                                    "assets/ball_blue.urdf"), [x, y, z])
    part_name, _ = _p.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    return BodyPart(_p, part_name, bodies, 0, -1)



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




class BoundedStadiumScene(Scene):
    """ 
        Custom-made playing field with walls and no reflection.
    """
    zero_at_running_strip_start_line = True  # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0
    multiplayer = False

    def __init__(self, ball_pos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball_pos = ball_pos

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)  # contains cpp_world.clean_everything()

        if (self.stadiumLoaded == 0):
            self.stadiumLoaded = 1
            # Add stadium with walls
            filename = os.path.join(os.path.dirname(__file__), 
                                    "assets/plane_bounded.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
            self._p.changeDynamics(0, -1, lateralFriction=.8, restitution=0.5, rollingFriction=0.005)
            # self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 1])
            # self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,i)
            for i in range(1, len(self.ground_plane_mjcf)):
                self._p.changeDynamics(i, -1, 
                                              # lateralFriction=100,  
                                              # linearDamping=100,
                                              # rollingFriction=0.1,
                                              # spinningFriction=0.03,
                                              restitution=1)
            # Add ball
            filename = os.path.join(os.path.dirname(__file__), 
                                    "assets/ball_blue.urdf")
            ball_body = self._p.loadURDF(filename, self.ball_pos)
            self._p.changeDynamics(ball_body, -1, restitution=1, mass=2.5)#,rollingFriction=0.001)
            # Add Obstacle to scene
            # self.obstacle = get_cube(self._p, 3.25, 0, 0.25)
            self.ground_plane_mjcf += (ball_body, )
            # # Update bouncyness
            # for i in range(0, len(self.ground_plane_mjcf)):
            #     print("===", self._p.getDynamicsInfo(i, -1))



class Quadruped(Ant):
    """
        same as ant added ball repositioning
    """

    def __init__(self):
        WalkerBase.__init__(self, "ant.xml", "torso", 
                            action_dim=8, obs_dim=32, power=2.5)
        self.walk_target_x = 0 
        self.walk_target_y = 0
        self.init_ball_pos = [0, 0, .25]
        self.init_robot_pos = [0, -1.5, .5]


    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        # Robot to initial position
        self.robot_body.reset_position(self.init_robot_pos)


    def calc_state(self):
        # standard_state = super().calc_state()

        j = np.array([j.current_relative_position() \
                for j in self.ordered_joints], dtype=np.float32).flatten()
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        body_pose = self.robot_body.pose()
        self.body_real_xyz = body_pose.xyz()
        self.body_xyz = body_pose.xyz()
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z == None:
          self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(self.walk_target_y-self.body_xyz[1],
                                            self.walk_target_x-self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1], 
             self.walk_target_x - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw
        # rotate speed back to body point of view
        rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], 
                              [np.sin(-yaw), np.cos(-yaw), 0], [0, 0, 1]])
        vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())  

        more = np.array(
            [
                z - self.initial_z,
                np.sin(angle_to_target),
                np.cos(angle_to_target),
                0.3 * vx,
                0.3 * vy,
                0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                r,
                p
            ],
            dtype=np.float32)

        standard_state = np.clip(np.concatenate([more] + \
                                                [j] + \
                                                [self.feet_contact]), -5, +5)

        if 'ball_blue' in self.parts.keys():
            ball_body = self.parts['ball_blue']
            augmented_state = np.concatenate([standard_state, 
                                              ball_body.get_position()[:2], 
                                              ball_body.speed()[:2]])
        else:
            augmented_state = np.concatenate([standard_state, 
                                              self.init_ball_pos[:2], 
                                              [0, 0]])
        return augmented_state



class QuadrupedKickerEnv(WalkerBaseBulletEnv):
    """
    Quadruped Ant agent
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    """
    MAX_AGENT_STEPS = 100

    def __init__(self, render=False):
        self.init = True
        self.init_body = np.zeros(2)
        # self.init_ball_pos = [0.5, -0.5, .25]
        self.init_ball_pos = [0, 0, .25]
        self.init_robot_pos = [0, -1.5, .5]

        self.robot = Quadruped()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
        self.param_ranges = np.vstack([self.action_space.low,
                                       self.action_space.high]).T
        _offset = 0.25/2
        self.ball_ranges = np.array([[ -6.+_offset, 6.-_offset ],
                                     [ -3.+_offset, 9-_offset ]])   
        self.env_info = dict(
            num_targets=1,
            num_obstacles=0,
            wall_geoms=[0, 1, 2, 3],
            ball_geom=5,
            target_info= [{'xy': (-0.5, 1.), 'radius': 0.25 }] ,
            striker_ranges=self.param_ranges,
            ball_ranges=self.ball_ranges)
        # self.camera_info = {'camera': {'distance': 10,
        #                                'yaw': -0,
        #                                'pitch': -69},
        #                     'lookat': [0, 0, 0]}
        self.camera_info = {'camera': {'distance': 12,
                                       'yaw': -0,
                                       'pitch': -89},
                            'lookat': [0, 3, 0]}
        self.camera = TopCamera(self)
        self._render_width = 240
        self._render_height = 240
        self.init = False


    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = BoundedStadiumScene(
                                bullet_client=bullet_client,
                                ball_pos=self.init_ball_pos,
                                gravity=9.8,
                                timestep=TIME_STEP_FIXED / FRAME_SKIP,
                                frame_skip=FRAME_SKIP)
        return self.stadium_scene


    def reset(self):
        self.nstep_internal = -1
        self.contact_objects = []
        r = super().reset()
        self.prev_ball_vx = 0
        self.prev_ball_vy = 0
        # self.parts['ball_blue'].reset_velocity(linearVelocity=[10, 10,0])              ##################
        return r


    def render(self, mode='human', close=False):
      
        if mode == "human":
            self.isRender = True
        if self.physicsClientId>=0:
            self.camera_adjust()

        if mode != "rgb_array":
            return np.array([])

        # base_pos = [0, 0, 0]
        # if (hasattr(self, 'robot')):
        #     if (hasattr(self.robot, 'body_real_xyz')):
        #         base_pos = self.robot.body_real_xyz
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


    def _get_info_dict(self, state=None, action=np.zeros(8)):
        hull_position = self.robot.body_xyz
        hull_angles = self.robot.body_rpy



        hull_pose = np.append(hull_position, hull_angles)
        info_dict = dict(position=state[28:30],
                         position_aux=np.concatenate([hull_position, 
                                                      hull_angles,
                                                      action]),
                         # position_aux=hull_pose,
                         velocity_info = self.robot_body.speed(),
                         # final_dist=np.linalg.norm(vec), 
                         # final_ctrl=np.linalg.norm(action),
                         contact_objects=self.contact_objects)


        return info_dict



    def _get_done(self, action, state):
        # episode is done when the ball stops, or complete miss
        ball_vel = np.linalg.norm(state[30:32])
        ball_pos = np.linalg.norm(state[28:30]) 
        strk_vel = np.linalg.norm(self.unwrapped.robot_body.speed()) \
                   + np.linalg.norm(action) 
        # Termination conditions
        done = ball_vel<=_VEL_THRSH and ball_pos>_EPS or \
               ball_vel<=_VEL_THRSH and strk_vel<=_VEL_THRSH
               # and np.isclose(ball_pos, 0., atol=_EPS)
        # print("\n===", self.nstep_internal, ball_vel, strk_vel, action)
        # print("===", ball_vel<=_VEL_THRSH , ball_pos>_EPS)
        # print("===", strk_vel<=_VEL_THRSH ,ball_vel<=_VEL_THRSH, np.isclose(ball_pos, 0., atol=_EPS))
        # print("===", done)
        return done and not self.init


    def _get_reward(self, state):
        # Reward vector contains: distances to targets, ball coordinates (x,y)
        # target_coms = [self.get_body_com(n)[:2] \
        #                   for n in self.unwrapped.model.body_names \
        #                   if 'target' in n]
        # target_dist = [np.linalg.norm(tc - self.get_body_com("ball")[:2]) \
        #                   for tc in target_coms]
        target_dist = -np.linalg.norm(state[-2:])
        return target_dist #+ [tuple(state[3:5])]



    def initialize(self, seed_task, **kwargs):
        # restart seed
        self.seed(seed_task)
        self.action_space.seed(seed_task)
        # standard reset
        state = self.reset()
        info_dict = self._get_info_dict(state)
        self.init_body = info_dict['position'][0:2]
        return state, info_dict['position'], info_dict['position_aux']


    def finalize(self, state, traj_aux, **kwargs):
        """ Define outcome: target index if within range, or -1 if failed """
        reward = self._get_reward(state)
        # returns closest target index if within threshold, otherwise -1
        trial_outcome = -1
        # trial_outcome = np.argmin(reward[:-1]) \
        #                 if np.sum(reward[-1])>0. and \
        #                    np.min(reward[:-1])<=_REW_THRSH else -1
        trial_fitness = -len(np.unique(traj_aux.astype(np.float16), axis=0))
        return np.array([trial_outcome, trial_fitness])


    def _check_contacts(self, state):
        ball_x, ball_y = state[28:30]
        ball_vx, ball_vy = state[-2:]
        # check wall vicinity
        wall_W, wall_E = np.isclose(ball_x, self.ball_ranges[0,:], atol=0.3)
        wall_S, wall_N = np.isclose(ball_y, self.ball_ranges[1,:], atol=0.3)
        # check change of direction
        dv_x = np.sign(ball_vx) != np.sign(self.prev_ball_vx)
        dv_y = np.sign(ball_vy) != np.sign(self.prev_ball_vy)
        # update prev ball_xy
        self.prev_ball_vx = ball_vx
        self.prev_ball_vy = ball_vy
        # evaluate contacts
        # wall_bool = np.array([wall_S, wall_E, wall_N, wall_W])
        # dv_bool = np.tile([dv_y, dv_x], 2)
        # contact = wall_bool * dv_bool
        # print("\n\n======{}\n{}\n{}\n{}".format((ball_vx, ball_vy), wall_bool, dv_bool, contact))
        contact = np.array([wall_S*dv_y, wall_E*dv_x, wall_N*dv_y, wall_W*dv_x])
        # return wall indices
        return np.where(contact)[0]+1


    def step(self, action):
        if self.nstep_internal > self.MAX_AGENT_STEPS: 
            action = 0 * action
        self.nstep_internal += 1
        state, rew, done, _ = super().step(action)
        assert len(state)==self.observation_space.shape[0]
        info_dict = self._get_info_dict(state, action)
        done = self._get_done(action, state)
        # # Add wall contacts
        # ball_contacts = []
        # if 'ball_blue' in self.parts.keys():
        #     contact_list = self.parts['ball_blue'].contact_list()
        #     if len(contact_list):
        #         ball_contacts = [cc[2] for cc in contact_list if 'wall' \
        #                         in self._p.getBodyInfo(cc[2])[0].decode("utf8")]
        #         if len(ball_contacts):
        #             self.contact_objects.append(ball_contacts[0])
        # Backup contact estimation
        alt_contact = self._check_contacts(state)
        if len(alt_contact):
            self.contact_objects.append(alt_contact[0])
        return state, rew, done, info_dict
