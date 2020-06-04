from .gym_env import GymEnv
from .multigoal import MultiGoalEnv

from gym.envs.registration import register


# Box2D envs

register(
    id='StrikerEnv-v0',
    entry_point='sac.envs.box2d.env_striker:StrikerEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)

register(
    id='BipedalWalkerEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_walker:BipedalWalkerEnv',
    max_episode_steps=500,
    reward_threshold=0,
)

register(
    id='BipedalKickerEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_kicker:BipedalKickerEnv',
    max_episode_steps=2000,
    reward_threshold=0,
)


# PyBullet envs

register(
    id='QuadrupedWalkerEnv-v0',
    entry_point='sac.envs.pybullet.env_quadruped_walker:QuadrupedWalkerEnv',
    max_episode_steps=300,
    reward_threshold=0,
)

register(
    id='QuadrupedKickerEnv-v0',
    entry_point='sac.envs.pybullet.env_quadruped_kicker:QuadrupedKickerEnv',
    max_episode_steps=5000,
    reward_threshold=0,
)
