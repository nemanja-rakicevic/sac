from .gym_env import GymEnv
from .multigoal import MultiGoalEnv

from gym.envs.registration import register



# Box2D envs

register(
    id='BipedalWalkerEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_walker:BipedalWalkerEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)

register(
    id='StrikerEnv-v0',
    entry_point='sac.envs.box2d.env_striker:StrikerEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)


# PyBullet envs

register(
    id='HalfCheetahEnv-v0',
    entry_point='sac.envs.pybullet.env_half_cheetah:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)

register(
    id='HumanoidEnv-v0',
    entry_point='sac.envs.pybullet.env_humanoid:HumanoidEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)

register(
    id='QuadrupedEnv-v0',
    entry_point='sac.envs.pybullet.env_ant:QuadrupedEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)
