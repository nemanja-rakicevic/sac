
from .gym_env import GymEnv
from .multigoal import MultiGoalEnv

from gym.envs.registration import register


### Box2D envs ###


### Striker

register(
    id='StrikerEnv-v0',
    entry_point='sac.envs.box2d.env_striker:StrikerEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)

register(
    id='StrikerAugmentedEnv-v0',
    entry_point='sac.envs.box2d.env_striker:StrikerAugmentedEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)

register(
    id='StrikerAugmentedMixScaleEnv-v0',
    entry_point='sac.envs.box2d.env_striker:StrikerAugmentedMixScaleEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)

register(
    id='StrikerAugmentedConstantEnv-v0',
    entry_point='sac.envs.box2d.env_striker:StrikerAugmentedConstantEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)

register(
    id='StrikerTargetEnv-v0',
    entry_point='sac.envs.box2d.env_striker:StrikerTargetEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)

### Biped

register(
    id='BipedalWalkerEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_walker:BipedalWalkerEnv',
    max_episode_steps=500,
    reward_threshold=0,
)


register(
    id='BipedalWalkerAugmentedEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_walker:BipedalWalkerAugmentedEnv',
    max_episode_steps=500,
    reward_threshold=0,
)


register(
    id='BipedalWalkerAugmentedMixScaleEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_walker:BipedalWalkerAugmentedMixScaleEnv',
    max_episode_steps=500,
    reward_threshold=0,
)


register(
    id='BipedalWalkerAugmentedNormalizedEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_walker:BipedalWalkerAugmentedNormalizedEnv',
    max_episode_steps=500,
    reward_threshold=0,
)



register(
    id='BipedalKickerEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_kicker:BipedalKickerEnv',
    max_episode_steps=2000,
    reward_threshold=0,
)


register(
    id='BipedalKickerAugmentedEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_kicker:BipedalKickerAugmentedEnv',
    max_episode_steps=500,
    reward_threshold=0,
)


register(
    id='BipedalKickerAugmentedMixScaleEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_kicker:BipedalKickerAugmentedMixScaleEnv',
    max_episode_steps=500,
    reward_threshold=0,
)


register(
    id='BipedalKickerAugmentedNormalizedEnv-v0',
    entry_point='sac.envs.box2d.env_bipedal_kicker:BipedalKickerAugmentedNormalizedEnv',
    max_episode_steps=500,
    reward_threshold=0,
)


### PyBullet envs ###

### Quadruped

register(
    id='QuadrupedWalkerEnv-v0',
    entry_point='sac.envs.pybullet.env_quadruped_walker:QuadrupedWalkerEnv',
    max_episode_steps=300,
    reward_threshold=0,
)


register(
    id='QuadrupedWalkerAugmentedEnv-v0',
    entry_point='sac.envs.pybullet.env_quadruped_walker:QuadrupedWalkerAugmentedEnv',
    max_episode_steps=300,
    reward_threshold=0,
)

register(
    id='QuadrupedWalkerAugmentedMixScaleEnv-v0',
    entry_point='sac.envs.pybullet.env_quadruped_walker:QuadrupedWalkerAugmentedMixScaleEnv',
    max_episode_steps=300,
    reward_threshold=0,
)




register(
    id='QuadrupedKickerEnv-v0',
    entry_point='sac.envs.pybullet.env_quadruped_kicker:QuadrupedKickerEnv',
    max_episode_steps=5000,
    reward_threshold=0,
)


register(
    id='QuadrupedKickerAugmentedEnv-v0',
    entry_point='sac.envs.pybullet.env_quadruped_kicker:QuadrupedAugmentedKickerEnv',
    max_episode_steps=5000,
    reward_threshold=0,
)


register(
    id='QuadrupedKickerAugmentedMixScaleEnv-v0',
    entry_point='sac.envs.pybullet.env_quadruped_kicker:QuadrupedKickerAugmentedMixScaleEnv',
    max_episode_steps=5000,
    reward_threshold=0,
)





register(
    id='QuadrupedKickerBoundedEnv-v0',
    entry_point='sac.envs.pybullet.env_quadruped_kicker:QuadrupedKickerEnv',
    max_episode_steps=5000,
    reward_threshold=0,
)
