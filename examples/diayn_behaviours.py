"""Script for launching DIAYN experiments.

Usage:
    python mujoco_all_diayn.py --env=point --log_dir=/dev/null
"""
from rllab.envs.env_spec import EnvSpec
from rllab.misc.instrument import VariantGenerator
from rllab.envs.normalized_env import normalize
from rllab import spaces

from sac.algos.diayn_behaviours_algo import DIAYN_BD
from sac.envs.gym_env import GymEnv
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp
from sac.policies.gmm import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction, NNDiscriminatorFunction

import argparse
import numpy as np
import os

import datetime

import pdb


DATETIME = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
EPISODE_LIMIT = 1000000

TAG_KEYS = ['seed', 'time']

SHARED_PARAMS = {
    'time': DATETIME,
    'seed': [1],
    'eval_freq': 2000,
    'lr': 3E-4,
    'discount': 0.99,
    'tau': 0.01,
    'K': 4,
    'layer_size': 300,
    'batch_size': 128,
    'max_pool_size': 5*1E4,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'snapshot_mode': 'gap',
    'snapshot_gap': 10,
    'sync_pkl': True,
    'num_skills': 5,
    'scale_entropy': 0.1,
    'include_actions': False,
    'learn_p_z': False,
    'add_p_z': True,
}


ENV_PARAMS = {
    'bd-striker': {
        'prefix': 'striker',
        'env_name': 'StrikerEnv-v0',
        'max_path_length': 1000,
        'n_epochs': 100000,
        'scale_entropy': 0.1,
        'num_skills': 15300,
        'metric':
            {"type": "contact_grid", 
             "dim": 30}

    },
    'bd-bipedal-walker': {
        'prefix': 'bipedal_walker',
        'env_name': 'BipedalWalkerEnv-v0',
        'max_path_length': 500,
        'n_epochs': 100000,
        'scale_entropy': 0.1,
        'num_skills': 12500,
        'metric':
            {"type": "gait_grid", 
             "dim": 10}
    },
    'bd-bipedal-kicker': {
        'prefix': 'bipedal_kicker',
        'env_name': 'BipedalKickerEnv-v0',
        'max_path_length': 2000,
        'n_epochs': 100000,
        'scale_entropy': 0.1,
        'num_skills': 10000,
        'metric':
            {"type": "simple_grid", 
             "dim": 10}
    },

    'bd-quadruped-walker': {
        'prefix': 'quadruped_walker',
        'env_name': 'QuadrupedWalkerEnv-v0',
        'max_path_length': 300,
        'n_epochs': 100000,
        'scale_entropy': 0.1,
        'num_skills': 10000,
        'metric':
            {"type": "simple_grid", 
             "dim": 10}
    },
    'bd-quadruped-kicker': {
        'prefix': 'quadruped_kicker',
        'env_name': 'QuadrupedKickerEnv-v0',
        'max_path_length': 5000,
        'n_epochs': 100000,
        'scale_entropy': 0.1,
        'num_skills': 10000,
        'metric':
            {"type": "simple_grid", 
             "dim": 10}
    },
}


DEFAULT_ENV = 'swimmer'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default='swimmer')
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_skills', type=int, default=None)
    parser.add_argument('--eval_freq', type=int, default=None)
    parser.add_argument('--xdir', default=None)
    parser.add_argument('--xname', default=None)

    args = parser.parse_args()

    return args


def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = SHARED_PARAMS
    params.update(env_params)

    params.update({'seed': args.seed})

    if args.num_skills is not None:
        params.update({'num_skills': args.num_skills})
    if args.eval_freq is not None:
        params.update({'eval_freq': args.eval_freq})

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])
    return vg


def get_logdir(args, variant):
    xname = '' if args.xname is None else '---'+args.xname
    xdir = '' if args.xdir is None else '---'+args.xdir
    if args.log_dir is None:
        log_dir = "experiment_data/env__{}{}/" \
                  "ENV_{}_nn_policy__CFG__DIAYN_nskills_{}{}".format(
                        variant['prefix'], xdir,
                        variant['prefix'], variant['num_skills'], xname)
    else:
        log_dir = args.log_dir
    tag = "S{}---{}".format(variant['seed'], variant['time'])
    log_dir = os.path.join(log_dir, tag)
    return log_dir


def run_experiment(variant):
    if variant['env_name'] == 'humanoid-rllab':
        from rllab.envs.mujoco.humanoid_env import HumanoidEnv
        env = normalize(HumanoidEnv())
    elif variant['env_name'] == 'swimmer-rllab':
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        env = normalize(SwimmerEnv())
    else:
        env = normalize(GymEnv(variant['env_name']))

    obs_space = env.spec.observation_space
    assert isinstance(obs_space, spaces.Box)
    low = np.hstack([obs_space.low, np.full(variant['num_skills'], 0)])
    high = np.hstack([obs_space.high, np.full(variant['num_skills'], 1)])
    aug_obs_space = spaces.Box(low=low, high=high)
    aug_env_spec = EnvSpec(aug_obs_space, env.spec.action_space)
    pool = SimpleReplayBuffer(
        env_spec=aug_env_spec,
        max_replay_buffer_size=variant['max_pool_size'],
    )

    base_kwargs = dict(
        min_pool_size=variant['max_path_length'],
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = variant['layer_size']
    qf = NNQFunction(
        env_spec=aug_env_spec,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        env_spec=aug_env_spec,
        hidden_layer_sizes=[M, M],
    )

    policy = GMMPolicy(
        env_spec=aug_env_spec,
        K=variant['K'],
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001,
    )

    discriminator = NNDiscriminatorFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
        num_skills=variant['num_skills'],
    )


    algorithm = DIAYN_BD(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        discriminator=discriminator,
        pool=pool,
        qf=qf,
        vf=vf,

        lr=variant['lr'],
        scale_entropy=variant['scale_entropy'],
        discount=variant['discount'],
        tau=variant['tau'],
        num_skills=variant['num_skills'],
        save_full_state=False,
        include_actions=variant['include_actions'],
        learn_p_z=variant['learn_p_z'],
        add_p_z=variant['add_p_z'],

        # Additional params for behaviour tracking
        metric=variant['metric'],
        env_id=variant['prefix'],
        eval_freq=variant['eval_freq'],
        log_dir=get_logdir(args, variant),

    )

    algorithm.train()


def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    for i, variant in enumerate(variants):
        log_dir = get_logdir(args, variant)
        print('Launching {} experiments.'.format(len(variants)))
        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + args.exp_name,
            exp_name=variant['prefix'] + '-' + args.exp_name + '-' + str(i).zfill(2),
            n_parallel=1,  # Increasing this barely effects performance,
                           # but breaks learning of hierarchical policy.
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )

if __name__ == '__main__':
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator)
