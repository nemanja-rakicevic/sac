
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from sac.algos.sac import SAC
from sac.algos.diayn import DIAYN
from sac.misc import tf_utils, utils
from sac.misc.sampler import rollouts
from sac.policies.hierarchical_policy import FixedOptionPolicy

import behaviour_representations.utils.behaviour_metrics as bmet

from collections import deque
import gtimer as gt
import json
import numpy as np
import os
import scipy.stats
import tensorflow as tf

import pdb

EPS = 1E-6

# env._wrapped_env.env_id
# self.environment.bd_metric.restart()

# self.environment.bd_metric.update(info_dict=info_dict)


# trial_metric = self.environment.bd_metric.calculate(
#                                               traj=self.traj_main,
#                                               traj_aux=self.traj_aux)

# # Initialise behaviour descriptor metric
# bd_class = ''.join([s.capitalize() for s in metric['type'].split('_')])
# self.bd_metric = bmet.__dict__[env_class+bd_class](**metric, 
#                                                 	 **self.env_info)




class DeterministicFixedOptionPolicy(FixedOptionPolicy):

    def get_action(self, obs):
        aug_obs = utils.concat_obs_z(obs, self._z, self._num_skills)
        with self._base_policy.deterministic(True): 
            d_action = self._base_policy.get_action(aug_obs)
        return d_action



class DIAYN_BD(DIAYN):

    # def __init__(self, *args, **kwargs):


    #     print("\n\n===", args)
    #     print("---", kwargs)

    #     super().__init__(*args, **kwargs)

    #     # Initialise behaviour descriptor metric
    #     env_class = env._wrapped_env.env_id.split('Env')[0]
    #     bd_class = 'GaitGrid'
    #     self.bd_metric = bmet.__dict__[env_class+bd_class](**metric, 
    #                                                        **self.env_info)



    def _init_training(self, env, policy, pool):
        """Method to be called at the start of training.

        :param env: Environment instance.
        :param policy:  Policy instance.
        :return: None
        """

        # Initialise behaviour descriptor metric
        env_class = env._wrapped_env.env_id.split('Env')[0]
        bd_class = 'ContactGrid' if 'Striker' in env_class else 'GaitGrid'
        metric = {"type": "contact_grid", "dim": 30} if 'Striker' in env_class \
                else {"type": "gait_grid", "dim": 10} 
        env_info = env._wrapped_env.env.env_info
        self.bd_metric = bmet.__dict__[env_class+bd_class](**metric, 
                                                           **env_info)

        self._env = env
        if self._eval_n_episodes > 0:
            if 'Bullet' in env._wrapped_env.env_id \
            or '-v0' in env._wrapped_env.env_id:
                self._eval_env = env
            else:
                self._eval_env = deep_clone(env)
        self._policy = policy
        self._pool = pool


    def sample_skills_to_bd(self, **kwargs):

        for z in range(self._num_skills):
            # Make policy  deterministic
            fixed_z_policy = DeterministicFixedOptionPolicy(self._policy, 
                                                            self._num_skills, z)
            paths = rollouts(env=self._eval_env, 
                             policy=fixed_z_policy,
                             path_length=self._max_path_length, 
                             n_paths=1,
                             render=False)

            # Convert paths to bd (list)
            print(paths[0]['env_infos'])
            

            # Save to file so it's loadable, nump epoch, num episodes




    def _train(self, env, policy, pool):
        """When training our policy expects an augmented observation."""
        self._init_training(env, policy, pool)

        with self._sess.as_default():
            observation = env.reset()
            policy.reset()
            log_p_z_episode = []  # Store log_p_z for this episode
            path_length = 0
            path_return = 0
            last_path_return = 0
            max_path_return = -np.inf
            n_episodes = 0

            if self._learn_p_z:
                log_p_z_list = [deque(maxlen=self._max_path_length) for _ in range(self._num_skills)]

            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs + 1),
                                      save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)


                path_length_list = []
                z = self._sample_z()
                aug_obs = utils.concat_obs_z(observation, z, self._num_skills)

                for t in range(self._epoch_length):
                    iteration = t + epoch * self._epoch_length

                    action, _ = policy.get_action(aug_obs)

                    if self._learn_p_z:
                        (obs, _) = utils.split_aug_obs(aug_obs, self._num_skills)
                        feed_dict = {self._discriminator._obs_pl: obs[None],
                                     self._discriminator._action_pl: action[None]}
                        logits = tf_utils.get_default_session().run(
                            self._discriminator._output_t, feed_dict)[0]
                        log_p_z = np.log(utils._softmax(logits)[z])
                        if self._learn_p_z:
                            log_p_z_list[z].append(log_p_z)

                    next_ob, reward, terminal, info = env.step(action)
                    aug_next_ob = utils.concat_obs_z(next_ob, z,
                                                     self._num_skills)
                    path_length += 1
                    path_return += reward

                    self._pool.add_sample(
                        aug_obs,
                        action,
                        reward,
                        terminal,
                        aug_next_ob,
                    )

                    if terminal or path_length >= self._max_path_length:
                        path_length_list.append(path_length)
                        observation = env.reset()
                        policy.reset()
                        log_p_z_episode = []
                        path_length = 0
                        max_path_return = max(max_path_return, path_return)
                        last_path_return = path_return

                        path_return = 0
                        n_episodes += 1

                        # EPISORE IS DONE n_episodes

                    else:
                        aug_obs = aug_next_ob
                    gt.stamp('sample')

                    if self._pool.size >= self._min_pool_size:
                        for i in range(self._n_train_repeat):
                            batch = self._pool.random_batch(self._batch_size)
                            self._do_training(iteration, batch)

                    gt.stamp('train')

                if self._learn_p_z:
                    print('learning p(z)')
                    for z in range(self._num_skills):
                        if log_p_z_list[z]:
                            print('\t skill = %d, min=%.2f, max=%.2f, mean=%.2f, len=%d' % (z, np.min(log_p_z_list[z]), np.max(log_p_z_list[z]), np.mean(log_p_z_list[z]), len(log_p_z_list[z])))
                    log_p_z = [np.mean(log_p_z) if log_p_z else np.log(1.0 / self._num_skills) for log_p_z in log_p_z_list]
                    print('log_p_z: %s' % log_p_z)
                    self._p_z = utils._softmax(log_p_z)


                # EPOCH IS DONE epoch
                self.sample_skills_to_bd(n_episodes=n_episodes, epoch=epoch)

                # Epoch
                self._evaluate(epoch)

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('time-train', times_itrs['train'][-1])
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-sample', times_itrs['sample'][-1])
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('episodes', n_episodes)
                logger.record_tabular('max-path-return', max_path_return)
                logger.record_tabular('last-path-return', last_path_return)
                logger.record_tabular('pool-size', self._pool.size)
                logger.record_tabular('path-length', np.mean(path_length_list))

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

                gt.stamp('eval')


            # EVALUATION
            self.sample_skills_to_bd()

            env.terminate()
