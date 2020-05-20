
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


import csv
import time
import pickle
from operator import itemgetter 
from multiprocessing import Pool


import pdb

EPS = 1E-6

EPISODE_LIMIT = 1000000
SEED_TASK = 100


class DeterministicFixedOptionPolicy(FixedOptionPolicy):

    def get_action(self, obs):
        aug_obs = utils.concat_obs_z(obs, self._z, self._num_skills)
        with self._base_policy.deterministic(True): 
            d_action = self._base_policy.get_action(aug_obs)
        return d_action



class DIAYN_BD(DIAYN):

    def __init__(self,
                 base_kwargs,
                 env,
                 policy,
                 discriminator,
                 qf,
                 vf,
                 pool,

                 metric,
                 env_id,
                 log_dir,
                 eval_freq,

                 plotter=None,
                 lr=3E-3,
                 scale_entropy=1,
                 discount=0.99,
                 tau=0.01,
                 num_skills=20,
                 save_full_state=False,
                 find_best_skill_interval=10,
                 best_skill_n_rollouts=10,
                 learn_p_z=False,
                 include_actions=False,
                 add_p_z=True):
        """
            Same as DIAYN just added behaviour descriptor tracking and passing 
            env and log info.
        """

        Serializable.quick_init(self, locals())
        super(SAC, self).__init__(**base_kwargs)

        self._env = env
        self._policy = policy
        self._discriminator = discriminator
        self._qf = qf
        self._vf = vf
        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._discriminator_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        self._scale_entropy = scale_entropy
        self._discount = discount
        self._tau = tau
        self._num_skills = num_skills
        self._p_z = np.full(num_skills, 1.0 / num_skills)
        self._find_best_skill_interval = find_best_skill_interval
        self._best_skill_n_rollouts = best_skill_n_rollouts
        self._learn_p_z = learn_p_z
        self._save_full_state = save_full_state
        self._include_actions = include_actions
        self._add_p_z = add_p_z

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._training_ops = list()

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_discriminator_update()
        self._init_target_ops()

        self._sess.run(tf.global_variables_initializer())

        ### Additional params for behaviour tracking ###

        self.dirname = log_dir
        self.eval_freq = eval_freq
        self.env_id = env_id
        logger.set_snapshot_dir('snapshots')

        # Initialise behaviour descriptor metric
        env_info = env._wrapped_env.env.env_info
        env_class = ''.join([c.capitalize() for c in env_id.split('_')])
        bd_class = ''.join([s.capitalize() for s in metric['type'].split('_')])
        self.bd_metric = bmet.__dict__[env_class+bd_class](**metric, **env_info)

        # Initialise data discovery writer
        if not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)
        filepath = '{}/ref_data_DIAYN_{}.csv'.format(self.dirname, self.env_id)
        with open(filepath, 'a') as outfile: 
            writer = csv.DictWriter(outfile, 
                fieldnames = ["nloop", "niter", "nsmp", "nstep", "coverage", 
                              "fitness", "outcomes", "ratios"])
            writer.writeheader()

        # Save the modified arguments
        metadata_dict = {"experiment": {"controller": {"architecture": [20, 20],
                                                        "type": "nn_policy"},
                                         "environment": {"id": env_id},
                                         "metric": metric,
                                         "type": "nn_policy"}}
        filename = '{}/experiment_metadata.json'.format(self.dirname)
        with open(filename, 'w') as outfile:  
            json.dump(metadata_dict, outfile, sort_keys=True, indent=4)



    def _init_training(self, env, policy, pool):
        """Method to be called at the start of training.

        :param env: Environment instance.
        :param policy:  Policy instance.
        :return: None
        """      
        self._env = env
        if self._eval_n_episodes > 0:
            if 'Bullet' in env._wrapped_env.env_id \
            or '-v0' in env._wrapped_env.env_id:
                self._eval_env = env
            else:
                self._eval_env = deep_clone(env)
        self._policy = policy
        self._pool = pool


    def _save_dataset(self, **task_data_dict):
        """ Save trial data """
        if not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)
        with open(self.dirname+"/experiment_dataset.dat", "wb") as f:
            pickle.dump(task_data_dict, f)


    def _write_discovery(self, eval_time, unique_bds, unique_outcomes,
                               n_epoch, n_episodes):
        """ Save trial output info """
        n_eval_episodes = n_episodes - self.prev_n_episodes
        self.prev_n_episodes = n_episodes

        # Log statistics
        logger.log("(EPOCH {}, EP{}) Evaluaiton of {} skills:"\
                   "\n\t\t\t\tTime: {};"
                   "\n\t\t\t\tEpisodes since last eval: {}"
                   "\n\t\t\t\tTotal behaviours: {}".format(n_epoch, n_episodes,
                        self._num_skills, eval_time,
                        n_eval_episodes, len(unique_bds)))

        exploration_data = [0]+[n_epoch]+[n_eval_episodes]+[0] \
                            + [len(unique_bds)] \
                            + [max(unique_outcomes[:, 1])] \
                            + [sum(unique_outcomes[:, 0]==0)] \
                            + [-1]
        if not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)
        filepath = '{}/ref_data_DIAYN_{}.csv'.format(self.dirname, self.env_id)
        with open(filepath, 'a') as outfile: 
            writer = csv.writer(outfile) 
            writer.writerows([exploration_data])




    def sample_skills_to_bd(self, final=False, **kwargs):
        """
            Evaluate all the latent skills 
            and extract the behaviour descriptors
        """
        list_traj_main = []
        list_traj_aux = []
        list_outcomes = []
        list_skill_bd = []

        eval_time = time.time()
                                                                                ### PARALELIZE THIS
        for z in range(self._num_skills):
            # Make policy deterministic
            fixed_z_policy = DeterministicFixedOptionPolicy(self._policy, 
                                                            self._num_skills, z)
            # Evaluate skill
            self._eval_env._wrapped_env.env.initialize(seed_task=SEED_TASK)
            paths = rollouts(env=self._eval_env, 
                             policy=fixed_z_policy,
                             path_length=self._max_path_length, 
                             n_paths=1,
                             render=False)
            # Extract trajectory from paths
            traj_main = paths[0]['env_infos']['position']
            traj_aux = paths[0]['env_infos']['position_aux']
            list_traj_main.append(traj_main)
            list_traj_aux.append(traj_main)
            # Extract outcomes from paths
            trial_outcome = self._eval_env._wrapped_env.env.finalize(
                                                state=paths[0]['last_obs'], 
                                                rew_list=paths[0]['rewards'],
                                                traj=traj_main)
            list_outcomes.append(trial_outcome)
            # Extract and convert bd from paths
            self.bd_metric.restart()
            if self.bd_metric.metric_name == 'contact_grid':
                [self.bd_metric.update({'contact_objects': idict}) \
                    for idict in paths[0]['env_infos']['contact_objects']]
            trial_metric = self.bd_metric.calculate(traj=traj_main,
                                                    traj_aux=traj_aux)
            list_skill_bd.append(np.argmax(trial_metric))

        eval_time = time.time()-eval_time

        # Extract unique data
        unique_bds, unique_idx = np.unique(list_skill_bd, return_index=True)
        n_bd = len(unique_bds)
        unique_outcomes = np.array(list_outcomes)[unique_idx]
        unique_traj_main = itemgetter(*unique_idx)(list_traj_main)
        unique_traj_aux = itemgetter(*unique_idx)(list_traj_aux)

        # Save to file at exact point: nump epoch, num episodes
        self._write_discovery(eval_time=eval_time, unique_bds=unique_bds, 
                              unique_outcomes=unique_outcomes, **kwargs)

        # Save data at the end
        if final==True:
            unique_bds_1hot = np.zeros((n_bd, self.bd_metric.metric_size))
            unique_bds_1hot[np.arange(n_bd), unique_bds] = 1
            self._save_dataset(**{"outcomes": unique_outcomes, 
                                  "traj_main": unique_traj_main,
                                  "traj_aux": unique_traj_aux,
                                  "metric_bd": unique_bds_1hot})




    def _train(self, env, policy, pool):
        """When training our policy expects an augmented observation."""
        self._init_training(env, policy, pool)

        with self._sess.as_default():
            env._wrapped_env.env.initialize(seed_task=SEED_TASK)
            observation = env.reset()
            policy.reset()
            log_p_z_episode = []  # Store log_p_z for this episode
            path_length = 0
            path_return = 0
            last_path_return = 0
            max_path_return = -np.inf
            n_episodes = 0
            self.prev_n_episodes = 0

            if self._learn_p_z:
                log_p_z_list = [deque(maxlen=self._max_path_length) for _ in range(self._num_skills)]

            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs + 1),
                                      save_itrs=True):

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
                        env._wrapped_env.env.initialize(seed_task=SEED_TASK)
                        observation = env.reset()
                        policy.reset()
                        log_p_z_episode = []
                        path_length = 0
                        max_path_return = max(max_path_return, path_return)
                        last_path_return = path_return

                        path_return = 0
                        n_episodes += 1


                        # EPOCH IS DONE epoch
                        if not n_episodes % self.eval_freq:
                            self.sample_skills_to_bd(final=epoch==self._n_epochs,
                                                     n_epoch=epoch, 
                                                     n_episodes=n_episodes)

                            gt.stamp('behaviours')

                    else:
                        aug_obs = aug_next_ob

                    gt.stamp('sample')

                    if self._pool.size >= self._min_pool_size:
                        for i in range(self._n_train_repeat):
                            batch = self._pool.random_batch(self._batch_size)
                            self._do_training(iteration, batch)

                    gt.stamp('train')


                    # Terminate after 1000000 episodes
                    if n_episodes >= EPISODE_LIMIT:
                        break


                else:
                    continue
                break


                if self._learn_p_z:
                    print('learning p(z)')
                    for z in range(self._num_skills):
                        if log_p_z_list[z]:
                            print('\t skill = %d, min=%.2f, max=%.2f, mean=%.2f, len=%d' % (z, np.min(log_p_z_list[z]), np.max(log_p_z_list[z]), np.mean(log_p_z_list[z]), len(log_p_z_list[z])))
                    log_p_z = [np.mean(log_p_z) if log_p_z else np.log(1.0 / self._num_skills) for log_p_z in log_p_z_list]
                    print('log_p_z: %s' % log_p_z)
                    self._p_z = utils._softmax(log_p_z)


            logger.push_prefix('Epoch #%d | ' % epoch)
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

            env.terminate()


################################################################################


    # def _parallel_eval(self, z):
    #     # Make policy deterministic
    #     fixed_z_policy = DeterministicFixedOptionPolicy(self._policy, 
    #                                                     self._num_skills, z)
    #     # Evaluate skill
    #     self._eval_env._wrapped_env.env.initialize(seed_task=SEED_TASK)
    #     paths = rollouts(env=self._eval_env, 
    #                      policy=fixed_z_policy,
    #                      path_length=self._max_path_length, 
    #                      n_paths=1,
    #                      render=False)
    #     # Extract trajectory from paths
    #     traj_main = paths[0]['env_infos']['position']
    #     traj_aux = paths[0]['env_infos']['position_aux']
    #     # list_traj_main.append(traj_main)
    #     # list_traj_aux.append(traj_aux)
    #     # Extract outcomes from paths
    #     trial_outcome = self._eval_env._wrapped_env.env.finalize(
    #                                         state=paths[0]['last_obs'], 
    #                                         rew_list=paths[0]['rewards'],
    #                                         traj=traj_main)
    #     # list_outcomes.append(trial_outcome)
    #     # Extract and convert bd from paths
    #     self.bd_metric.restart()
    #     if self.bd_metric.metric_name == 'contact_grid':
    #         [self.bd_metric.update({'contact_objects': idict}) \
    #             for idict in paths[0]['env_infos']['contact_objects']]
    #     trial_metric = self.bd_metric.calculate(traj=traj_main,
    #                                             traj_aux=traj_aux)
    #     # list_skill_bd.append(np.argmax(trial_metric))
    #     return np.argmax(trial_metric), trial_outcome, traj_main, traj_aux




    # def sample_skills_to_bd(self, final=False, **kwargs):
    #     """
    #         Evaluate all the latent skills 
    #         and extract the behaviour descriptors
    #     """
    #     list_traj_main = []
    #     list_traj_aux = []
    #     list_outcomes = []
    #     list_skill_bd = []

    #     eval_time = time.time()
                                          
    
    #     with Pool(processes=10) as pool:  
    #         outputs = pool.map(self._parallel_eval, np.arange(self._num_skills))


    #     pdb.set_trace()


    #     eval_time = time.time()-eval_time

    #     # Extract unique data
    #     unique_bds, unique_idx = np.unique(list_skill_bd, return_index=True)
    #     n_bd = len(unique_bds)
    #     unique_outcomes = np.array(list_outcomes)[unique_idx]
    #     unique_traj_main = itemgetter(*unique_idx)(list_traj_main)
    #     unique_traj_aux = itemgetter(*unique_idx)(list_traj_aux)

    #     # Save to file at exact point: nump epoch, num episodes
    #     self._write_discovery(eval_time=eval_time, unique_bds=unique_bds, 
    #                           unique_outcomes=unique_outcomes, **kwargs)

    #     # Save data at the end
    #     if final==True:
    #         unique_bds_1hot = np.zeros((n_bd, self.bd_metric.metric_size))
    #         unique_bds_1hot[np.arange(n_bd), unique_bds] = 1
    #         self._save_dataset(**{"outcomes": unique_outcomes, 
    #                               "traj_main": unique_traj_main,
    #                               "traj_aux": unique_traj_aux,
    #                               "metric_bd": unique_bds_1hot})

