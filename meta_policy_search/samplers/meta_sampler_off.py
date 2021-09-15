from meta_policy_search.samplers.base import Sampler
from meta_policy_search.samplers.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from meta_policy_search.utils import utils, logger

from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools



'''
idx // self.envs_per_task, 
np.asarray(running_paths[idx]["observations"]), 
np.asarray(running_paths[idx]["actions"]),
np.asarray(running_paths[idx]["rewards"]),
np.asarray(running_paths[idx]["next_observations"]),
np.asarray(running_paths[idx]["dones"]),
utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"])
'''
class ReplayBuffer(object):
    def __init__(self, buffer_length, num_tasks, ob_dim, ac_dim, traj_length):
        self.buffer_length  = int(buffer_length)
        self.num_tasks      = num_tasks

        self.traj_len       = traj_length
        
        self.ob_buffs          = OrderedDict()
        self.ac_buffs          = OrderedDict()
        self.rew_buffs         = OrderedDict()
        self.next_ob_buffs     = OrderedDict()
        self.done_buffs        = OrderedDict()



        self.env_info_buffs    = OrderedDict()
        self.agent_info_buffs  = OrderedDict()

        self.return_buffs      = OrderedDict()



        for i in range(self.num_tasks):
            self.ob_buffs[i]      = np.zeros((self.buffer_length, ob_dim), dtype=np.float32)
            self.ac_buffs[i]      = np.zeros((self.buffer_length, ac_dim), dtype=np.float32)
            self.rew_buffs[i]     = np.zeros(self.buffer_length, dtype=np.float32)
            self.next_ob_buffs[i] = np.zeros((self.buffer_length, ob_dim), dtype=np.float32)            
            self.done_buffs[i]    = np.zeros(self.buffer_length, dtype=np.uint8)

            self.return_buffs[i]  = np.zeros(self.buffer_length, dtype=np.float32)

            self.env_info_buffs[i]          = dict(reward_run=np.zeros(self.buffer_length, dtype=np.float32),          reward_ctrl=np.zeros(self.buffer_length, dtype=np.float32))
            self.agent_info_buffs[i]        = dict(mean      =np.zeros((self.buffer_length, ac_dim), dtype=np.float32),log_std=np.zeros((self.buffer_length, ac_dim), dtype=np.float32))


        self.filled_i = [0]* self.num_tasks # index of first empty location in buffer (last index when full)
        self.curr_i   = [0]* self.num_tasks    # current index to write to (ovewrite oldest data)


    def push(self, task_id, observations, actions, rewards, next_observations, dones, env_infos, agent_infos, returns):
        nentries = observations.shape[0]  # handle multiple parallel environments


        if self.curr_i[task_id] + nentries > self.buffer_length:
            rollover = self.buffer_length - self.curr_i[task_id] # num of indices to roll over

            self.ob_buffs[task_id]       = np.roll(self.ob_buffs[task_id],
                                                        rollover, axis=0)
            self.ac_buffs[task_id]       = np.roll(self.ac_buffs[task_id],
                                                        rollover, axis=0)
            self.rew_buffs[task_id]      = np.roll(self.rew_buffs[task_id],
                                                        rollover)
            self.next_ob_buffs[task_id]  = np.roll(self.next_ob_buffs[task_id], 
                                                        rollover, axis=0)
            self.done_buffs[task_id]     = np.roll(self.done_buffs[task_id],
                                                        rollover)

            self.return_buffs[task_id]   = np.roll(self.return_buffs[task_id],
                                                        rollover)

            self.env_info_buffs[task_id]['reward_run']      = np.roll(self.env_info_buffs[task_id]['reward_run'],
                                                        rollover)
            self.env_info_buffs[task_id]['reward_ctrl']     = np.roll(self.env_info_buffs[task_id]['reward_ctrl'],
                                                        rollover)
            self.agent_info_buffs[task_id]['mean']          = np.roll(self.agent_info_buffs[task_id]['mean'],
                                                        rollover, axis=0)
            self.agent_info_buffs[task_id]['log_std']       = np.roll(self.agent_info_buffs[task_id]['log_std'],
                                                        rollover, axis=0)



            self.curr_i[task_id] = 0
            self.filled_i[task_id] = self.buffer_length

        self.ob_buffs[task_id][self.curr_i[task_id]:self.curr_i[task_id] + nentries]        = observations
        self.ac_buffs[task_id][self.curr_i[task_id]:self.curr_i[task_id] + nentries]        = actions
        self.rew_buffs[task_id][self.curr_i[task_id]:self.curr_i[task_id] + nentries]       = rewards
        self.next_ob_buffs[task_id][self.curr_i[task_id]:self.curr_i[task_id] + nentries]   = next_observations
        self.done_buffs[task_id][self.curr_i[task_id]:self.curr_i[task_id] + nentries]      = dones

        self.return_buffs[task_id][self.curr_i[task_id]:self.curr_i[task_id] + nentries]    = returns
  

        self.env_info_buffs[task_id]['reward_run'][self.curr_i[task_id]:self.curr_i[task_id] + nentries]      = env_infos['reward_run']
        self.env_info_buffs[task_id]['reward_ctrl'][self.curr_i[task_id]:self.curr_i[task_id] + nentries]     = env_infos['reward_ctrl']
        self.agent_info_buffs[task_id]['mean'][self.curr_i[task_id]:self.curr_i[task_id] + nentries]          = agent_infos['mean']
        self.agent_info_buffs[task_id]['log_std'][self.curr_i[task_id]:self.curr_i[task_id] + nentries]       = agent_infos['log_std']

        self.curr_i[task_id] += nentries
        if self.filled_i[task_id] < self.buffer_length:
            self.filled_i[task_id] += nentries
        if self.curr_i[task_id] == self.buffer_length:
            self.curr_i[task_id] = 0





    def sample(self, tasks_id, N):
        

        paths = OrderedDict()
        for meta_id, task_id in enumerate(tasks_id):

            if N > (min(self.filled_i) // self.traj_len)-1:
                inds = np.random.choice(np.arange((min(self.filled_i) // self.traj_len)-1), size=N, replace=True)
            elif N <= (min(self.filled_i) // self.traj_len)-1:
                inds = np.random.choice(np.arange((min(self.filled_i) // self.traj_len)-1), size=N, replace=False)
            paths[meta_id] = []
            for traj_id in list(inds):
                #print(self.done_buffs[task_id][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)][-1])
                paths[meta_id].append(dict(
                            observations       = self.ob_buffs[task_id][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)],
                            actions            = self.ac_buffs[task_id][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)],
                            rewards            = self.rew_buffs[task_id][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)],
                            next_observations  = self.next_ob_buffs[task_id][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)],
                            env_infos          = dict(reward_run  = self.env_info_buffs[task_id]['reward_run'][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)],
                                                      reward_ctrl = self.env_info_buffs[task_id]['reward_ctrl'][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)]),
                            agent_infos        = dict(mean        = self.agent_info_buffs[task_id]['mean'][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)],
                                                       log_std     = self.agent_info_buffs[task_id]['log_std'][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)]),
  
                            returns            = self.return_buffs[task_id][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)],
                            task_ids           = np.expand_dims(np.eye(self.num_tasks)[task_id], axis=0).repeat(self.traj_len, axis=0),
                            dones              = self.done_buffs[task_id][np.arange(traj_id*self.traj_len,traj_id*self.traj_len+self.traj_len)],
                        ))
        
        return paths


class MetaSampler_off(Sampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(
            self,
            env,
            policy,
            rollouts_per_meta_task,
            meta_batch_size,
            max_path_length,
            envs_per_task=None,
            parallel=False,
            buffer_length = 1e4,
            discount = 0.99,
            num_tasks=2
            ):
        super(MetaSampler_off, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task   = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.meta_batch_size = meta_batch_size
        self.total_samples   = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.parallel        = parallel
        self.total_timesteps_sampled = 0
        self.num_tasks       = num_tasks
   
        self.max_path_length  = max_path_length
        self.discount         = discount

        self.buffer_length    = buffer_length
        self.ob_dim           = np.prod(env.observation_space.shape)
        self.ac_dim           = np.prod(env.action_space.shape)

        self.buffer           = ReplayBuffer(self.buffer_length, self.num_tasks,
                                  self.ob_dim,
                                  self.ac_dim,
                                  self.max_path_length)

        # setup vectorized environment


        if self.parallel:
            self.vec_env = MetaParallelEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MetaIterativeEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)



    def set_seeds(self, seeds):
        self.vec_env.set_seeds(seeds)



    def update_tasks(self):
        """
        Samples a new goal for each meta task
        """
        tasks = self.env.sample_tasks(self.meta_batch_size)
        assert len(tasks) == self.meta_batch_size
        self.vec_env.set_tasks(tasks)


    def update_tasks_with_id(self):
         """
         Samples a new goal for each meta task
         """
         tasks, tasks_id = self.env.sample_tasks_with_id(self.meta_batch_size, return_id=True)
         assert len(tasks) == self.meta_batch_size
         self.vec_env.set_tasks(tasks)
         return tasks, tasks_id

    def obtain_samples(self, tasks_id, step_id, log=False, log_prefix=''):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns: 
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = OrderedDict()
        for i in range(self.meta_batch_size):
            paths[i] = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy

        # initial reset of envs
        obses = self.vec_env.reset()
        
        while n_samples < self.total_samples:
            
            # execute policy
            t = time.time()
            obs_per_task = np.split(np.asarray(obses), self.meta_batch_size)
            actions, agent_infos = policy.get_actions(obs_per_task)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            actions = np.concatenate(actions) # stack meta batch
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t
            

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)

            new_samples = 0
            for idx, observation, action, reward, next_observation, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, next_obses, env_infos, agent_infos,
                                                                                    dones):
                # append new samples to running paths
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["next_observations"].append(next_observation)
                running_paths[idx]["dones"].append(done)


                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)



                # if running path is done, add it to paths and empty the running path
                if done:
                    paths[idx // self.envs_per_task].append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))

                    discount_reward = utils.discount_cumsum(np.asarray(running_paths[idx]["rewards"]), self.discount)

                    if step_id == 0:
                        self.buffer.push(tasks_id[idx // self.envs_per_task], 
                                         np.asarray(running_paths[idx]["observations"]), 
                                         np.asarray(running_paths[idx]["actions"]),
                                         np.asarray(running_paths[idx]["rewards"]),
                                         np.asarray(running_paths[idx]["next_observations"]),
                                         np.asarray(running_paths[idx]["dones"]),
                                         utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                                         utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                                         discount_reward
                                        )
                
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()
                
            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
        return paths

    def _handle_info_dicts(self, agent_infos, env_infos):
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        else:
            assert len(agent_infos) == self.meta_batch_size
            assert len(agent_infos[0]) == self.envs_per_task
            agent_infos = sum(agent_infos, [])  # stack agent_infos

        assert len(agent_infos) == self.meta_batch_size * self.envs_per_task == len(env_infos)
        return agent_infos, env_infos


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], next_observations=[], dones=[], env_infos=[], agent_infos=[])
