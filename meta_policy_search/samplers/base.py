from meta_policy_search.utils import utils, logger
import numpy as np
import wandb

class Sampler(object):
    """
    Sampler interface

    Args:
        env (gym.Env) : environment object
        policy (meta_policy_search.policies.policy) : policy object
        batch_size (int) : number of trajectories per task
        max_path_length (int) : max number of steps per trajectory
    """

    def __init__(self, env, policy, batch_size, max_path_length):
        assert hasattr(env, 'reset') and hasattr(env, 'step')

        self.env             = env
        self.policy          = policy
        self.batch_size      = batch_size
        self.max_path_length = max_path_length

    def obtain_samples(self):
        """
        Collect batch_size trajectories

        Returns: 
            (list) : A list of paths.
        """
        raise NotImplementedError


class SampleProcessor(object):
    """
    Sample processor interface
        - fits a reward baseline (use zero baseline to skip this step)
        - performs Generalized Advantage Estimation to provide advantages (see Schulman et al. 2015 - https://arxiv.org/abs/1506.02438)

    Args:
        baseline (Baseline) : a reward baseline object
        discount (float) : reward discount factor
        gae_lambda (float) : Generalized Advantage Estimation lambda
        normalize_adv (bool) : indicates whether to normalize the estimated advantages (zero mean and unit std)
        positive_adv (bool) : indicates whether to shift the (normalized) advantages so that they are all positive
    """

    def __init__(
            self,
            policy,
            critic_1,
            critic_2,
            baseline,
            baseline_value,
            baseline_value_fit_style,
            discount=0.99,
            gae_lambda=1,
            normalize_adv=False,
            positive_adv=False,
            ):

        assert 0 <= discount <= 1.0, 'discount factor must be in [0,1]'
        assert 0 <= gae_lambda <= 1.0, 'gae_lambda must be in [0,1]'
        assert hasattr(baseline, 'fit') and hasattr(baseline, 'predict')
        

        self.policy                    = policy
        self.critic_1                  = critic_1
        self.critic_2                  = critic_2
        self.baseline                  = baseline
        self.baseline_value            = baseline_value
        self.baseline_value_fit_style  = baseline_value_fit_style

        self.discount                  = discount
        self.gae_lambda                = gae_lambda
        self.normalize_adv             = normalize_adv
        self.positive_adv              = positive_adv
        self.Step_1_AverageReturn      = []
        self.test_Step_1_AverageReturn = []

    def process_samples(self, paths, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths (list): A list of paths of size (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (dict) : Processed sample data of size [7] x (batch_size x max_path_length)
        """
        assert type(paths) == list, 'paths must be a list'
        assert paths[0].keys() >= {'observations', 'actions', 'rewards'}
        assert self.baseline, 'baseline must be specified - use self.build_sample_processor(baseline_obj)'

        # fits baseline, compute advantages and stack path data
        samples_data, paths = self._compute_samples_data(paths)

        # 7) log statistics if desired
        self._log_path_stats(paths, log=log, log_prefix='')

        assert samples_data.keys() >= {'observations', 'actions', 'rewards', 'advantages', 'returns'}
        return samples_data

    """ helper functions """

    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)
        for idx, path in enumerate(paths):
            path["returns"]                = utils.discount_cumsum(path["rewards"], self.discount)
            normalized_rewards_on          = (path["rewards"] - np.mean(path["rewards"])) / (np.std(path["rewards"]) +1e-8)
            path["discounted_rewards"]     = utils.discount_cumsum(normalized_rewards_on, self.discount)
            #path["discounted_rewards"]    =  path["returns"]

            #normalized_rewards              = (path["rewards"] - np.mean(path["rewards"])) / (np.std(path["rewards"]) +1e-7)

        # 2) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths, target_key="returns")
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 3) compute advantages and adjusted rewards
        paths              = self._compute_advantages(paths, all_path_baselines)

        '''
        print(paths)
        time.sleep(100)
        paths              = self._compute_advantages_off_value(paths)
        '''

        # 4) stack path data
        observations, actions, rewards, returns, task_ids, discounted_rewards, advantages, env_infos, agent_infos = self._stack_path_data(paths)

        # 5) if desired normalize / shift advantages
        if self.normalize_adv:
            advantages = utils.normalize_advantages(advantages)
        if self.positive_adv:
            advantages = utils.shift_advantages_to_positive(advantages)

        # 6) create samples_data object
        samples_data = dict(
            observations        = observations,
            actions             = actions,
            rewards             = rewards,
            returns             = returns,

            discounted_rewards  = discounted_rewards,

            task_ids            = task_ids,
            advantages          = advantages,
            env_infos           = env_infos,
            agent_infos         = agent_infos,

        )

        return samples_data, paths



    def _compute_samples_data_off_value(self, paths):
        assert type(paths) == list

        paths = self._compute_advantages_off_value(paths)

        observations, actions, rewards, returns, value_advantages, env_infos, agent_infos = self._stack_path_data_value(paths)


        value_advantages         = utils.normalize_advantages(value_advantages)

        samples_data = dict(
            observations         = observations,
            actions              = actions,
            rewards              = rewards,

            returns              = returns,
            value_advantages     = value_advantages,

            env_infos            = env_infos,
            agent_infos          = agent_infos,
        )


        return samples_data, paths


    def _compute_samples_data_off_critic(self, paths):
        assert type(paths) == list


        # 1) compute discounted rewards (returns)

        # 2) predict the return baselines

        # 3) compute advantages and adjusted rewards
        paths = self._compute_current_q_values(paths)
        paths = self._compute_next_actions(paths)          
        paths = self._compute_next_q_values(paths)

        # 4) stack path data
        observations, actions, rewards, next_observations, next_actions, returns, current_q_values_1, current_q_values_2, next_q_values_1, next_q_values_2, task_ids, next_task_ids, dones, discounts, env_infos, agent_infos = self._stack_off_path_data(paths)



        # 6) create samples_data object
        samples_data = dict(
            observations         = observations,
            actions              = actions,
            rewards              = rewards,
            next_observations    = next_observations,
            next_actions         = next_actions,

            returns              = returns,

            current_q_values_1   = current_q_values_1.squeeze(-1),
            current_q_values_2   = current_q_values_2.squeeze(-1),
            
            next_q_values_1      = next_q_values_1.squeeze(-1),
            next_q_values_2      = next_q_values_2.squeeze(-1),


            task_ids             = task_ids,
            next_task_ids        = next_task_ids,
            
            dones                = dones,
            discounts            = discounts,

            env_infos            = env_infos,
            agent_infos          = agent_infos,
        )


        return samples_data, paths



    def _compute_advantages_off_value(self, paths):
        assert type(paths) == list

        for idx, path in enumerate(paths):

            states_value,_                  = self.baseline_value.get_state_values(path['observations'], path['task_ids'])
            next_states_value,_             = self.baseline_value.get_state_values(path['next_observations'], path['task_ids'])

            normalized_rewards              = (path["rewards"] - np.mean(path["rewards"])) / (np.std(path["rewards"]) +1e-8)
            states_value                    = states_value[0].squeeze(-1)
            next_states_value               = next_states_value[0].squeeze(-1)

            path["value_advantages"]        = normalized_rewards + self.discount * next_states_value - states_value

        return paths



    def _compute_next_actions(self, paths):
        assert type(paths) == list

        for idx, path in enumerate(paths):

            next_actions,_       = self.policy.get_actions_critic(path['observations'])
            path["next_actions"] = next_actions
            path["discounts"]    = np.array([self.discount]*next_actions.shape[0])

        return paths


    def _compute_current_q_values(self, paths):
        assert type(paths) == list

        for idx, path in enumerate(paths):

            current_q_values_1,_       = self.critic_1.get_q_values(path['observations'], path['actions'], path['task_ids'])
            path["current_q_values_1"] = current_q_values_1[0]
            current_q_values_2,_       = self.critic_2.get_q_values(path['observations'], path['actions'], path['task_ids'])
            path["current_q_values_2"] = current_q_values_2[0]

        return paths

    def _compute_next_q_values(self, paths):
        assert type(paths) == list

        for idx, path in enumerate(paths):

            next_q_values_1,_          = self.critic_1.get_next_q_values(path['next_observations'], path['next_actions'], path['task_ids'])
            path["next_q_values_1"] = next_q_values_1[0]
            next_q_values_2,_          = self.critic_2.get_next_q_values(path['next_observations'], path['next_actions'], path['task_ids'])
            path["next_q_values_2"] = next_q_values_2[0]

        return paths




    def _log_path_stats(self, paths, log=False, log_prefix=''):
        # compute log stats
        average_discounted_return = np.mean([path["returns"][0] for path in paths])
        undiscounted_returns      = [sum(path["rewards"]) for path in paths]

        if log == 'reward':
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))

        elif log == 'all' or log is True:
            logger.logkv(log_prefix + 'AverageDiscountedReturn', average_discounted_return)
            logger.logkv(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.logkv(log_prefix + 'NumTrajs', len(paths))
            logger.logkv(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.logkv(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.logkv(log_prefix + 'MinReturn', np.min(undiscounted_returns))

        return np.mean(undiscounted_returns)

    def _compute_advantages(self, paths, all_path_baselines):
        assert len(paths) == len(all_path_baselines)

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda)

        return paths

    def _compute_advantages_off(self, paths, all_path_baselines_ob, all_path_baselines_next_ob):
        assert len(paths) == len(all_path_baselines_ob)
        assert len(all_path_baselines_next_ob) == len(all_path_baselines_ob)


        for idx, path in enumerate(paths):
            path_baselines_ob       = all_path_baselines_ob[idx]
            path_baselines_next_ob  = all_path_baselines_next_ob[idx]

            advs = path["rewards"] + \
                     self.discount * path_baselines_next_ob- \
                     path_baselines_ob
            path["advantages"] = advs

        return paths

    def _stack_path_data(self, paths):
        observations        = np.concatenate([path["observations"] for path in paths])
        actions             = np.concatenate([path["actions"]      for path in paths])
        rewards             = np.concatenate([path["rewards"]      for path in paths])
        returns             = np.concatenate([path["returns"]      for path in paths])
        task_ids            = np.concatenate([path["task_ids"]      for path in paths])
        discounted_rewards  = np.concatenate([path["discounted_rewards"]      for path in paths])
        advantages          = np.concatenate([path["advantages"]   for path in paths])
        env_infos           = utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos         = utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
        return observations, actions, rewards, returns, task_ids, discounted_rewards, advantages, env_infos, agent_infos



    def _stack_path_data_value(self, paths):
        observations        = np.concatenate([path["observations"] for path in paths])
        actions             = np.concatenate([path["actions"]      for path in paths])
        rewards             = np.concatenate([path["rewards"]      for path in paths])
        returns             = np.concatenate([path["returns"]      for path in paths])

        value_advantages    = np.concatenate([path["value_advantages"]   for path in paths])
        env_infos           = utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos         = utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        return observations, actions, rewards, returns, value_advantages, env_infos, agent_infos

    def _stack_off_path_data(self, paths):
        observations           = np.concatenate([path["observations"] for path in paths])
        actions                = np.concatenate([path["actions"]      for path in paths])
        rewards                = np.concatenate([path["rewards"]      for path in paths])
        next_observations      = np.concatenate([path["next_observations"]   for path in paths])
        next_actions      = np.concatenate([path["next_actions"]   for path in paths])

        returns                = np.concatenate([path["returns"]      for path in paths])
        current_q_values_1     = np.concatenate([path["current_q_values_1"]   for path in paths])
        current_q_values_2     = np.concatenate([path["current_q_values_2"]   for path in paths])  

        next_q_values_1        = np.concatenate([path["next_q_values_1"]   for path in paths])
        next_q_values_2        = np.concatenate([path["next_q_values_2"]   for path in paths])  

        task_ids               = np.concatenate([path["task_ids"]   for path in paths])
        next_task_ids          = np.concatenate([path["task_ids"]   for path in paths])   

        dones                  = np.concatenate([path["dones"]   for path in paths])   
        discounts              = np.concatenate([path["discounts"]   for path in paths])   


        env_infos              = utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos            = utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
        return observations, actions, rewards, next_observations, next_actions, returns, current_q_values_1, current_q_values_2, next_q_values_1, next_q_values_2, task_ids, next_task_ids, dones, discounts, env_infos, agent_infos
