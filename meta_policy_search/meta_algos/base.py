from meta_policy_search import utils
from meta_policy_search.policies.base import Policy
from meta_policy_search.critics.base import Critic
from meta_policy_search.values.base import Value_net


from collections import OrderedDict
import tensorflow as tf
import numpy as np


class MetaAlgo(object):
    """
    Base class for algorithms

    Args:
        policy (Policy) : policy object
    """

    def __init__(self, policy, critic_1, critic_2, baseline_value):
        assert isinstance(policy,  Policy)
        assert isinstance(critic_1, Critic)
        assert isinstance(critic_2, Critic)
        assert isinstance(baseline_value, Value_net)
       
        self.policy                      = policy
        self.critic_1                    = critic_1
        self.critic_2                    = critic_2
        self.baseline_value              = baseline_value

        self._optimization_keys          = None
        self._critic_optimization_keys   = None

    def build_graph(self):
        """
        Creates meta-learning computation graph

        Pseudocode::

            for task in meta_batch_size:
                make_vars
                init_dist_info_sym
            for step in num_grad_steps:
                for task in meta_batch_size:
                    make_vars
                    update_dist_info_sym
            set objectives for optimizer
        """
        raise NotImplementedError

    def make_vars(self, prefix=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable

        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task
        """
        raise NotImplementedError

    def _adapt_sym(self, surr_obj, params_var):
        """
        Creates the symbolic representation of the tf policy after one gradient step towards the surr_obj

        Args:
            surr_obj (tf_op) : tensorflow op for task specific (inner) objective
            params_var (dict) : dict of placeholders for current policy params

        Returns:
            (dict):  dict of tf.Tensors for adapted policy params
        """
        raise NotImplementedError

    def _adapt(self, samples):
        """
        Performs MAML inner step for each task and stores resulting gradients # (in the policy?)

        Args:
            samples (list) : list of lists of samples (each is a dict) split by meta task

        Returns:
            None
        """
        raise NotImplementedError

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step for each task

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        raise NotImplementedError



class MAMLAlgo(MetaAlgo):
    """
    Provides some implementations shared between all MAML algorithms
    
    Args:
        policy (Policy): policy object
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable
    """
    def __init__(self, policy, critic_1, critic_2, baseline_value, inner_lr=0.1, meta_batch_size=20, num_inner_grad_steps=1, trainable_inner_step_size=False):
        super(MAMLAlgo, self).__init__(policy, critic_1, critic_2, baseline_value)

        assert type(num_inner_grad_steps) and num_inner_grad_steps >= 0
        assert type(meta_batch_size) == int

        self.inner_lr = float(inner_lr)
        self.meta_batch_size = meta_batch_size
        self.num_inner_grad_steps = num_inner_grad_steps
        self.trainable_inner_step_size = trainable_inner_step_size #TODO: make sure this actually works

        self.adapt_input_ph_dict = None
        self.adapted_policies_params = None
        self.step_sizes = None

    def _make_input_placeholders(self, prefix=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable

        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task, 
            and for convenience, a list containing all placeholders created
        """
        obs_phs, action_phs, adv_phs, dist_info_phs, dist_info_phs_list = [], [], [], [], []
        dist_info_specs = self.policy.distribution.dist_info_specs

        all_phs_dict = OrderedDict()

        for task_id in range(self.meta_batch_size):
            # observation ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.policy.obs_dim], name='obs' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s'%(prefix, task_id, 'observations')] = ph
            obs_phs.append(ph)

            # action ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.policy.action_dim], name='action' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'actions')] = ph
            action_phs.append(ph)

            # advantage ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'advantages')] = ph
            adv_phs.append(ph)

            # distribution / agent info
            dist_info_ph_dict = {}
            for info_key, shape in dist_info_specs:
                ph = tf.placeholder(dtype=tf.float32, shape=[None] + list(shape), name='%s_%s_%i' % (info_key, prefix, task_id))
                all_phs_dict['%s_task%i_agent_infos/%s' % (prefix, task_id, info_key)] = ph
                dist_info_ph_dict[info_key] = ph
            dist_info_phs.append(dist_info_ph_dict)

        return obs_phs, action_phs, adv_phs, dist_info_phs, all_phs_dict

    def _make_input_placeholders_off_value(self, prefix=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable

        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task, 
            and for convenience, a list containing all placeholders created
        """
        obs_phs_off_value, action_phs_off_value, value_adv_phs_off_value, dist_info_phs_off_value, dist_info_phs_list_off_value = [], [], [], [], []
        dist_info_specs      = self.policy.distribution.dist_info_specs

        all_phs_dict_off_value = OrderedDict()

        for task_id in range(self.meta_batch_size):
            # observation ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.policy.obs_dim], name='obs' + '_' + prefix + '_' + str(task_id))
            all_phs_dict_off_value['%s_task%i_%s'%(prefix, task_id, 'observations')] = ph
            obs_phs_off_value.append(ph)

            # action ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.policy.action_dim], name='action' + '_' + prefix + '_' + str(task_id))
            all_phs_dict_off_value['%s_task%i_%s' % (prefix, task_id, 'actions')] = ph
            action_phs_off_value.append(ph)

            # advantage ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None], name='value_advantage' + '_' + prefix + '_' + str(task_id))
            all_phs_dict_off_value['%s_task%i_%s' % (prefix, task_id, 'value_advantages')] = ph
            value_adv_phs_off_value.append(ph)

            # distribution / agent info
            dist_info_ph_dict = {}
            for info_key, shape in dist_info_specs:
                ph = tf.placeholder(dtype=tf.float32, shape=[None] + list(shape), name='%s_%s_%i' % (info_key, prefix, task_id))
                all_phs_dict_off_value['%s_task%i_agent_infos/%s' % (prefix, task_id, info_key)] = ph
                dist_info_ph_dict[info_key] = ph
            dist_info_phs_off_value.append(dist_info_ph_dict)

        return obs_phs_off_value, action_phs_off_value, value_adv_phs_off_value, dist_info_phs_off_value, all_phs_dict_off_value









    def _make_input_placeholders_critic(self, prefix=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable

        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task, 
            and for convenience, a list containing all placeholders created
        """
        obs_phs, actions_phs, rews_phs, next_obs_phs, next_actions_phs, dones_phs, discounts_phs, task_ids_phs, next_task_ids_phs, current_q_values_1_phs, current_q_values_2_phs, next_q_values_1_phs, next_q_values_2_phs, dist_infos_phs = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
        dist_info_specs    = self.policy.distribution.dist_info_specs

        all_phs_dict       = OrderedDict()


        for task_id in range(self.meta_batch_size):
            # observation ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.policy.obs_dim],    name='ob' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s'%(prefix, task_id, 'observations')] = ph
            obs_phs.append(ph)

            # action ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.policy.action_dim], name='action' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'actions')] = ph
            actions_phs.append(ph)

            # reward ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None], name='reward' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'rewards')] = ph
            rews_phs.append(ph)

            # next_observation ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.policy.obs_dim], name='next_ob' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s'%(prefix, task_id, 'next_observations')] = ph
            next_obs_phs.append(ph)


            # next action ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.policy.action_dim], name='next_action' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'next_actions')] = ph
            next_actions_phs.append(ph)

            # done ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None], name='done' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'dones')] = ph
            dones_phs.append(ph)

            # discount ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None], name='discount' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'discounts')] = ph
            discounts_phs.append(ph)

            # true_task_id ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.critic_1.task_id_dim], name='task_id' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s'%(prefix, task_id, 'task_ids')] = ph
            task_ids_phs.append(ph)

            # true_next_task_id ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.critic_1.task_id_dim], name='next_task_id' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s'%(prefix, task_id, 'next_task_ids')] = ph
            next_task_ids_phs.append(ph)


            # current_q_value_1 ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None], name='current_q_value_1' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'current_q_values_1')] = ph
            current_q_values_1_phs.append(ph)


            # current_q_value_2 ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None], name='current_q_value_2' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'current_q_values_2')] = ph
            current_q_values_2_phs.append(ph)

            # next_q_value_1 ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None], name='next_q_value_1' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'next_q_values_1')] = ph
            next_q_values_1_phs.append(ph)


            # current_q_value_2 ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None], name='next_q_value_2' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'next_q_values_2')] = ph
            next_q_values_2_phs.append(ph)

            # distribution / agent info
            dist_info_ph_dict = {}
            for info_key, shape in dist_info_specs:
                ph = tf.placeholder(dtype=tf.float32, shape=[None] + list(shape), name='%s_%s_%i' % (info_key, prefix, task_id))
                all_phs_dict['%s_task%i_agent_infos/%s' % (prefix, task_id, info_key)] = ph
                dist_info_ph_dict[info_key] = ph
            dist_infos_phs.append(dist_info_ph_dict)

        return obs_phs, actions_phs, rews_phs, next_obs_phs, next_actions_phs, dones_phs, discounts_phs, task_ids_phs, next_task_ids_phs, current_q_values_1_phs, current_q_values_2_phs, next_q_values_1_phs, next_q_values_2_phs,  dist_infos_phs, all_phs_dict



    def _adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
        raise NotImplementedError



    def _build_inner_adaption_off_critic(self):
        """
        Creates the symbolic graph for the one-step inner gradient update (It'll be called several times if
        more gradient steps are needed)

        Args:
            some placeholders

        Returns:
            adapted_policies_params (list): list of Ordered Dict containing the symbolic post-update parameters
            adapt_input_list_ph (list): list of placeholders

        """
        obs_phs,  action_phs,  adv_phs,   dist_info_old_phs,  adapt_input_ph_dict     = self._make_input_placeholders('adapt')

        obs_phs_off_critic, \
        actions_phs_off_critic,\
        rews_phs_off_critic, \
        next_obs_phs_off_critic, \
        next_actions_phs_off_critic,  \
        dones_phs_off_critic,\
        discounts_phs_off_critic,\
        task_ids_phs_off_critic,  \
        next_task_ids_phs_off_critic,\
        current_q_values_1_phs_off_critic,   \
        current_q_values_2_phs_off_critic,   \
        next_q_values_1_phs_off_critic,   \
        next_q_values_2_phs_off_critic,   \
        dist_infos_old_phs_off_critic, \
        adapt_input_ph_dict_off_critic                                                 = self._make_input_placeholders_critic('adapt1')


        adapted_policies_params = []

        for i in range(self.meta_batch_size):
            with tf.variable_scope("adapt_task_%i" % i):
                with tf.variable_scope("adapt_objective"):
                    distribution_info_new            = self.policy.distribution_info_sym(obs_phs[i],
                                                                                  params=self.policy.policies_params_phs[i])


                    # inner surrogate objective
                    surr_obj_adapt                   = self._adapt_objective_sym(    action_phs[i], adv_phs[i],
                                                                              dist_info_old_phs[i], distribution_info_new)

            with tf.variable_scope("adapt1_task_%i" % i):
                with tf.variable_scope("adapt1_objective"):

                    distribution_infos_new_off_critic = self.policy.distribution_info_sym(obs_phs_off_critic[i],
                                                                                  params=self.policy.policies_params_phs[i])
                    
                    # inner off_policy surrogate objective
                    surr_obj_adapt_off               = self._adapt_objective_sym_off_critic(obs_phs_off_critic[i],
                                                                                            actions_phs_off_critic[i], 
                                                                                            rews_phs_off_critic[i],
                                                                                            next_obs_phs_off_critic[i],
                                                                                            next_actions_phs_off_critic[i],

                                                                                            dones_phs_off_critic[i],
                                                                                            discounts_phs_off_critic[i],
                                                                                            task_ids_phs_off_critic[i],
                                                                                            next_task_ids_phs_off_critic[i],

                                                                                            current_q_values_1_phs_off_critic[i],
                                                                                            current_q_values_2_phs_off_critic[i],
                                                                                            next_q_values_1_phs_off_critic[i],
                                                                                            next_q_values_2_phs_off_critic[i],
                                                                                            dist_infos_old_phs_off_critic[i], 
                                                                                            distribution_infos_new_off_critic)  

            # get tf operation for adapted (post-update) policy
            with tf.variable_scope("adapt_step"):
                all_surr_obj_adapt   = 0.5 * surr_obj_adapt + 0.5 * surr_obj_adapt_off
                adapted_policy_param = self._adapt_sym(all_surr_obj_adapt, self.policy.policies_params_phs[i])

            adapted_policies_params.append(adapted_policy_param)

        return adapted_policies_params, adapt_input_ph_dict, adapt_input_ph_dict_off_critic



    def _build_inner_adaption_off_value(self):
        """
        Creates the symbolic graph for the one-step inner gradient update (It'll be called several times if
        more gradient steps are needed)
        Args:
            some placeholders
        Returns:
            adapted_policies_params (list): list of Ordered Dict containing the symbolic post-update parameters
            adapt_input_list_ph (list): list of placeholders
        """
        obs_phs,           action_phs,           adv_phs,                 dist_info_old_phs,           adapt_input_ph_dict           = self._make_input_placeholders('adapt')

        obs_phs_off_value, action_phs_off_value, value_adv_phs_off_value, dist_info_old_phs_off_value, adapt_input_ph_dict_off_value = self._make_input_placeholders_off_value('adapt1')


        adapted_policies_params = []

        for i in range(self.meta_batch_size):
            with tf.variable_scope("adapt_task_%i" % i):
                with tf.variable_scope("adapt_objective"):
                    distribution_info_new           = self.policy.distribution_info_sym(obs_phs[i],
                                                                              params=self.policy.policies_params_phs[i])


                    # inner surrogate objective
                    surr_obj_adapt                  = self._adapt_objective_sym(action_phs[i], adv_phs[i],
                                                               dist_info_old_phs[i], distribution_info_new)

            with tf.variable_scope("adapt1_task_%i" % i):
                with tf.variable_scope("adapt1_objective"):

                    distribution_info_new_off_value = self.policy.distribution_info_sym(obs_phs_off_value[i],
                                                                              params=self.policy.policies_params_phs[i])
                    
                    # inner off_policy surrogate objective
                    surr_obj_adapt_off_value        = self._adapt_objective_sym_off_value(action_phs_off_value[i],         value_adv_phs_off_value[i],
                                                                                    dist_info_old_phs_off_value[i], distribution_info_new_off_value)  

            # get tf operation for adapted (post-update) policy
            with tf.variable_scope("adapt_step"):
                all_surr_obj_adapt   = 0.5 * surr_obj_adapt + 0.5 * surr_obj_adapt_off_value
                adapted_policy_param = self._adapt_sym(all_surr_obj_adapt, self.policy.policies_params_phs[i])

            adapted_policies_params.append(adapted_policy_param)

        return adapted_policies_params, adapt_input_ph_dict, adapt_input_ph_dict_off_value








    def _build_inner_adaption(self):
        """
        Creates the symbolic graph for the one-step inner gradient update (It'll be called several times if
        more gradient steps are needed)

        Args:
            some placeholders

        Returns:
            adapted_policies_params (list): list of Ordered Dict containing the symbolic post-update parameters
            adapt_input_list_ph (list): list of placeholders

        """
        test_obs_phs, test_action_phs, test_adv_phs, test_dist_info_old_phs, test_adapt_input_ph_dict = self._make_input_placeholders('test_adapt')

        test_adapted_policies_params = []

        for i in range(self.meta_batch_size):
            with tf.variable_scope("test_adapt_task_%i" % i):
                with tf.variable_scope("test_adapt_objective"):
                    test_distribution_info_new = self.policy.distribution_info_sym(test_obs_phs[i],
                                                                                   params=self.policy.policies_params_phs[i])

                    # inner surrogate objective
                    test_surr_obj_adapt        = self._adapt_objective_sym(test_action_phs[i], test_adv_phs[i],
                                                                           test_dist_info_old_phs[i], test_distribution_info_new)

                # get tf operation for adapted (post-update) policy
                with tf.variable_scope("test_adapt_step"):
                    test_adapted_policy_param = self._adapt_sym(test_surr_obj_adapt, self.policy.policies_params_phs[i])
                test_adapted_policies_params.append(test_adapted_policy_param)

        return test_adapted_policies_params, test_adapt_input_ph_dict


    def _adapt_sym(self, surr_obj, params_var):
        """
        Creates the symbolic representation of the tf policy after one gradient step towards the surr_obj

        Args:
            surr_obj (tf_op) : tensorflow op for task specific (inner) objective
            params_var (dict) : dict of tf.Tensors for current policy params

        Returns:
            (dict):  dict of tf.Tensors for adapted policy params
        """
        # TODO: Fix this if we want to learn the learning rate (it isn't supported right now).
        update_param_keys = list(params_var.keys())

        grads = tf.gradients(surr_obj, [params_var[key] for key in update_param_keys])
        gradients = dict(zip(update_param_keys, grads))

        # gradient descent
        adapted_policy_params = [params_var[key] - tf.multiply(self.step_sizes[key], gradients[key])
                          for key in update_param_keys]

        adapted_policy_params_dict = OrderedDict(zip(update_param_keys, adapted_policy_params))

        return adapted_policy_params_dict

    def _adapt(self, samples):
        """
        Performs MAML inner step for each task and stores the updated parameters in the policy

        Args:
            samples (list) : list of dicts of samples (each is a dict) split by meta task

        """
        assert len(samples) == self.meta_batch_size
        assert [sample_dict.keys() for sample_dict in samples]
        sess = tf.get_default_session()

        # prepare feed dict
        test_input_dict       = self._extract_input_dict(samples, self._optimization_keys, prefix='test_adapt')
        test_input_ph_dict    = self.test_adapt_input_ph_dict

        test_feed_dict_inputs = utils.create_feed_dict(placeholder_dict=test_input_ph_dict, value_dict=test_input_dict)
        test_feed_dict_params = self.policy.policies_params_feed_dict

        test_feed_dict = {**test_feed_dict_inputs, **test_feed_dict_params}  # merge the two feed dicts

        # compute the post-update / adapted policy parameters
        test_adapted_policies_params_vals = sess.run(self.test_adapted_policies_params, feed_dict=test_feed_dict)

        # store the new parameter values in the policy
        self.policy.update_task_parameters(test_adapted_policies_params_vals)


    def _adapt_off_critic(self, on_samples, off_critic_samples):
        """
        Performs MAML inner step for each task and stores the updated parameters in the policy

        Args:
            samples (list) : list of dicts of samples (each is a dict) split by meta task

        """
        assert len(on_samples)         == self.meta_batch_size
        assert len(off_critic_samples) == self.meta_batch_size

        assert [sample_dict.keys() for sample_dict in on_samples]
        assert [sample_dict.keys() for sample_dict in off_critic_samples]


        sess = tf.get_default_session()

        # prepare feed dict
        input_dict_on                   = self._extract_input_dict(on_samples,         self._optimization_keys, prefix='adapt')
        input_dict_off_critic           = self._extract_input_dict(off_critic_samples, self._critic_optimization_keys, prefix='adapt1')


        #print(input_dict_off)

        input_ph_dict_on                = self.adapt_input_ph_dict
        input_ph_dict_off_critic        = self.adapt_input_ph_dict_off_critic


        #print(input_dict_off_critic)
        #print(input_ph_dict_off_critic)

        feed_dict_inputs_on             = utils.create_feed_dict(placeholder_dict=input_ph_dict_on,  value_dict=input_dict_on)
        feed_dict_inputs_off_critic     = utils.create_feed_dict(placeholder_dict=input_ph_dict_off_critic, value_dict=input_dict_off_critic)


        feed_dict_inputs_all            = {**feed_dict_inputs_on, **feed_dict_inputs_off_critic}
        
        feed_dict_params                = self.policy.policies_params_feed_dict


 
        feed_dict                       = {**feed_dict_inputs_all, **feed_dict_params}  # merge the two feed dicts

        
        # compute the post-update / adapted policy parameters
        adapted_policies_params_vals    = sess.run(self.adapted_policies_params, feed_dict=feed_dict)

        # store the new parameter values in the policy
        self.policy.update_task_parameters(adapted_policies_params_vals)



    def _adapt_off_value(self, on_samples, off_value_samples):
        """
        Performs MAML inner step for each task and stores the updated parameters in the policy

        Args:
            samples (list) : list of dicts of samples (each is a dict) split by meta task

        """
        assert len(on_samples)         == self.meta_batch_size
        assert len(off_value_samples)  == self.meta_batch_size

        assert [sample_dict.keys() for sample_dict in on_samples]
        assert [sample_dict.keys() for sample_dict in off_value_samples]


        sess                            = tf.get_default_session()

        # prepare feed dict
        input_dict_on                   = self._extract_input_dict(on_samples,         self._optimization_keys, prefix='adapt')
        input_dict_off_value            = self._extract_input_dict(off_value_samples,  self._value_optimization_keys, prefix='adapt1')


        #print(input_dict_off)

        input_ph_dict_on                = self.adapt_input_ph_dict
        input_ph_dict_off_value         = self.adapt_input_ph_dict_off_value


        #print(input_dict_off_critic)
        #print(input_ph_dict_off_critic)

        feed_dict_inputs_on             = utils.create_feed_dict(placeholder_dict=input_ph_dict_on,  value_dict=input_dict_on)
        feed_dict_inputs_off_value      = utils.create_feed_dict(placeholder_dict=input_ph_dict_off_value, value_dict=input_dict_off_value)


        feed_dict_inputs_all            = {**feed_dict_inputs_on, **feed_dict_inputs_off_value}
        
        feed_dict_params                = self.policy.policies_params_feed_dict


 
        feed_dict                       = {**feed_dict_inputs_all, **feed_dict_params}  # merge the two feed dicts

        
        # compute the post-update / adapted policy parameters
        adapted_policies_params_vals    = sess.run(self.adapted_policies_params, feed_dict=feed_dict)

        # store the new parameter values in the policy
        self.policy.update_task_parameters(adapted_policies_params_vals)


    def _extract_input_dict(self, samples_data_meta_batch, keys, prefix=''):
        """
        Re-arranges a list of dicts containing the processed sample data into a OrderedDict that can be matched
        with a placeholder dict for creating a feed dict

        Args:
            samples_data_meta_batch (list) : list of dicts containing the processed data corresponding to each meta-task
            keys (list) : a list of keys that should exist in each dict and whose values shall be extracted
            prefix (str): prefix to prepend the keys in the resulting OrderedDict

        Returns:
            OrderedDict containing the data from all_samples_data. The data keys follow the naming convention:
                '<prefix>_task<task_number>_<key_name>'
        """
        assert len(samples_data_meta_batch) == self.meta_batch_size

        input_dict = OrderedDict()

        for meta_task in range(self.meta_batch_size):
            extracted_data = utils.extract(
                samples_data_meta_batch[meta_task], *keys
            )

            # iterate over the desired data instances and corresponding keys
            for j, (data, key) in enumerate(zip(extracted_data, keys)):
                if isinstance(data, dict):
                    # if the data instance is a dict -> iterate over the items of this dict
                    for k, d in data.items():
                        assert isinstance(d, np.ndarray)
                        input_dict['%s_task%i_%s/%s' % (prefix, meta_task, key, k)] = d

                elif isinstance(data, np.ndarray):
                    input_dict['%s_task%i_%s'%(prefix, meta_task, key)] = data
                else:
                    raise NotImplementedError
        return input_dict

    def _extract_input_dict_meta_op(self, all_samples_data, keys, keys_value):
        """
        Creates the input dict for all the samples data required to perform the meta-update

        Args:
            all_samples_data (list):list (len = num_inner_grad_steps + 1) of lists (len = meta_batch_size) containing
                                    dicts that hold processed samples data
            keys (list): a list of keys (str) that should exist in each dict and whose values shall be extracted

        Returns:

        """
        assert len(all_samples_data) == self.num_inner_grad_steps + 2

        meta_op_input_dict = OrderedDict()
        for step_id, samples_data in enumerate(all_samples_data):  # these are the gradient steps
            if step_id   != 2:
                dict_input_dict_step = self._extract_input_dict(samples_data, keys, prefix='step%i'%step_id)
            elif step_id == 2:
                dict_input_dict_step = self._extract_input_dict(samples_data, keys_value, prefix='step%i'%step_id)

            meta_op_input_dict.update(dict_input_dict_step)

        return meta_op_input_dict

    def _create_step_size_vars(self):
        # Step sizes
        with tf.variable_scope('inner_step_sizes'):
            step_sizes = dict()
            for key, param in self.policy.policy_params.items():
                shape = param.get_shape().as_list()
                init_stepsize = np.ones(shape, dtype=np.float32) * self.inner_lr
                step_sizes[key] = tf.Variable(initial_value=init_stepsize,
                                              name='%s_step_size' % key,
                                              dtype=tf.float32, trainable=self.trainable_inner_step_size)
        return step_sizes
