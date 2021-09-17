from meta_policy_search.utils import Serializable, logger
from meta_policy_search.utils.utils import remove_scope_from_name

from meta_policy_search.values.networks.mlp import create_mlp
from meta_policy_search.values.base import Value_net



import tensorflow as tf
import numpy as np
from collections import OrderedDict


class MLPValue_net(Value_net):
    """
    multi-layer perceptron value_net
    Provides functions for executing and updating value_net parameters

    Args:
        obs_dim (int)                       : dimensionality of the observation space 
        task_dim(int)

        name (str)                          : name of the policy used as tf variable scope
        hidden_sizes (tuple)                : tuple of integers specifying the hidden layer sizes of the MLP
        hidden_nonlinearity (tf.op)         : nonlinearity function of the hidden layers
        output_nonlinearity (tf.op or None) : nonlinearity function of the output layer


    """

    def __init__(self, value_learning_rate, num_value_steps, *args, **kwargs):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        Value_net.__init__(self, *args, **kwargs)

        self.value_learning_rate                = value_learning_rate
        self.num_value_steps                    = num_value_steps


        self.main_value_net_params              =  None
        self.value_net_params                   =  None


        self.obs_task_ids_var                   =  None
        self.value_net_var                      =  None
        self.obs_task_ids_var                   =  None
        self.value_net_var                      =  None



        self.build_graph()
        self.create_TD_training_method()
        self.create_MC_training_method()


    def build_graph(self):
        """
        Builds computational graph for value_net
        """
        with tf.variable_scope(self.name):
            # build the actual value network
            self.main_obs_task_ids_var, self.main_value_net_var      = create_mlp(name                   =  'main_network',
                                                                       output_dim             =  1,
                                                                       hidden_sizes           =  self.hidden_sizes,
                                                                       hidden_nonlinearity    =  self.hidden_nonlinearity,
                                                                       output_nonlinearity    =  self.output_nonlinearity,
                                                                       input_dim              =  (None, self.ob_dim + self.task_id_dim,)
                                                                       )

            self.target_obs_task_ids_var, self.target_value_net_var  = create_mlp(name                   =  'target_network',
                                                                       output_dim             =  1,
                                                                       hidden_sizes           =  self.hidden_sizes,
                                                                       hidden_nonlinearity    =  self.hidden_nonlinearity,
                                                                       output_nonlinearity    =  self.output_nonlinearity,
                                                                       input_dim              =  (None, self.ob_dim + self.task_id_dim,)
                                                                       )

            # save the policy's trainable variables in dicts
            current_scope                                 = tf.get_default_graph().get_name_scope()
            trainable_policy_vars                         = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)


            self.main_value_net_params                    = OrderedDict()
            self.target_value_net_params                  = OrderedDict()

            for var in trainable_policy_vars:
                if 'main_network' in remove_scope_from_name(var.name, current_scope):
                    self.main_value_net_params[remove_scope_from_name(var.name, current_scope)]   = var
                if 'target_network' in remove_scope_from_name(var.name, current_scope):
                    self.target_value_net_params[remove_scope_from_name(var.name, current_scope)]   = var
            #print(self.main_value_net_params)
            #print(self.target_value_net_params)



    def create_TD_training_method(self):

        self.rewards_phs                     = tf.placeholder(dtype=tf.float32, shape=[None,1], name='rewards')
        self.discounts_phs                   = tf.placeholder(dtype=tf.float32, shape=[None,1], name='discounts' )
        self.dones_phs                       = tf.placeholder(dtype=tf.float32, shape=[None,1], name='dones')
        self.next_state_values_phs           = tf.placeholder(dtype=tf.float32, shape=[None,1], name='next_q_values_1')
       
        with tf.variable_scope('value_loss_TD'):

            self.target_state_values         = tf.stop_gradient(self.rewards_phs + self.next_state_values_phs * self.discounts_phs * (1. - self.dones_phs))
            self.value_TD_objective          = tf.reduce_mean(0.5 * tf.square(self.main_value_net_var - tf.stop_gradient(self.target_state_values)))

            self.TD_optimizer                = tf.train.AdamOptimizer(learning_rate = self.value_learning_rate).minimize(self.value_TD_objective, var_list=self.main_value_net_params)

    def create_MC_training_method(self):

        self.disounted_rewards_phs           = tf.placeholder(dtype=tf.float32, shape=[None,1], name='disounted_rewards')
       
        with tf.variable_scope('value_loss_MC'):
            self.value_MC_objective          = tf.reduce_mean(0.5 * tf.square(self.main_value_net_var - tf.stop_gradient(self.disounted_rewards_phs)))
            self.MC_optimizer                = tf.train.AdamOptimizer(learning_rate = self.value_learning_rate).minimize(self.value_MC_objective, var_list=self.main_value_net_params)



    def optimize_baseline_value(self, value_samples_data, fit_style, log=True):
        """
        Performs TD or MC update

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """




        '''
        next_q_values_1               = np.expand_dims(np.concatenate([path["next_q_values_1"]   for path in off_critic_samples_data]), axis=-1)
        next_q_values_2               = np.expand_dims(np.concatenate([path["next_q_values_2"]   for path in off_critic_samples_data]) , axis=-1)
        '''

        if log: logger.log("Optimizing Baseline ValueNet")
        sess            = tf.get_default_session()

        if fit_style == "MC":
            observations                  = np.concatenate([path["observations"] for path in value_samples_data])            
            task_ids                      = np.concatenate([path["task_ids"] for path in value_samples_data])            
            observations_task_ids         = np.concatenate([observations, task_ids], axis=-1)
            discounted_rewards            = np.expand_dims(np.concatenate([path["returns"] for path in value_samples_data]), axis=-1)            

            for epoch in range(self.num_value_steps):

                _, self.value_loss             = sess.run([self.MC_optimizer, self.value_MC_objective], feed_dict= {self.disounted_rewards_phs     : discounted_rewards,
                                                                                                                    self.main_obs_task_ids_var     : observations_task_ids

                                                                                                           })
                print(self.value_loss)
        elif fit_style == "TD":
            observations                  = np.concatenate([path["observations"] for path in value_samples_data])            
            task_ids                      = np.concatenate([path["task_ids"]   for path in value_samples_data])
            observations_task_ids         = np.concatenate([observations, task_ids], axis=-1)
            rewards                       = np.expand_dims(np.concatenate([path["rewards"]      for path in value_samples_data]), axis=-1)
            discounts                     = np.expand_dims(np.concatenate([path["discounts"]   for path in value_samples_data]), axis=-1) 
            dones                         = np.expand_dims(np.concatenate([path["dones"]   for path in value_samples_data])   , axis=-1)   
            next_state_values             = np.expand_dims(np.concatenate([path["next_state_values"]   for path in value_samples_data]), axis=-1)
         
            for epoch in range(self.num_value_steps):

                _, self.value_loss             = sess.run([self.TD_optimizer, self.value_TD_objective], feed_dict= { self.rewards_phs               : rewards,
                                                                                                                     self.discounts_phs             : discounts,
                                                                                                                     self.dones_phs                 : dones,  
                                                                                                                     self.next_state_values_phs     : next_state_values,
                                                                                                                     self.main_obs_task_ids_var     : observations_task_ids
                                                                                                                   })



        if log: logger.log("Computing Baseline ValueNet statistics")

        if log:
            logger.log("*"*40)
            logger.log(self.value_loss)
            logger.log("*"*40)

            logger.logkv('VLoss_%s' % self.name, self.value_loss)
















    def get_state_value(self, observation, task_id):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observations                       = np.expand_dims(observation, axis=0)
        task_ids                           = np.expand_dims(task_id, axis=0)

        state_values, value_infos          = self.get_state_values(observations, task_ids)
        state_value,  value_info           = state_values[0], dict(state_value=value_infos['state_values'][0])
        return q_value, critic_info

    def get_state_values(self, observations, task_ids):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (ndarray) : array of observations - shape: (batch_size, obs_dim)

        Returns:
            (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
        """

        assert observations.ndim == 2 and observations.shape[1] == self.ob_dim
        assert task_ids.ndim     == 2 and task_ids.shape[1]     == self.task_id_dim

        observations_task_ids         = np.concatenate([observations, task_ids], axis=-1)

        sess                          = tf.get_default_session()
        state_values                  = sess.run([self.main_value_net_var],
                                                  feed_dict={self.main_obs_task_ids_var: observations_task_ids})

        return state_values, dict(state_values=state_values)



    def get_next_state_value(self, next_observation, next_task_id):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        next_observations                            = np.expand_dims(next_observation, axis=0)
        next_task_ids                                = np.expand_dims(next_task_id, axis=0)

        next_state_values, next_value_infos          = self.get_next_state_values(next_observations, next_task_ids)
        next_state_value, next_value_info            = next_state_values[0], dict(next_state_value=next_state_infos['next_state_values'][0])
        return next_state_value, next_value_info

    def get_next_state_values(self, next_observations, next_task_ids):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (ndarray) : array of observations - shape: (batch_size, obs_dim)

        Returns:
            (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
        """
        assert next_observations.ndim == 2 and next_observations.shape[1] == self.ob_dim
        assert next_task_ids.ndim     == 2 and next_task_ids.shape[1]     == self.task_id_dim

        next_observations_task_ids         = np.concatenate([next_observations, next_task_ids], axis=-1)

        sess                               = tf.get_default_session()
        target_q_values                    = sess.run([self.target_value_net_var],
                                                  feed_dict={self.target_obs_task_ids_var: next_observations_task_ids})

        return target_q_values , dict(next_state_values=target_q_values)



    def update_target_value_network(self, tau):
        update_ops     = []
        for var_name, var in self.main_value_net_params.items():
            target     =  self.target_value_net_params['target_network'+var_name.split('main_network')[1]]
            op         = tf.assign(target, tau* var.value() + (1-tau) * target.value())
            update_ops.append(op)

        sess                          = tf.get_default_session()
        sess.run(update_ops)



