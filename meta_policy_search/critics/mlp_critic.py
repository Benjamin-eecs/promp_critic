from meta_policy_search.utils import Serializable, logger
from meta_policy_search.utils.utils import remove_scope_from_name

from meta_policy_search.critics.networks.mlp import create_mlp
from meta_policy_search.critics.base import Critic



import tensorflow as tf
import numpy as np
from collections import OrderedDict


class MLPCritic(Critic):
    """
    multi-layer perceptron critic 
    Provides functions for executing and updating critic parameters

    Args:
        obs_dim (int)                       : dimensionality of the observation space 
        action_dim (int)                    : dimensionality of the action space 
        task_dim(int)

        name (str)                          : name of the policy used as tf variable scope
        hidden_sizes (tuple)                : tuple of integers specifying the hidden layer sizes of the MLP
        hidden_nonlinearity (tf.op)         : nonlinearity function of the hidden layers
        output_nonlinearity (tf.op or None) : nonlinearity function of the output layer


    """

    def __init__(self, *args, **kwargs):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        Critic.__init__(self, *args, **kwargs)


        self.main_critic_params                 =  None
        self.target_critic_params               =  None
        self.main_obs_acs_task_ids_var          =  None
        self.main_q_value_var                   =  None
        self.target_obs_acs_task_ids_var        =  None
        self.target_q_value_var                 =  None 

        self.build_graph()


    def build_graph(self):
        """
        Builds computational graph for policy
        """
        with tf.variable_scope(self.name):
            # build the actual policy network
            self.main_obs_acs_task_ids_var, self.main_q_value_var     = create_mlp(name                   =  'main_network',
                                                                                   output_dim             =  1,
                                                                                   hidden_sizes           =  self.hidden_sizes,
                                                                                   hidden_nonlinearity    =  self.hidden_nonlinearity,
                                                                                   output_nonlinearity    =  self.output_nonlinearity,
                                                                                   input_dim              =  (None, self.ob_dim + self.action_dim + self.task_id_dim,)
                                                                                   )
            self.target_obs_acs_task_ids_var, self.target_q_value_var = create_mlp(name                   =  'target_network',
                                                                                   output_dim             =  1,
                                                                                   hidden_sizes           =  self.hidden_sizes,
                                                                                   hidden_nonlinearity    =  self.hidden_nonlinearity,
                                                                                   output_nonlinearity    =  self.output_nonlinearity,
                                                                                   input_dim              =  (None, self.ob_dim + self.action_dim + self.task_id_dim,)
                                                                                   )

            # save the policy's trainable variables in dicts
            current_scope                               = tf.get_default_graph().get_name_scope()
            trainable_policy_vars                       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)

            self.main_critic_params                     = OrderedDict()
            self.target_critic_params                   = OrderedDict()

            for var in trainable_policy_vars:
                print(var)
                if 'main_network' in remove_scope_from_name(var.name, current_scope):
                    self.main_critic_params[remove_scope_from_name(var.name, current_scope)]   = var
                if 'target_network' in remove_scope_from_name(var.name, current_scope):           
                    self.target_critic_params[remove_scope_from_name(var.name, current_scope)] = var


    def get_q_value(self, observation, action, task_id):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observations                    = np.expand_dims(observation, axis=0)
        actions                         = np.expand_dims(action, axis=0)
        task_ids                        = np.expand_dims(task_id, axis=0)

        q_values, critic_infos          = self.get_q_values(observations, actions, task_ids)
        q_value, critic_info            = q_values[0], dict(q_value=critic_infos['q_value'][0])
        return q_value, critic_info

    def get_q_values(self, observations, actions, task_ids):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (ndarray) : array of observations - shape: (batch_size, obs_dim)

        Returns:
            (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
        """

        assert observations.ndim == 2 and observations.shape[1] == self.ob_dim
        assert actions.ndim      == 2 and actions.shape[1]      == self.action_dim
        assert task_ids.ndim     == 2 and task_ids.shape[1]     == self.task_id_dim

        observations_actions_task_ids = np.concatenate([observations, actions, task_ids], axis=-1)

        sess                          = tf.get_default_session()
        main_q_values                 = sess.run([self.main_q_value_var],
                                                  feed_dict={self.main_obs_acs_task_ids_var: observations_actions_task_ids})

        return main_q_values, dict(q_value=main_q_values)



    def get_next_q_value(self, next_observation, next_action, next_task_id):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observations                    = np.expand_dims(observation, axis=0)
        actions                         = np.expand_dims(action, axis=0)
        task_ids                        = np.expand_dims(task_id, axis=0)

        q_values, critic_infos          = self.get_q_values(observations, actions, task_ids)
        q_value, critic_info            = q_values[0], dict(q_value=critic_infos['q_value'][0])
        return q_value, critic_info

    def get_next_q_values(self, next_observations, next_actions, next_task_ids):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (ndarray) : array of observations - shape: (batch_size, obs_dim)

        Returns:
            (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
        """
        print(observations)
        assert observations.ndim == 2 and observations.shape[1] == self.ob_dim
        assert actions.ndim      == 2 and actions.shape[1]      == self.action_dim
        assert task_ids.ndim     == 2 and task_ids.shape[1]     == self.task_id_dim

        observations_actions_task_ids = np.concatenate([observations, actions, task_ids], axis=-1)

        sess                          = tf.get_default_session()
        main_q_values                 = sess.run([self.main_q_value_var],
                                                  feed_dict={self.main_obs_acs_task_ids_var: observations_actions_task_ids})

        return main_q_values, dict(q_value=main_q_values)







    def update_target_critic_network(self, tau):
        update_ops = []
        for var_name, var in self.main_critic_params.items():
            target  =  self.target_critic_params['target_network'+var_name.split('main_network')[1]]
            op      = tf.assign(target, tau* var.value() + (1-tau) * target.value())
            update_ops.append(op)

        sess                          = tf.get_default_session()
        sess.run(update_ops)



