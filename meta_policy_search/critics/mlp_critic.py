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

    def __init__(self, critic_learning_rate, num_critic_steps, *args, **kwargs):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        Critic.__init__(self, *args, **kwargs)


        self.critic_learning_rate                        =  critic_learning_rate
        self.num_critic_steps                            =  num_critic_steps

        self.main_critic_params                          =  None
        self.target_critic_params                        =  None

        self.main_obs_acs_task_ids_var                   =  None
        self.main_q_value_var                            =  None
        self.target_obs_acs_task_ids_var                 =  None
        self.target_q_value_var                          =  None 
        
        self.build_graph()
        self.create_training_method()

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
                #print(var)
                if 'main_network' in remove_scope_from_name(var.name, current_scope):
                    self.main_critic_params[remove_scope_from_name(var.name, current_scope)]   = var
                if 'target_network' in remove_scope_from_name(var.name, current_scope):           
                    self.target_critic_params[remove_scope_from_name(var.name, current_scope)] = var

    def create_training_method(self):

        #self.test_1                     = tf.placeholder(dtype=tf.float32, shape=[None,1], name='test_1')
        #self.test_2                     = tf.placeholder(dtype=tf.float32, shape=[None], name='test_2')

        #self.test_3                     = self.test_1 *self.test_2



        self.rewards_phs                = tf.placeholder(dtype=tf.float32, shape=[None,1], name='rewards')
        self.discounts_phs              = tf.placeholder(dtype=tf.float32, shape=[None,1], name='discounts' )
        self.dones_phs                  = tf.placeholder(dtype=tf.float32, shape=[None,1], name='dones')
        self.next_q_values_1_phs        = tf.placeholder(dtype=tf.float32, shape=[None,1], name='next_q_values_1')
        self.next_q_values_2_phs        = tf.placeholder(dtype=tf.float32, shape=[None,1], name='next_q_values_2')
       
        with tf.variable_scope('critic_loss'):
            self.next_q_values_min           = tf.minimum(self.next_q_values_1_phs, self.next_q_values_1_phs)
            self.target_q_values             = tf.stop_gradient(self.rewards_phs + self.next_q_values_min * self.discounts_phs * (1. - self.dones_phs))

            #self.test                        = self.main_q_value_var - tf.stop_gradient(self.target_q_values)
            self.critic_objective            = tf.reduce_mean(0.5 * tf.square(self.main_q_value_var - tf.stop_gradient(self.target_q_values)))

            self.optimizer                   = tf.train.AdamOptimizer(learning_rate = self.critic_learning_rate).minimize(self.critic_objective, var_list=self.main_critic_params)


    def optimize_critic(self, off_critic_samples_data, log=True):
        """
        Performs TD update

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        observations                  = np.concatenate([path["observations"] for path in off_critic_samples_data])
        actions                       = np.concatenate([path["actions"]      for path in off_critic_samples_data])
        task_ids                      = np.concatenate([path["task_ids"]   for path in off_critic_samples_data])

        observations_actions_task_ids = np.concatenate([observations, actions, task_ids], axis=-1)

        rewards                       = np.expand_dims(np.concatenate([path["rewards"]      for path in off_critic_samples_data]), axis=-1)
        discounts                     = np.expand_dims(np.concatenate([path["discounts"]   for path in off_critic_samples_data]), axis=-1) 
        dones                         = np.expand_dims(np.concatenate([path["dones"]   for path in off_critic_samples_data])   , axis=-1)

        next_q_values_1               = np.expand_dims(np.concatenate([path["next_q_values_1"]   for path in off_critic_samples_data]), axis=-1)
        next_q_values_2               = np.expand_dims(np.concatenate([path["next_q_values_2"]   for path in off_critic_samples_data]) , axis=-1)

        # add kl_coeffs / clip_eps to meta_op_input_dict
        if log: logger.log("Optimizing Critics")
        sess            = tf.get_default_session()

        '''
        print(sess.run(self.test, feed_dict= { self.rewards_phs               : rewards,
                                                 self.discounts_phs             : discounts,
                                                 self.dones_phs                 : dones,  
                                                 self.next_q_values_1_phs       : next_q_values_1,
                                                 self.next_q_values_2_phs       : next_q_values_2,
                                                 self.main_obs_acs_task_ids_var : observations_actions_task_ids
                                                                                                       }))
        '''
        #time.sleep(100)
        for epoch in range(self.num_critic_steps):

            _, self.critic_loss             = sess.run([self.optimizer, self.critic_objective], feed_dict= { self.rewards_phs               : rewards,
                                                                                                             self.discounts_phs             : discounts,
                                                                                                             self.dones_phs                 : dones,  
                                                                                                             self.next_q_values_1_phs       : next_q_values_1,
                                                                                                             self.next_q_values_2_phs       : next_q_values_2,
                                                                                                             self.main_obs_acs_task_ids_var : observations_actions_task_ids
                                                                                                       })

        if log: logger.log("Computing Critics statistics")

        if log:
            logger.log("*"*40)
            logger.log(self.critic_loss)
            logger.log("*"*40)

            logger.logkv('QLoss_%s' % self.name, self.critic_loss)



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
        q_value, critic_info            = q_values[0], dict(q_value=critic_infos['q_values'][0])
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

        return main_q_values, dict(q_values=main_q_values)



    def get_next_q_value(self, next_observation, next_action, next_task_id):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        next_observations                      = np.expand_dims(next_observation, axis=0)
        next_actions                           = np.expand_dims(next_action, axis=0)
        next_task_ids                          = np.expand_dims(next_task_id, axis=0)

        next_q_values, next_critic_infos       = self.get_next_q_values(next_observations, next_actions, next_task_ids)
        next_q_value,  next_critic_info        = next_q_values[0], dict(next_q_value=critic_infos['next_q_values'][0])
        return next_q_value, next_critic_info

    def get_next_q_values(self, next_observations, next_actions, next_task_ids):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (ndarray) : array of observations - shape: (batch_size, obs_dim)

        Returns:
            (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
        """
        assert next_observations.ndim == 2 and next_observations.shape[1] == self.ob_dim
        assert next_actions.ndim      == 2 and next_actions.shape[1]      == self.action_dim
        assert next_task_ids.ndim     == 2 and next_task_ids.shape[1]     == self.task_id_dim

        next_observations_actions_task_ids = np.concatenate([next_observations, next_actions, next_task_ids], axis=-1)

        sess                               = tf.get_default_session()
        target_q_values                    = sess.run([self.target_q_value_var],
                                                       feed_dict={self.target_obs_acs_task_ids_var: next_observations_actions_task_ids})

        return target_q_values, dict(next_q_values=target_q_values)







    def update_target_critic_network(self, tau):
        update_ops = []
        for var_name, var in self.main_critic_params.items():
            target  =  self.target_critic_params['target_network'+var_name.split('main_network')[1]]
            op      = tf.assign(target, tau* var.value() + (1-tau) * target.value())
            update_ops.append(op)

        sess                          = tf.get_default_session()
        sess.run(update_ops)



