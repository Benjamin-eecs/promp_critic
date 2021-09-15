import tensorflow as tf
import numpy as np
import time
from meta_policy_search.utils import logger
import wandb

class Trainer_off(object):
    """
    Performs steps of meta-policy search.

     Pseudocode::

            for iter in n_iter:
                sample tasks
                for task in tasks:
                    for adapt_step in num_inner_grad_steps
                        sample trajectories with policy
                        perform update/adaptation step
                    sample trajectories with post-update policy
                perform meta-policy gradient step(s)

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            sampler,
            sample_processor,
            policy,
            critic_1,
            critic_2,
            n_itr,
            seeds,

            start_itr=0,
            num_inner_grad_steps=1,
            sample_batch_size=32,
            sess=None,
            ):
        self.algo                 = algo
        self.env                  = env
        self.sampler              = sampler
        self.sample_processor     = sample_processor
        self.baseline             = sample_processor.baseline
        self.policy               = policy
        self.critic_1             = critic_1
        self.critic_2             = critic_2
        self.n_itr                = n_itr
        self.start_itr            = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.sample_batch_size    = sample_batch_size
        self.seeds                = seeds
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        """
        Trains policy on env using algo

        Pseudocode::
        
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        self.sampler.set_seeds(self.seeds)

        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()


            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                #self.sampler.update_tasks()
                tasks, tasks_id = self.sampler.update_tasks_with_id()
                
                self.policy.switch_to_pre_update()  # Switch to pre-update policy

                all_samples_data, all_paths = [], []
                list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
                start_total_inner_time = time.time()
                for step in range(self.num_inner_grad_steps+1):
                    logger.log('** Step ' + str(step) + ' **')

                    """ -------------------- Sampling --------------------------"""

                    logger.log("Obtaining samples...")
                    time_env_sampling_start = time.time()
                    if step == 0:
                        paths_step_0 = self.sampler.obtain_samples(tasks_id, step_id=step, log=True, log_prefix='Step_%d-' % step)
                    elif step ==1:
                        paths_step_1 = self.sampler.obtain_samples(tasks_id, step_id=step, log=True, log_prefix='Step_%d-' % step)

                    list_sampling_time.append(time.time() - time_env_sampling_start)
                    if step   == 0:
                        all_paths.append(paths_step_0)
                    elif step == 1:
                        all_paths.append(paths_step_1)

                    """ ----------------- Processing Samples ---------------------"""

                    logger.log("Processing samples...")
                    time_proc_samples_start = time.time()
                    if step   == 0:
                        off_sample_path                          = self.sampler.buffer.sample(tasks_id, self.sample_batch_size)
                        samples_data_step_0, off_sample_data     = self.sample_processor.process_samples(off_sample = off_sample_path, paths_meta_batch=paths_step_0, log='all', log_prefix='Step_%d-' % step)
                    elif step == 1:
                        samples_data_step_1, _                   = self.sample_processor.process_samples(off_sample = None,            paths_meta_batch=paths_step_1, log='all', log_prefix='Step_%d-' % step)
                    if step   == 0:
                        all_samples_data.append(samples_data_step_0)
                    elif step == 1:
                        all_samples_data.append(samples_data_step_1)

                    list_proc_samples_time.append(time.time() - time_proc_samples_start)
                    
                    if step   == 0:
                        self.log_diagnostics(sum(list(paths_step_0.values()), []), prefix='Step_%d-' % step)
                    elif step == 1:
                        self.log_diagnostics(sum(list(paths_step_1.values()), []), prefix='Step_%d-' % step)

                    """ ------------------- Inner Policy Update --------------------"""

                    time_inner_step_start = time.time()
                    if step < self.num_inner_grad_steps:
                        logger.log("Computing inner policy updates...")
                        
                        self.algo._adapt_off_critic(samples_data_step_0, off_sample_data)

                    list_inner_step_time.append(time.time() - time_inner_step_start)

                all_samples_data.append(off_sample_data)

                test_time_inner_step_start = time.time()

                logger.log('** test **')
                self.policy.switch_to_pre_update()

                """ ------------------- Inner Policy Update --------------------"""
                logger.log("test-Computing inner policy updates...")
                self.algo._adapt(samples_data_step_0)

                """ -------------------- Sampling --------------------------"""
                logger.log("test-Obtaining samples...")
                test_paths_step_1             = self.sampler.obtain_samples(tasks_id, step_id=1, log=True, log_prefix='test-Step_%d-' % 1)
                
                """ ----------------- Processing Samples ---------------------"""
                logger.log("test-Processing samples...")
                test_samples_data_step_1      = self.sample_processor.process_samples(off_sample = None, paths_meta_batch=test_paths_step_1, log='all', log_prefix='test-Step_%d-' % 1)


                list_inner_step_time.append(time.time() - test_time_inner_step_start)

                total_inner_time = time.time() - start_total_inner_time

                time_maml_opt_start = time.time()
                """ ------------------ Outer Policy Update ---------------------"""

                logger.log("Optimizing policy...")
                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_outer_step_start = time.time()
                self.algo.optimize_policy(all_samples_data)
                self.algo.optimize_critic()

                self.critic_1.update_target_critic_network(self.tau)
                self.critic_1.update_target_critic_network(self.tau)



                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time-OuterStep', time.time() - time_outer_step_start)
                logger.logkv('Time-TotalInner', total_inner_time)
                logger.logkv('Time-InnerStep', np.sum(list_inner_step_time))
                logger.logkv('Time-SampleProc', np.sum(list_proc_samples_time))
                logger.logkv('Time-Sampling', np.sum(list_sampling_time))

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)
                logger.logkv('Time-MAMLSteps', time.time() - time_maml_opt_start)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()

        logger.log("Training finished")
        res      = np.array(self.sample_processor.Step_1_AverageReturn)
        test_res = np.array(self.sample_processor.test_Step_1_AverageReturn)

        res_dict = dict([('Step_1-AverageReturn',res),
                         ('test-Step_1-AverageReturn',test_res)])
        self.sess.close()
        tf.reset_default_graph()
         
        return res_dict      

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
