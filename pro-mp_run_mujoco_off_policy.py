from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.critics.mlp_critic import MLPCritic

from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder





from meta_policy_search.meta_algos.pro_mp_off import ProMP_off
from meta_policy_search.meta_trainer_off import Trainer_off
from meta_policy_search.samplers.meta_sampler_off import MetaSampler_off
from meta_policy_search.samplers.meta_sample_processor_off import MetaSampleProcessor_off


import numpy as np
import tensorflow as tf
import os
import json
import argparse
import time

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])



def main(config):
    set_seed(config['seed'])


    baseline                      =  globals()[config['baseline']]() #instantiate baseline

    env                           =  globals()[config['env']](config['seed']) # instantiate env

    env                           =  normalize(env) # apply normalize wrapper to env

    policy  = MetaGaussianMLPPolicy(
            name                   = "meta-policy",
            obs_dim                = np.prod(env.observation_space.shape),
            action_dim             = np.prod(env.action_space.shape),
            meta_batch_size        = config['meta_batch_size'],
            hidden_sizes           = config['hidden_sizes'],
        )

    critic_1 = MLPCritic(
            ob_dim                 = np.prod(env.observation_space.shape),
            action_dim             = np.prod(env.action_space.shape),
            task_id_dim            = config['num_tasks'],
            name                   = "first_critic",
            hidden_sizes           = config['hidden_sizes'],
            )

    critic_2 = MLPCritic(
            ob_dim                 = np.prod(env.observation_space.shape),
            action_dim             = np.prod(env.action_space.shape),
            task_id_dim            = config['num_tasks'],
            name                   = "second_critic",
            hidden_sizes           = config['hidden_sizes'],
            )

    sampler = MetaSampler_off(
        env                        = env,
        policy                     = policy,
        rollouts_per_meta_task     = config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size            = config['meta_batch_size'],
        max_path_length            = config['max_path_length'],
        parallel                   = config['parallel'],
        buffer_length              = config['buffer_length'],
        discount                   = config['discount'],
        num_tasks                  = config['num_tasks']
    )

    # use critic to process sample
    sample_processor = MetaSampleProcessor_off(
        critic_1                   = critic_1,
        critic_2                   = critic_2,
        baseline                   = baseline,
        discount                   = config['discount'],
        gae_lambda                 = config['gae_lambda'],
        normalize_adv              = config['normalize_adv'],
    )
    
    # update critic
    algo = ProMP_off(
        policy                     =policy,
        critic_1                   =critic_1,
        critic_2                   =critic_2,

        inner_lr                   =config['inner_lr'],
        meta_batch_size            =config['meta_batch_size'],
        num_inner_grad_steps       =config['num_inner_grad_steps'],

        learning_rate              =config['learning_rate'],


        critic_learning_rate       =config['critic_learning_rate'],
        num_critic_steps           =config['num_critic_steps'],

        num_ppo_steps              =config['num_promp_steps'],
        clip_eps                   =config['clip_eps'],
        target_inner_step          =config['target_inner_step'],
        init_inner_kl_penalty      =config['init_inner_kl_penalty'],
        adaptive_inner_kl_penalty  =config['adaptive_inner_kl_penalty'],
        off_clip_eps_upper         =config['off_clip_eps_upper'],
        off_clip_eps_lower         =config['off_clip_eps_lower'],
        clip_style                 =config['clip_style']                 #0:TRPO 1:off policy pg
    )

    #update target
    trainer = Trainer_off(
        algo                       = algo,
        env                        = env,
        sampler                    = sampler,
        sample_processor           = sample_processor,
        policy                     = policy,
        critic_1                   = critic_1,
        critic_2                   = critic_2,
        n_itr                      = config['n_itr'],
        seeds                      = [config['seed']] * config['rollouts_per_meta_task'] * config['meta_batch_size'],
        num_inner_grad_steps       = config['num_inner_grad_steps'],
        sample_batch_size          = config['sample_batch_size'],
        
    )

    trainer.train()

if __name__=="__main__":
    idx = int(time.time())

    parser = argparse.ArgumentParser(description='ProMP: Proximal Meta-Policy Search')



    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--sampler', type=int, default=1, help='parameter setting')

    parser.add_argument('--lr',   type=int, default=1e-4, help='learning rate')
    parser.add_argument('--env',  type=str, default='HalfCheetahRandDirecEnv', help='environment')



    parser.add_argument('--meta_batch_size',   type=int, default=2, help='meta batch size')
    parser.add_argument('--rollout_per_task',  type=int, default=3, help='rollout per task')






    parser.add_argument('--config_file', type=str, default='', help='json file with run specifications')
    #parser.add_argument('--dump_path', type=str, default=meta_policy_search_path + '/data/pro-mp/run_%d' % idx)
    parser.add_argument('--dump_path', type=str, default= './data/pro-mp-off/run_%d' % idx)

    args = parser.parse_args()





    args.dump_path = meta_policy_search_path + '/data/pro-mp/test_params_%d_seed_%d' % (args.sampler, args.seed)

    if args.config_file: # load configuration from json file
        with open(args.config_file, 'r') as f:
            config = json.load(f)



    else: # use default config

        config = {
            'seed'    : args.seed,
            'sampler' : args.sampler,



            #off_policy config
            'num_tasks'                           : 2,

            'buffer_length'                       : 200, # meta_batch_size * rollout_per_task * max_path_length *constant
            'sample_batch_size'                   : 1,    # for each meta task
            'off_clip_eps_upper'                  : 0.8,
            'off_clip_eps_lower'                  : 1,
            'clip_style'                          : 0,

            #critic optimizer settings
            'critic_learning_rate'                : 1e-4,
            'num_critic_steps'                    : 1,


            'baseline'                            : 'LinearFeatureBaseline',
            'env'                                 : args.env,



            # sampler config
            'rollouts_per_meta_task'              : args.rollout_per_task,
            'max_path_length'                     : 100,
            'parallel'                            : True,

            # sample processor config
            'discount'                            : 0.99,
            'gae_lambda'                          : 1,
            'normalize_adv'                       : True,

            # policy config
            'hidden_sizes'                        : (64, 64),
            'learn_std'                           : True, # whether to learn the standard deviation of the gaussian policy

            # ProMP config
            'inner_lr'                            : 0.1, # adaptation step size
            'learning_rate'                       : args.lr, # meta-policy gradient step size
            'num_promp_steps'                     : 5, # number of ProMp steps without re-sampling
            'clip_eps'                            : 0.3, # clipping range
            'target_inner_step'                   : 0.01,
            'init_inner_kl_penalty'               : 5e-4,
            'adaptive_inner_kl_penalty'           : False, # whether to use an adaptive or fixed KL-penalty coefficient
            'n_itr'                               : 1001, # number of overall training iterations
            'meta_batch_size'                     : args.meta_batch_size, # number of sampled meta-tasks per iterations
            'num_inner_grad_steps'                : 1, # number of inner / adaptation gradient steps

        }

    # configure logger
    logger.configure(dir=args.dump_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')

    # dump run configuration before starting training
    json.dump(config, open(args.dump_path + '/params.json', 'w'), cls=ClassEncoder)

    # start the actual algorithm
    main(config)
