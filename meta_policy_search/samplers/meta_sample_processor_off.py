from meta_policy_search.samplers.base import SampleProcessor
from meta_policy_search.utils import utils
import numpy as np

class MetaSampleProcessor_off(SampleProcessor):

    def process_samples(self, off_sample, paths_meta_batch, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths_meta_batch (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """

        #assert isinstance(off_sample, dict), 'paths must be a dict'
        assert isinstance(paths_meta_batch, dict), 'paths must be a dict'
        assert self.baseline, 'baseline must be specified'

        samples_data_meta_batch                 = []
        off_policy_samples_data_meta_batch      = []

        all_paths = []

        for meta_task, paths in paths_meta_batch.items():

            # fits baseline, compute advantages and stack path data
            samples_data,          paths       = self._compute_samples_data(paths)
            if isinstance(off_sample, dict):
                off_samples_data,  off_paths   = self._compute_samples_data_off_critic(off_sample.get(meta_task))
                off_policy_samples_data_meta_batch.append(off_samples_data)

            samples_data_meta_batch.append(samples_data)
            
            all_paths.extend(paths)

        # 7) compute normalized trajectory-batch rewards (for E-MAML)
        overall_avg_reward         = np.mean(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))
        overall_avg_reward_std     = np.std(np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))
        
        if isinstance(off_sample, dict):
            off_overall_avg_reward     = np.mean(np.concatenate([off_samples_data['rewards'] for off_samples_data in off_policy_samples_data_meta_batch]))
            off_overall_avg_reward_std = np.std(np.concatenate([off_samples_data['rewards'] for off_samples_data in off_policy_samples_data_meta_batch]))
            for off_samples_data in off_policy_samples_data_meta_batch:
                off_samples_data['adj_avg_rewards'] = (off_samples_data['rewards'] - off_overall_avg_reward) / (off_overall_avg_reward_std + 1e-8)
        for samples_data in samples_data_meta_batch:
            samples_data['adj_avg_rewards'] = (samples_data['rewards'] - overall_avg_reward) / (overall_avg_reward_std + 1e-8)


        # 8) log statistics if desired
        undiscounted_returns_mean = self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)

        if log_prefix == 'test-Step_1-':
            self.test_Step_1_AverageReturn.append(undiscounted_returns_mean)
        elif log_prefix == 'Step_1-':
            self.Step_1_AverageReturn.append(undiscounted_returns_mean)
        return samples_data_meta_batch, off_policy_samples_data_meta_batch




