import pandas as pd
from pandas.core.algorithms import isin
import numpy as np
import torch
from copy import deepcopy
import logging

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src import ROOT_PATH
from src.data.dataset_collection import RealDatasetCollection
from src.data.mimic_iii.load_data import load_mimic3_data_processed
from CBD.MBs.pc_simple import pc_simple
from src.data.simulate_data.simulated_data import AutoregressiveSimulation
logger = logging.getLogger(__name__)


class SimulateDataset(Dataset):
    """
    Pytorch-style simulate dataset
    """
    def __init__(self, treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params, subset_name, coso_vitals, COSO_index):
        """
        Args:
            treatments, outcomes, vitals, coso_vitals, outcomes_unscaled: NumPy arrays with shape (num_patients, num_timesteps, num_features)
            static_features: NumPy array with shape (num_patients, num_static_features)
            scaling_params: Dictionary of standard normalization scaling parameters
            subset_name: 'train', 'val', or 'test'
        """
        assert treatments.shape[0] == outcomes.shape[0] == vitals.shape[0] == outcomes_unscaled.shape[0] == coso_vitals.shape[0], "All input arrays must have the same number of patients."
        self.subset_name = subset_name

        # Calculate active_entries based on treatments not being NaN
        # Assuming treatments is a good proxy for activity across all features
        active_entries = np.isnan(treatments).any(axis=2) == False
        active_entries = active_entries.astype(float)[:, :, np.newaxis]  # Ensure it is 3D and float type

        # Handling specific feature extraction and removal
        COSO = coso_vitals[:, :, COSO_index, np.newaxis]
        coso_vitals = np.delete(coso_vitals, COSO_index, axis=2)

        # Convert NaNs to zero where necessary
        treatments = np.nan_to_num(treatments, nan=0.0)
        outcomes = np.nan_to_num(outcomes, nan=0.0)
        vitals = np.nan_to_num(vitals, nan=0.0)
        outcomes_unscaled = np.nan_to_num(outcomes_unscaled, nan=0.0)
        coso_vitals = np.nan_to_num(coso_vitals, nan=0.0)

        # Reorganizing the data into the needed structure
        self.data = {
            'sequence_lengths': np.sum(active_entries.squeeze(), axis=1),
            'prev_treatments': treatments[:, :-1, :],
            'vitals': vitals[:, 1:, :],
            'next_vitals': vitals[:, 2:, :],
            'current_treatments': treatments[:, 1:, :],
            'static_features': static_features,
            'active_entries': active_entries[:, 1:, :],  # shifted to align with the rest of the time series data
            'outputs': outcomes[:, 1:, :],
            'unscaled_outputs': outcomes_unscaled[:, 1:, :],
            'prev_outputs': outcomes[:, :-1, :],
            #'coso_vitals': coso_vitals[:, 1:, :],
            'Adjustment_vitals': coso_vitals[:, 1:, :],
            'COSO': COSO[:, 1:, :]
        }
        self.scaling_params = scaling_params
        self.processed = True
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.exploded = False

        data_shapes = {k: v.shape for k, v in self.data.items()}
        logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

        self.norm_const = 1.0

    def __getitem__(self, index) -> dict:
        result = {k: v[index] for k, v in self.data.items()}
        if hasattr(self, 'encoder_r'):
            if 'original_index' in self.data:
                result.update({'encoder_r': self.encoder_r[int(result['original_index'])]})
            else:
                result.update({'encoder_r': self.encoder_r[index]})
        return result

    def __len__(self):
        return len(self.data['active_entries'])

    def explode_trajectories(self, projection_horizon):
        """
        Convert test dataset to a dataset with rolling origin
        Args:
            projection_horizon: projection horizon
        """
        assert self.processed

        logger.info(f'Exploding {self.subset_name} dataset before testing (multiple sequences)')

        outputs = self.data['outputs']
        prev_outputs = self.data['prev_outputs']
        sequence_lengths = self.data['sequence_lengths']
        vitals = self.data['vitals']
        next_vitals = self.data['next_vitals']
        active_entries = self.data['active_entries']
        current_treatments = self.data['current_treatments']
        previous_treatments = self.data['prev_treatments']
        static_features = self.data['static_features']
        if 'stabilized_weights' in self.data:
            stabilized_weights = self.data['stabilized_weights']

        num_patients, max_seq_length, num_features = outputs.shape
        num_seq2seq_rows = num_patients * max_seq_length

        seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, max_seq_length, previous_treatments.shape[-1]))
        seq2seq_current_treatments = np.zeros((num_seq2seq_rows, max_seq_length, current_treatments.shape[-1]))
        seq2seq_static_features = np.zeros((num_seq2seq_rows, static_features.shape[-1]))
        seq2seq_outputs = np.zeros((num_seq2seq_rows, max_seq_length, outputs.shape[-1]))
        seq2seq_prev_outputs = np.zeros((num_seq2seq_rows, max_seq_length, prev_outputs.shape[-1]))
        seq2seq_vitals = np.zeros((num_seq2seq_rows, max_seq_length, vitals.shape[-1]))
        seq2seq_next_vitals = np.zeros((num_seq2seq_rows, max_seq_length - 1, next_vitals.shape[-1]))
        seq2seq_active_entries = np.zeros((num_seq2seq_rows, max_seq_length, active_entries.shape[-1]))
        seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
        if 'stabilized_weights' in self.data:
            seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, max_seq_length))

        total_seq2seq_rows = 0  # we use this to shorten any trajectories later

        for i in range(num_patients):
            sequence_length = int(sequence_lengths[i])
            for t in range(projection_horizon, sequence_length):  # shift outputs back by 1
                seq2seq_active_entries[total_seq2seq_rows, :(t + 1), :] = active_entries[i, :(t + 1), :]
                if 'stabilized_weights' in self.data:
                    seq2seq_stabilized_weights[total_seq2seq_rows, :(t + 1)] = stabilized_weights[i, :(t + 1)]
                seq2seq_previous_treatments[total_seq2seq_rows, :(t + 1), :] = previous_treatments[i, :(t + 1), :]
                seq2seq_current_treatments[total_seq2seq_rows, :(t + 1), :] = current_treatments[i, :(t + 1), :]
                seq2seq_outputs[total_seq2seq_rows, :(t + 1), :] = outputs[i, :(t + 1), :]
                seq2seq_prev_outputs[total_seq2seq_rows, :(t + 1), :] = prev_outputs[i, :(t + 1), :]
                seq2seq_vitals[total_seq2seq_rows, :(t + 1), :] = vitals[i, :(t + 1), :]
                seq2seq_next_vitals[total_seq2seq_rows, :min(t + 1, sequence_length - 1), :] = \
                    next_vitals[i, :min(t + 1, sequence_length - 1), :]
                seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
                seq2seq_static_features[total_seq2seq_rows] = static_features[i]

                total_seq2seq_rows += 1

        # Filter everything shorter
        seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
        seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
        seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
        seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
        seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
        seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
        seq2seq_next_vitals = seq2seq_next_vitals[:total_seq2seq_rows, :, :]
        seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
        seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

        if 'stabilized_weights' in self.data:
            seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

        new_data = {
            'prev_treatments': seq2seq_previous_treatments,
            'current_treatments': seq2seq_current_treatments,
            'static_features': seq2seq_static_features,
            'prev_outputs': seq2seq_prev_outputs,
            'outputs': seq2seq_outputs,
            'vitals': seq2seq_vitals,
            'next_vitals': seq2seq_next_vitals,
            'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
            'sequence_lengths': seq2seq_sequence_lengths,
            'active_entries': seq2seq_active_entries,
        }
        if 'stabilized_weights' in self.data:
            new_data['stabilized_weights'] = seq2seq_stabilized_weights

        self.data = new_data

        data_shapes = {k: v.shape for k, v in self.data.items()}
        logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

    def process_sequential(self, encoder_r, projection_horizon, encoder_outputs=None, save_encoder_r=False):
        """
        Pre-process dataset for multiple-step-ahead prediction: explodes dataset to a larger one with rolling origin
        Args:
            encoder_r: Representations of encoder
            projection_horizon: Projection horizon
            encoder_outputs: One-step-ahead predcitions of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(f'Processing {self.subset_name} dataset before training (multiple sequences)')

            outputs = self.data['outputs']
            prev_outputs = self.data['prev_outputs']
            sequence_lengths = self.data['sequence_lengths']
            active_entries = self.data['active_entries']
            current_treatments = self.data['current_treatments']
            previous_treatments = self.data['prev_treatments']
            static_features = self.data['static_features']
            stabilized_weights = self.data['stabilized_weights'] if 'stabilized_weights' in self.data else None

            num_patients, max_seq_length, num_features = outputs.shape

            num_seq2seq_rows = num_patients * max_seq_length

            seq2seq_state_inits = np.zeros((num_seq2seq_rows, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((num_seq2seq_rows, max_seq_length))
            seq2seq_original_index = np.zeros((num_seq2seq_rows, ))
            seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, projection_horizon, previous_treatments.shape[-1]))
            seq2seq_current_treatments = np.zeros((num_seq2seq_rows, projection_horizon, current_treatments.shape[-1]))
            seq2seq_static_features = np.zeros((num_seq2seq_rows, static_features.shape[-1]))
            seq2seq_outputs = np.zeros((num_seq2seq_rows, projection_horizon, outputs.shape[-1]))
            seq2seq_prev_outputs = np.zeros((num_seq2seq_rows, projection_horizon, prev_outputs.shape[-1]))
            seq2seq_active_entries = np.zeros((num_seq2seq_rows, projection_horizon, active_entries.shape[-1]))
            seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
            seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, projection_horizon + 1)) \
                if stabilized_weights is not None else None

            total_seq2seq_rows = 0  # we use this to shorten any trajectories later

            for i in range(num_patients):

                sequence_length = int(sequence_lengths[i])

                for t in range(1, sequence_length - projection_horizon):  # shift outputs back by 1
                    seq2seq_state_inits[total_seq2seq_rows, :] = encoder_r[i, t - 1, :]  # previous state output
                    seq2seq_original_index[total_seq2seq_rows] = i
                    seq2seq_active_encoder_r[total_seq2seq_rows, :t] = 1.0

                    max_projection = min(projection_horizon, sequence_length - t)

                    seq2seq_active_entries[total_seq2seq_rows, :max_projection, :] = active_entries[i, t:t + max_projection, :]
                    seq2seq_previous_treatments[total_seq2seq_rows, :max_projection, :] = \
                        previous_treatments[i, t:t + max_projection, :]
                    seq2seq_current_treatments[total_seq2seq_rows, :max_projection, :] = \
                        current_treatments[i, t:t + max_projection, :]
                    seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[i, t:t + max_projection, :]
                    seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
                    seq2seq_static_features[total_seq2seq_rows] = static_features[i]
                    if encoder_outputs is not None:  # For auto-regressive evaluation
                        seq2seq_prev_outputs[total_seq2seq_rows, :max_projection, :] = \
                            encoder_outputs[i, t - 1:t + max_projection - 1, :]
                    else:  # train / val of decoder
                        seq2seq_prev_outputs[total_seq2seq_rows, :max_projection, :] = prev_outputs[i, t:t + max_projection, :]

                    if seq2seq_stabilized_weights is not None:  # Also including SW of one-step-ahead prediction
                        seq2seq_stabilized_weights[total_seq2seq_rows, :] = stabilized_weights[i, t - 1:t + max_projection]

                    total_seq2seq_rows += 1

            # Filter everything shorter
            seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
            seq2seq_original_index = seq2seq_original_index[:total_seq2seq_rows]
            seq2seq_active_encoder_r = seq2seq_active_encoder_r[:total_seq2seq_rows, :]
            seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
            seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
            seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
            seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
            seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
            seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
            seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

            if seq2seq_stabilized_weights is not None:
                seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

            # Package outputs
            seq2seq_data = {
                'init_state': seq2seq_state_inits,
                'original_index': seq2seq_original_index,
                'active_encoder_r': seq2seq_active_encoder_r,
                'prev_treatments': seq2seq_previous_treatments,
                'current_treatments': seq2seq_current_treatments,
                'static_features': seq2seq_static_features,
                'prev_outputs': seq2seq_prev_outputs,
                'outputs': seq2seq_outputs,
                'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
                'sequence_lengths': seq2seq_sequence_lengths,
                'active_entries': seq2seq_active_entries,
            }
            if seq2seq_stabilized_weights is not None:
                seq2seq_data['stabilized_weights'] = seq2seq_stabilized_weights

            self.data_original = deepcopy(self.data)
            self.data_processed_seq = deepcopy(seq2seq_data)  # For auto-regressive evaluation (self.data will be changed)
            self.data = seq2seq_data

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r:
                self.encoder_r = encoder_r[:, :max_seq_length, :]

            self.processed_sequential = True
            self.exploded = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (multiple sequences)')

        return self.data

    def process_sequential_test(self, projection_horizon, encoder_r=None, save_encoder_r=False):
        """
        Pre-process test dataset for multiple-step-ahead prediction: takes the last n-steps according to the projection horizon
        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(f'Processing {self.subset_name} dataset before testing (multiple sequences)')

            outputs = self.data['outputs']
            prev_outputs = self.data['prev_outputs']
            sequence_lengths = self.data['sequence_lengths']
            current_treatments = self.data['current_treatments']
            previous_treatments = self.data['prev_treatments']
            # vitals = self.data['vitals']

            num_patient_points, max_seq_length, num_features = outputs.shape

            if encoder_r is not None:
                seq2seq_state_inits = np.zeros((num_patient_points, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((num_patient_points, max_seq_length - projection_horizon))
            seq2seq_previous_treatments = np.zeros((num_patient_points, projection_horizon, previous_treatments.shape[-1]))
            seq2seq_current_treatments = np.zeros((num_patient_points, projection_horizon, current_treatments.shape[-1]))
            seq2seq_outputs = np.zeros((num_patient_points, projection_horizon, outputs.shape[-1]))
            seq2seq_prev_outputs = np.zeros((num_patient_points, projection_horizon, outputs.shape[-1]))
            seq2seq_active_entries = np.zeros((num_patient_points, projection_horizon, 1))
            seq2seq_sequence_lengths = np.zeros(num_patient_points)
            seq2seq_original_index = np.zeros(num_patient_points)

            for i in range(num_patient_points):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                if encoder_r is not None:
                    seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]
                seq2seq_active_encoder_r[i, :fact_length] = 1.0
                seq2seq_original_index[i] = i

                seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
                seq2seq_previous_treatments[i] = previous_treatments[i, fact_length:fact_length + projection_horizon, :]
                seq2seq_current_treatments[i] = current_treatments[i, fact_length:fact_length + projection_horizon, :]
                seq2seq_outputs[i] = outputs[i, fact_length: fact_length + projection_horizon, :]
                seq2seq_prev_outputs[i] = prev_outputs[i, fact_length: fact_length + projection_horizon, :]

                seq2seq_sequence_lengths[i] = projection_horizon

            # Package outputs
            seq2seq_data = {
                'original_index': seq2seq_original_index,
                'active_encoder_r': seq2seq_active_encoder_r,
                'prev_treatments': seq2seq_previous_treatments,
                'current_treatments': seq2seq_current_treatments,
                'static_features': self.data['static_features'],
                'prev_outputs': seq2seq_prev_outputs,
                'outputs': seq2seq_outputs,
                'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
                'sequence_lengths': seq2seq_sequence_lengths,
                'active_entries': seq2seq_active_entries,
            }
            if encoder_r is not None:
                seq2seq_data['init_state'] = seq2seq_state_inits

            self.data_original = deepcopy(self.data)
            self.data_processed_seq = deepcopy(seq2seq_data)  # For auto-regressive evaluation (self.data will be changed)
            self.data = seq2seq_data

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r and encoder_r is not None:
                self.encoder_r = encoder_r[:, :max_seq_length - projection_horizon, :]

            self.processed_sequential = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (multiple sequences)')

        return self.data

    def process_autoregressive_test(self, encoder_r, encoder_outputs, projection_horizon, save_encoder_r=False):
        """
        Pre-process test dataset for multiple-step-ahead prediction: axillary dataset placeholder for autoregressive prediction
        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            logger.info(f'Processing {self.subset_name} dataset before testing (autoregressive)')

            current_treatments = self.data_original['current_treatments']
            prev_treatments = self.data_original['prev_treatments']

            sequence_lengths = self.data_original['sequence_lengths']
            num_patient_points = current_treatments.shape[0]

            current_dataset = dict()  # Same as original, but only with last n-steps
            current_dataset['prev_treatments'] = np.zeros((num_patient_points, projection_horizon,
                                                           self.data_original['prev_treatments'].shape[-1]))
            current_dataset['current_treatments'] = np.zeros((num_patient_points, projection_horizon,
                                                              self.data_original['current_treatments'].shape[-1]))
            current_dataset['prev_outputs'] = np.zeros((num_patient_points, projection_horizon,
                                                        self.data_original['outputs'].shape[-1]))
            current_dataset['init_state'] = np.zeros((num_patient_points, encoder_r.shape[-1]))
            current_dataset['active_encoder_r'] = np.zeros((num_patient_points, int(sequence_lengths.max() - projection_horizon)))
            current_dataset['active_entries'] = np.ones((num_patient_points, projection_horizon, 1))

            for i in range(num_patient_points):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                current_dataset['init_state'][i] = encoder_r[i, fact_length - 1]
                current_dataset['prev_outputs'][i, 0, :] = encoder_outputs[i, fact_length - 1]
                current_dataset['active_encoder_r'][i, :fact_length] = 1.0

                current_dataset['prev_treatments'][i] = \
                    prev_treatments[i, fact_length - 1:fact_length + projection_horizon - 1, :]
                current_dataset['current_treatments'][i] = current_treatments[i, fact_length:fact_length + projection_horizon, :]

            current_dataset['static_features'] = self.data_original['static_features']

            self.data_processed_seq = deepcopy(self.data)
            self.data = current_dataset
            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r:
                self.encoder_r = encoder_r[:, :int(max(sequence_lengths) - projection_horizon), :]

            self.processed_autoregressive = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (autoregressive)')

        return self.data

    def process_sequential_multi(self, projection_horizon):
        """
        Pre-process test dataset for multiple-step-ahead prediction for multi-input model: marking rolling origin with
            'future_past_split'
        Args:
            projection_horizon: Projection horizon
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            self.data_processed_seq = self.data
            self.data = deepcopy(self.data_original)
            self.data['future_past_split'] = self.data['sequence_lengths'] - projection_horizon
            self.processed_autoregressive = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (autoregressive)')

        return self.data


class SimulateDatasetCollection(RealDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_f)
    """
    def __init__(self,
                 seed: int = 100,
                 num_confounder: int = 1,
                 num_u: int = 1,
                 num_covariates: int = 5,
                 datasize: int = 5000,
                 total_time_step: int = 30,
                 split: dict = {'val': 0.2, 'test': 0.2},
                 projection_horizon: int = 5,
                 autoregressive=True,
                 test_cf=True,
                 gamma = 0.5,
                 **kwargs):
        """
        Args:
            path: Path with MIMIC-3 dataset (HDFStore)
            min_seq_length: Min sequence lenght in cohort
            max_seq_length: Max sequence lenght in cohort
            seed: Seed for random cohort patient selection
            max_number: Maximum number of patients in cohort
            split: Ratio of train / val / test split
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            autoregressive:
        """
        super(SimulateDatasetCollection, self).__init__()
        self.seed = seed
        autoregressive = AutoregressiveSimulation(gamma, num_confounder, num_u, num_covariates)
        treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params, coso_vitals = \
            autoregressive.generate_dataset(datasize, total_time_step)
        
        indices = np.arange(static_features.shape[0])
        train_indices, test_indices = train_test_split(indices, test_size=split['test'], random_state=seed)

        treatments_train = treatments[train_indices]
        outcomes_train = outcomes[train_indices]
        vitals_train = vitals[train_indices]
        outcomes_unscaled_train = outcomes_unscaled[train_indices]
        coso_vitals_train = coso_vitals[train_indices]
        static_features_train = static_features[train_indices]
        treatments_test = treatments[test_indices]
        outcomes_test = outcomes[test_indices]
        vitals_test = vitals[test_indices]
        outcomes_unscaled_test = outcomes_unscaled[test_indices]
        coso_vitals_test = coso_vitals[test_indices]
        static_features_test = static_features[test_indices]
        if split['val'] > 0.0:
            train_indices, val_indices = train_test_split(train_indices, test_size=split['val'] / (1 - split['test']), random_state=2 * seed)
            treatments_train = treatments[train_indices]
            outcomes_train = outcomes[train_indices]
            vitals_train = vitals[train_indices]
            outcomes_unscaled_train = outcomes_unscaled[train_indices]
            coso_vitals_train = coso_vitals[train_indices]
            static_features_train = static_features[train_indices]
            treatments_val = treatments[val_indices]
            outcomes_val = outcomes[val_indices]
            vitals_val = vitals[val_indices]
            outcomes_unscaled_val = outcomes_unscaled[val_indices]
            coso_vitals_val = coso_vitals[val_indices]
            static_features_val = static_features[val_indices]
        active_entries = np.isnan(treatments).any(axis=2) == False
        active_entries = active_entries.astype(float)[:, :, np.newaxis]  # Ensure it is 3D and float type  
        COSO_index = find_S_variable(treatments, outcomes_unscaled, coso_vitals, active_entries)
        if not isinstance(COSO_index, (int, float)):
            COSO_index = 0
        self.train_f = SimulateDataset(treatments_train, outcomes_train, vitals_train, static_features_train, outcomes_unscaled_train, scaling_params, 'train', coso_vitals_train,COSO_index)
        if split['val'] > 0.0:
            self.val_f = SimulateDataset(treatments_val, outcomes_val, vitals_val, static_features_val, outcomes_unscaled_val, scaling_params, 'val', coso_vitals_val,COSO_index)
        self.test_f = SimulateDataset(treatments_test, outcomes_test, vitals_test, static_features_test, outcomes_unscaled_test, scaling_params, 'test', coso_vitals_test,COSO_index)
        if test_cf:
            treatments_cf,outcomes_cf_unscaled,outcomes_cf_scaled=autoregressive.generate_cf(treatments_test,outcomes_unscaled_test)
            self.test_cf_one_step = SimulateDataset(treatments_cf, outcomes_cf_scaled, vitals_test, static_features_test, outcomes_cf_unscaled, scaling_params, 'test', coso_vitals_test,COSO_index)
        self.projection_horizon = projection_horizon
        self.has_vitals = True
        self.autoregressive = autoregressive
        self.processed_data_encoder = True

def find_S_variable(treatments, outcomes, coso_vitals, active_entries):
    num_patients, timesteps, num_covariates = coso_vitals.shape
    most_relevant_var_for_each_patient = np.full((num_patients, 1), np.nan)
    all_data_for_analysis = []
    patient=0
    for patient in range(num_patients):
        for time in range(1, timesteps):
            if active_entries[patient, time, 0] == 1:
                features = coso_vitals[patient, time, :].flatten()
                current_treatment = treatments[patient, time, :].flatten()
                current_outcome = outcomes[patient, time, :].flatten()
                data_for_analysis = np.hstack([features, current_treatment, current_outcome]).reshape(1, -1)
                all_data_for_analysis.append(data_for_analysis)
        patient = patient + 1
    all_data_tensor = torch.tensor(all_data_for_analysis, dtype=torch.float)
    covariates_tensor = all_data_tensor[:, 0, :num_covariates]
    treatments_tensor = all_data_tensor[:, 0, num_covariates:num_covariates + 1]
    outcomes_tensor = all_data_tensor[:, 0, -1:]
    concatenated_data_oucome = torch.cat([covariates_tensor, outcomes_tensor], -1)
    concatenated_data_treatment = torch.cat([covariates_tensor, treatments_tensor], -1)
    _, _, treatment_pvals = pc_simple(concatenated_data_treatment, target=num_covariates, alpha=0.05, isdiscrete=False)
    #print('treatment_pvals',treatment_pvals)
    
    _, _, outcome_pvals = pc_simple(concatenated_data_oucome, target=num_covariates, alpha=0.05, isdiscrete=False)
    #print('outcome_pvals',outcome_pvals)


    treatment_related_vars = {var for var, pval in treatment_pvals.items() if pval >= 0.95}

    outcome_related_vars = {var for var, pval in outcome_pvals.items() if pval >= 0.95}

    relevant_vars = treatment_related_vars.difference(outcome_related_vars)

    if relevant_vars:
        min_pval = float('0')
        most_relevant_var = None
        for var in relevant_vars:
            if var in treatment_pvals and treatment_pvals[var] > min_pval:
                min_pval = treatment_pvals[var]
                most_relevant_var = var
        most_relevant_var_for_each_patient = most_relevant_var

    return most_relevant_var_for_each_patient

