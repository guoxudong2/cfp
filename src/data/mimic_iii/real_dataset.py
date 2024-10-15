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
logger = logging.getLogger(__name__)


class MIMIC3RealDataset(Dataset):
    """
    Pytorch-style real-world MIMIC-III dataset
    """
    def __init__(self,
                 treatments: pd.DataFrame,
                 outcomes: pd.DataFrame,
                 vitals: pd.DataFrame,
                 static_features: pd.DataFrame,
                 outcomes_unscaled: pd.DataFrame,
                 scaling_params: dict,
                 subset_name: str,
                 coso_vitals:pd.DataFrame,
                 COSO_index):
        """
        Args:
            treatments: DataFrame with treatments; multiindex by (patient_id, timestep)
            outcomes: DataFrame with outcomes; multiindex by (patient_id, timestep)
            vitals: DataFrame with vitals (time-varying covariates); multiindex by (patient_id, timestep)
            static_features: DataFrame with static features
            outcomes_unscaled: DataFrame with unscaled outcomes; multiindex by (patient_id, timestep)
            scaling_params: Standard normalization scaling parameters
            subset_name: train / val / test
        """
        assert treatments.shape[0] == outcomes.shape[0]
        assert outcomes.shape[0] == vitals.shape[0]

        self.subset_name = subset_name
        user_sizes = vitals.groupby('subject_id').size()

        # Padding with nans
        treatments = treatments.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        outcomes = outcomes.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        vitals = vitals.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        outcomes_unscaled = outcomes_unscaled.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        active_entries = (~treatments.isna().any(axis=1)).astype(float)
        coso_vitals=coso_vitals.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        static_features = static_features.sort_index()
        user_sizes = user_sizes.sort_index()

        # Conversion to np.arrays
        treatments = treatments.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1)).astype(float)
        outcomes = outcomes.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1))
        vitals = vitals.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1))
        outcomes_unscaled = outcomes_unscaled.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1))
        active_entries = active_entries.values.reshape((len(user_sizes), max(user_sizes), 1))
        static_features = static_features.values
        user_sizes = user_sizes.values
        coso_vitals = coso_vitals.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1))

        COSO = coso_vitals[:, :, COSO_index, np.newaxis]
        coso_vitals = np.delete(coso_vitals, COSO_index, axis=2)

        self.data = {
            'sequence_lengths': user_sizes - 1,
            'prev_treatments': treatments[:, :-1, :],
            'vitals': vitals[:, 1:, :],
            'next_vitals': vitals[:, 2:, :],
            'current_treatments': treatments[:, 1:, :],
            'static_features': static_features,
            'active_entries': active_entries[:, 1:, :],
            'outputs': outcomes[:, 1:, :],
            'unscaled_outputs': outcomes_unscaled[:, 1:, :],
            'prev_outputs': outcomes[:, :-1, :],
            #'coso_vitals': coso_vitals[:, 1:, :],
            'Adjustment_vitals': coso_vitals[:, 1:, :],
            'COSO':COSO[:, 1:, :]
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


class MIMIC3RealDatasetCollection(RealDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_f)
    """
    def __init__(self,
                 path: str,
                 min_seq_length: int = 30,
                 max_seq_length: int = 60,
                 seed: int = 100,
                 max_number: int = None,
                 split: dict = {'val': 0.2, 'test': 0.2},
                 projection_horizon: int = 5,
                 autoregressive=True,
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
        super(MIMIC3RealDatasetCollection, self).__init__()
        self.seed = seed
        treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params, coso_vitals = \
            load_mimic3_data_processed(ROOT_PATH + '/' + path, min_seq_length=min_seq_length, max_seq_length=max_seq_length,
                                       max_number=max_number, data_seed=seed, **kwargs)

        logger.info(f'AAAAreal_dataset: treatments.shape {treatments.shape}, outcomes.shape {outcomes.shape}, vitals.shape{vitals.shape}, coso_vitals.shape {coso_vitals.shape}2222')
        #             AAAAreal_dataset: treatments.shape (261137, 1),        outcomes.shape (261137, 1),      vitals.shape(261137, 25),    coso_vitals.shape (261137, 25)2222
        logger.info(f'static_features.shape: {static_features.shape}') #(5000, 44)
        logger.info(f'static_features.cloumns: {static_features.columns}')
        '''Index(['F', 'M', 'AMERICAN INDIAN/ALASKA NATIVE',
       'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE', 'ASIAN',
       'ASIAN - ASIAN INDIAN', 'ASIAN - CAMBODIAN', 'ASIAN - CHINESE',
       'ASIAN - FILIPINO', 'ASIAN - JAPANESE', 'ASIAN - KOREAN',
       'ASIAN - OTHER', 'ASIAN - THAI', 'ASIAN - VIETNAMESE', 'BLACK/AFRICAN',
       'BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/HAITIAN',
       'CARIBBEAN ISLAND', 'HISPANIC OR LATINO',
       'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)',
       'HISPANIC/LATINO - COLOMBIAN', 'HISPANIC/LATINO - CUBAN',
       'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - GUATEMALAN',
       'HISPANIC/LATINO - HONDURAN', 'HISPANIC/LATINO - MEXICAN',
       'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - SALVADORAN',
       'MIDDLE EASTERN', 'MULTI RACE ETHNICITY',
       'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'OTHER',
       'PATIENT DECLINED TO ANSWER', 'PORTUGUESE', 'SOUTH AMERICAN',
       'UNABLE TO OBTAIN', 'UNKNOWN/NOT SPECIFIED', 'WHITE',
       'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN',
       'WHITE - OTHER EUROPEAN', 'WHITE - RUSSIAN', 'age'],
      dtype='object')'''
        logger.info(f'static_features.head: {static_features.head(5)}')
        logger.info(f'static_features.index: {static_features.index}')
        ''''[    3,     6,    23,    26,    32,    41,    52,    55,    65,
            99573, 99633, 99693, 99777, 99810, 99819, 99864, 99881, 99934,
            99939]'''

        # Train/val/test random_split
        static_features, static_features_test = train_test_split(static_features, test_size=split['test'], random_state=seed)
        treatments, outcomes, vitals, outcomes_unscaled, treatments_test, outcomes_test, vitals_test, outcomes_unscaled_test, coso_vitals, coso_vitals_test = \
            treatments.loc[static_features.index], \
            outcomes.loc[static_features.index], \
            vitals.loc[static_features.index], \
            outcomes_unscaled.loc[static_features.index], \
            treatments.loc[static_features_test.index], \
            outcomes.loc[static_features_test.index], \
            vitals.loc[static_features_test.index], \
            outcomes_unscaled.loc[static_features_test.index], \
            coso_vitals.loc[static_features.index],\
            coso_vitals.loc[static_features_test.index]
        #Prepare data to compute S
        logger.info(f'BBBBtreatments: {treatments.shape}, outcomes: {outcomes.shape}, coso_vitals: {coso_vitals.shape}')
        #             BBBBtreatments: (221894, 1),        outcomes: (221894, 1),      coso_vitals: (221894, 25)
        logger.info(f'treatments.head(): {treatments.head()}, treatments index {treatments.index}')
        #treatments.head():                      vaso
        # subject_id hours_in
        #   63430      0            0
        #              1            0
        #              2            0
        #              3            0
        #              4            0,        treatments index
        #MultiIndex([(63430,  0),
        #    (63430,  1),
        #    (63430,  2),
        #    (63430,  3),
        #    (63430,  4),
        #    (63430,  5),
        #    (63430,  6),
        #    (63430,  7),
        #    (63430,  8),
        #    (63430,  9),
        #    ...
        #    ( 2262, 50),
        #    ( 2262, 51),
        #    ( 2262, 52),
        #    ( 2262, 53),
        #    ( 2262, 54),
        #    ( 2262, 55),
        #    ( 2262, 56),
        #    ( 2262, 57),
        #    ( 2262, 58),
        #    ( 2262, 59)],
        #   names=['subject_id', 'hours_in'], length=221894)
        user_sizes = treatments.index.get_level_values(0).value_counts().sort_index()
        logger.info(f'user_sizes shape: {user_sizes.shape}, user_sizes index: {user_sizes.index}')
        #user_sizes shape: (4250,),
        #                                                  user_sizes index:
        #Int64Index([    3,     6,    23,    26,    32,    52,    55,    65,    68,
        #       71,
        #    ...
        #    99545, 99573, 99633, 99693, 99777, 99810, 99819, 99864, 99934,
        #    99939],
        #   dtype='int64', length=4250)
        treatments_processed = treatments.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        outcomes_processed = outcomes.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        coso_vitals_processed = coso_vitals.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        active_entries = (~treatments_processed.isna().any(axis=1)).astype(float)
        logger.info(f'XXXXtreatments_processed: {treatments_processed.shape}, outcomes_processed: {outcomes_processed.shape}, coso_vitals_processed: {coso_vitals_processed.shape}, active_entries: {active_entries.shape}')
        #             XXXXtreatments_processed: (255000, 1),                  outcomes_processed: (255000, 1),                coso_vitals_processed: (255000, 25),                  active_entries: (255000,)

        treatments_np = treatments_processed.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1)).astype(float)
        outcomes_np = outcomes_processed.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1))
        coso_vitals_np = coso_vitals_processed.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1))
        active_entries_np = active_entries.values.reshape((len(user_sizes), max(user_sizes), 1))
        logger.info(f'treatments_np: {treatments_np.shape}, outcomes_np: {outcomes_np.shape}, coso_vitals_np: {coso_vitals_np.shape}, active_entries_np: {active_entries_np.shape}1111')
        #             treatments_np: (4250, 60, 1),         outcomes_np: (4250, 60, 1),       coso_vitals_np: (4250, 60, 25),         active_entries_np: (4250, 60, 1)1111
        treatments_df_np = treatments_np
        treatments_df_np = treatments_df_np.reshape(4250,60)
        treatments_df_np = pd.DataFrame(treatments_df_np)
        logger.info(f'treatments_np head(300): {treatments_df_np.head(300)}')
        COSO_index = find_S_variable(treatments_np, outcomes_np, coso_vitals_np, active_entries_np)
        COSO_index = 16
        if split['val'] > 0.0:
            static_features_train, static_features_val = train_test_split(static_features,
                                                                        test_size=split['val'] / (1 - split['test']),
                                                                        random_state=2 * seed)
            treatments_train, outcomes_train, vitals_train, outcomes_unscaled_train, treatments_val, outcomes_val, vitals_val, \
                outcomes_unscaled_val, coso_vitals_train, coso_vitals_val= \
                treatments.loc[static_features_train.index], \
                outcomes.loc[static_features_train.index], \
                vitals.loc[static_features_train.index], \
                outcomes_unscaled.loc[static_features_train.index], \
                treatments.loc[static_features_val.index], \
                outcomes.loc[static_features_val.index], \
                vitals.loc[static_features_val.index], \
                outcomes_unscaled.loc[static_features_val.index], \
                coso_vitals.loc[static_features_train.index], \
                coso_vitals.loc[static_features_val.index]
        else:
            static_features_train = static_features
            treatments_train, outcomes_train, vitals_train, outcomes_unscaled_train, coso_vitals_train = \
                treatments, outcomes, vitals, outcomes_unscaled, coso_vitals
            

        self.train_f = MIMIC3RealDataset(treatments_train, outcomes_train, vitals_train, static_features_train, outcomes_unscaled_train, scaling_params, 'train', coso_vitals_train,COSO_index)
        if split['val'] > 0.0:
            self.val_f = MIMIC3RealDataset(treatments_val, outcomes_val, vitals_val, static_features_val, outcomes_unscaled_val, scaling_params, 'val', coso_vitals_val,COSO_index)
        self.test_f = MIMIC3RealDataset(treatments_test, outcomes_test, vitals_test, static_features_test, outcomes_unscaled_test, scaling_params, 'test', coso_vitals_test,COSO_index)
        
        self.projection_horizon = projection_horizon
        self.has_vitals = True
        self.autoregressive = autoregressive
        self.processed_data_encoder = True

def find_S_variable(treatments, outcomes, coso_vitals, active_entries):
    num_patients, timesteps, num_covariates = coso_vitals.shape
    most_relevant_var_for_each_patient = np.full((num_patients, 1), np.nan)
    all_data_for_analysis = []
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
    _, _, outcome_pvals = pc_simple(concatenated_data_oucome, target=num_covariates, alpha=0.05, isdiscrete=False)
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

    print(most_relevant_var_for_each_patient)
    return most_relevant_var_for_each_patient
