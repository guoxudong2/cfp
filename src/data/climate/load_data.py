import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from omegaconf import ListConfig

from src import ROOT_PATH

logger = logging.getLogger(__name__)


def process_static_features(static_features: pd.DataFrame, drop_first=False) -> pd.DataFrame:
    """
    Global standard normalisation of static features & one hot encoding
    Args:
        static_features: pd.DataFrame with unprocessed static features
        drop_first: Dropping first class of one-hot-encoded features

    Returns: pd.DataFrame with pre-processed static features

    """
    processed_static_features = []
    for feature in static_features.columns:
        if isinstance(static_features[feature].iloc[0], float):
            mean = np.mean(static_features[feature])
            std = np.std(static_features[feature])
            processed_static_features.append((static_features[feature] - mean) / std)
        else:
            one_hot = pd.get_dummies(static_features[feature], drop_first=drop_first)
            processed_static_features.append(one_hot.astype(float))

    static_features = pd.concat(processed_static_features, axis=1)
    return static_features


def load_climate_data_processed(data_path: str,
                               min_seq_length: int = None,
                               max_seq_length: int = None,
                               treatment_list: List[str] = None,
                               outcome_list: List[str] = None,
                               vital_list: List[str] = None,
                               static_list: List[str] = None,
                               max_number: int = None,
                               data_seed: int = 100,
                               drop_first=False,
                               **kwargs) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict):
    """
    Load and pre-process MIMIC-3 hourly averaged dataset (for real-world experiments)
    :param data_path: Path with MIMIC-3 dataset (HDFStore)
    :param min_seq_length: Min sequence lenght in cohort
    :param min_seq_length: Max sequence lenght in cohort
    :param treatment_list: List of treaments
    :param outcome_list: List of outcomes
    :param vital_list: List of vitals (time-varying covariates)
    :param static_list: List of static features
    :param max_number: Maximum number of patients in cohort
    :param data_seed: Seed for random cohort patient selection
    :param drop_first: Dropping first class of one-hot-encoded features
    :return: tuple of DataFrames and params (treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params)
    """

    logger.info(f'Loading Climate dataset from {data_path}.')

    df = pd.read_csv(data_path)
    df = df.set_index(['lat', 'lon'])
    logger.info(f'dffff columns: {df.columns}')
    if treatment_list is None:
        treatment_list = ['cfnlf','wspd','skt']
    if outcome_list is None:
        outcome_list = ['prate']
    else:
        outcome_list = ListConfig([outcome.replace('_', ' ') for outcome in outcome_list])
    if vital_list is None:
        vital_list = [
            'pres',
            'air',
            'dswrf',
            'csusf',
            'csdlf',
            'tmin'
        ]
    if static_list is None:
        static_list = [
        ]

    treatments = df[treatment_list]
    logger.info(f'treatment shape1 {treatments.shape}')
    all_vitals = df[outcome_list + vital_list]
    static_features = df[static_list]

    logger.info(f'df index len: {len(df.index)}, is unique: {df.index.is_unique}')
    # 如果需要基于lat进行过滤（假设lat是ID）
    #lat_group_sizes = df.groupby('lat').size()
    lat_group_indexs = df.groupby(level=[0,1])
    logger.info(f'lat_group_indexs: {lat_group_indexs}')
    lat_group_sizes = df.groupby(level=[0,1]).size()
    logger.info(f'lat_group_sizes shape: {lat_group_sizes.shape}')
    filtered_ids = lat_group_sizes.index[lat_group_sizes >= min_seq_length] if min_seq_length is not None else lat_group_sizes.index
    logger.info(f'filtered_ids shape: {filtered_ids.shape}')#403

    # 如果定义了最大数量，进行随机选择
    if max_number is not None:
        if max_number > len(filtered_ids):
            max_number = len(filtered_ids)
        np.random.seed(data_seed)
        filtered_ids = np.random.choice(filtered_ids, size=max_number, replace=False)
        logger.info(f'filtered_ids shape: {filtered_ids.shape}')

    #treatments = treatments[df['lat'].isin(filtered_ids)]
    #all_vitals = all_vitals[df['lat'].isin(filtered_ids)]

    logger.info(f'filtered_ids shape: {filtered_ids.shape}')
    treatments = treatments.loc[filtered_ids]
    all_vitals = all_vitals.loc[filtered_ids]

    logger.info(f'aaaaaaaaaaaaaaaa')
    logger.info(f'treatment columns {treatments.columns}')
    logger.info(f'treatment head2 {treatments.head(2)}, treatment shape {treatments.shape}')
    # 如果定义了最大序列长度，进行限制
    if max_seq_length is not None:
        treatments = treatments.groupby(level=[0,1]).head(max_seq_length)
        all_vitals = all_vitals.groupby(level=[0,1]).head(max_seq_length)
        logger.info(f'treatment shape2 {treatments.shape}, treatment: {treatments.head(2)}')
        #                               (12000,1)

    #static_features = static_features[df['lat'].isin(filtered_ids)]
    static_features = static_features.loc[filtered_ids]

    # 填充缺失值（如果必要）
    #all_vitals = all_vitals.fillna(method='ffill').fillna(method='bfill')

    logger.info(f'bbbbbbbbbbbbbbbbbbb')
    # 缩放体征和结果数据
    outcomes_unscaled = all_vitals[outcome_list].copy()
    mean = all_vitals.mean(axis=0)
    std = all_vitals.std(axis=0)
    all_vitals = (all_vitals - mean) / std

    logger.info(f'cccccccccccccccccc')
    # 分离结果和体征
    outcomes = all_vitals[outcome_list].copy()
    vitals = all_vitals[vital_list].copy()
    coso_vitals = vitals.copy()
    if static_features is not None and not static_features.empty:
        static_features = process_static_features(static_features, drop_first=drop_first)

    logger.info(f'ddddddddddddddddd')
    # 准备缩放参数
    scaling_params = {
        'output_means': mean[outcome_list].to_numpy(),
        'output_stds': std[outcome_list].to_numpy(),
    }

    logger.info(f'eeeeeeeeeeeeeeeee')
    logger.info(f'len(filtered_ids: {len(filtered_ids)}.')
    logger.info(f'filtered_ids: {filtered_ids}.')
    logger.info(f'treatments.shape {treatments.shape}, outcomes.shape {outcomes.shape}, vitals.shape{vitals.shape}, coso_vitals.shape {coso_vitals.shape}2222')
    #                               (12000,1)
    logger.info(f'treatments.head5 {treatments.head(125)}, outcomes.head5 {outcomes.head(125)}, vitals.head5{vitals.head(125)}, coso_vitals.shape {coso_vitals.shape}2222')
    return treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params, coso_vitals



def load_mimic3_data_raw(data_path: str,
                         min_seq_length: int = None,
                         max_seq_length: int = None,
                         max_number: int = None,
                         vital_list: List[str] = None,
                         static_list: List[str] = None,
                         data_seed: int = 100,
                         drop_first=False,
                         **kwargs) -> (pd.DataFrame, pd.DataFrame):
    """
    Load MIMIC-3 hourly averaged dataset, without preprocessing (for semi-synthetic experiments)
    :param data_path: Path with MIMIC-3 dataset (HDFStore)
    :param min_seq_length: Min sequence lenght in cohort
    :param max_seq_length: Max sequence length in cohort
    :param vital_list: List of vitals (time-varying covariates)
    :param static_list: List of static features
    :param max_number: Maximum number of patients in cohort
    :param data_seed: Seed for random cohort patient selection
    :param drop_first: Dropping first class of one-hot-encoded features
    :return: Tuple of DataFrames (all_vitals, static_features)
    """
    logger.info(f'Loading MIMIC-III dataset from {data_path}.')

    h5 = pd.HDFStore(data_path, 'r')
    if vital_list is None:
        vital_list = [
            'heart rate',
            'red blood cell count',
            'sodium',
            'mean blood pressure',
            'systemic vascular resistance',
            'glucose',
            'chloride urine',
            'glascow coma scale total',
            'hematocrit',
            'positive end-expiratory pressure set',
            'respiratory rate',
            'prothrombin time pt',
            'cholesterol',
            'hemoglobin',
            'creatinine',
            'blood urea nitrogen',
            'bicarbonate',
            'calcium ionized',
            'partial pressure of carbon dioxide',
            'magnesium',
            'anion gap',
            'phosphorous',
            'venous pvo2',
            'platelets',
            'calcium urine'
        ]
    if static_list is None:
        static_list = [
            'gender',
            'ethnicity',
            'age'
        ]

    all_vitals = h5['/vitals_labs_mean'][vital_list]
    static_features = h5['/patients'][static_list]

    all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
    column_names = []
    for column in all_vitals.columns:
        if isinstance(column, str):
            column_names.append(column)
        else:
            column_names.append(column[0])
    all_vitals.columns = column_names
    static_features = static_features.droplevel(['hadm_id', 'icustay_id'])

    # Filling NA
    all_vitals = all_vitals.fillna(method='ffill')
    all_vitals = all_vitals.fillna(method='bfill')

    # Filtering longer then min_seq_length and cropping to max_seq_length
    user_sizes = all_vitals.groupby('subject_id').size()
    filtered_users = user_sizes.index[user_sizes >= min_seq_length] if min_seq_length is not None else user_sizes.index
    if max_number is not None:
        np.random.seed(data_seed)
        filtered_users = np.random.choice(filtered_users, size=max_number, replace=False)
    all_vitals = all_vitals.loc[filtered_users]
    static_features = static_features.loc[filtered_users]
    if max_seq_length is not None:
        all_vitals = all_vitals.groupby('subject_id').head(max_seq_length)
    logger.info(f'Number of patients filtered: {len(filtered_users)}.')

    # Global Mean-Std Normalisation
    mean = np.mean(all_vitals, axis=0)
    std = np.std(all_vitals, axis=0)
    all_vitals = (all_vitals - mean) / std

    static_features = process_static_features(static_features, drop_first=drop_first)

    h5.close()
    return all_vitals, static_features


if __name__ == "__main__":
    data_path = ROOT_PATH + '/' + 'data/processed/all_hourly_data.h5'
    treatments, outcomes, vitals, stat_features, outcomes_unscaled, scaling_params = \
        load_mimic3_data_processed(data_path, min_seq_length=100, max_seq_length=100)
