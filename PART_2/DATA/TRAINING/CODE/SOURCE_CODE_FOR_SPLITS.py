# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 11:08:16 2025

@author: rodolphe.lucas
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import timedelta

def compute_global_stats(df, feature_cols, normalization_methods):
    stats = {}
    for col in feature_cols:
        data = df[col].values.astype(np.float32)
        if normalization_methods[col] == 'minmax':
            stats[col] = (np.min(data), np.max(data))
        elif normalization_methods[col] == 'zscore':
            stats[col] = (np.mean(data), np.std(data))
    return stats

def normalize_sequence(sequence, feature_cols, normalization_methods, stats):
    normalized_sequence = sequence.copy().astype(np.float32)
    epsilon = 1e-8
    for i, col in enumerate(feature_cols):
        method = normalization_methods[col]
        if method == 'minmax':
            min_val, max_val = stats[col]
            if max_val > min_val:
                normalized_sequence[:, i] = (sequence[:, i] - min_val) / (max_val - min_val)
            else:
                normalized_sequence[:, i] = 0.0
        elif method == 'zscore':
            mean_val, std_val = stats[col]
            if std_val > 0:
                normalized_sequence[:, i] = (sequence[:, i] - mean_val) / (std_val + epsilon)
            else:
                normalized_sequence[:, i] = 0.0
    return normalized_sequence

def extract_split_data(split_id, df_splits, df_H1, df_H4, df_D):
    split_row = df_splits[df_splits['split_id'] == split_id].iloc[0]
    train_start = pd.to_datetime(split_row['train_start'])
    train_end = pd.to_datetime(split_row['train_end'])
    val_start = pd.to_datetime(split_row['val_start'])
    val_end = pd.to_datetime(split_row['val_end'])

    train_H1 = df_H1[(df_H1['timestamp'] >= train_start) & (df_H1['timestamp'] <= train_end)].copy()
    val_H1 = df_H1[(df_H1['timestamp'] >= val_start) & (df_H1['timestamp'] <= val_end)].copy()
    train_H4 = df_H4[(df_H4['timestamp'] >= train_start) & (df_H4['timestamp'] <= train_end)].copy()
    val_H4 = df_H4[(df_H4['timestamp'] >= val_start) & (df_H4['timestamp'] <= val_end)].copy()
    train_D = df_D[(df_D['timestamp'] >= train_start) & (df_D['timestamp'] <= train_end)].copy()
    val_D = df_D[(df_D['timestamp'] >= val_start) & (df_D['timestamp'] <= val_end)].copy()

    return train_H1, train_H4, train_D, val_H1, val_H4, val_D

def build_sequences_and_labels(df_H1, df_H4, df_D, stats_H1, stats_H4, stats_D, feature_cols, normalization_methods, sequence_lengths=(100, 25, 30), horizons=[1, 4, 12, 24, 48]):
    H1_seq_len, H4_seq_len, D_seq_len = sequence_lengths
    F = len(feature_cols)
    
    # Initialiser les DataFrames pour chaque timeframe
    records_H1, records_H4, records_D = [], [], []
    
    start_time = max(
        df_H1['timestamp'].min() + timedelta(hours=H1_seq_len),
        df_H4['timestamp'].min() + timedelta(hours=4*H4_seq_len),
        df_D['timestamp'].min() + timedelta(days=D_seq_len)
    )
    end_time = min(
        df_H1['timestamp'].max() - timedelta(hours=max(horizons)),
        df_H4['timestamp'].max(),
        df_D['timestamp'].max()
    )

    current_time = start_time
    seq_idx = 0
    while current_time <= end_time:
        H1_slice = df_H1[(df_H1['timestamp'] <= current_time) & 
                         (df_H1['timestamp'] > current_time - timedelta(hours=H1_seq_len))]
        H4_slice = df_H4[(df_H4['timestamp'] <= current_time) & 
                         (df_H4['timestamp'] > current_time - timedelta(hours=4*H4_seq_len))]
        D_slice = df_D[(df_D['timestamp'] <= current_time) & 
                       (df_D['timestamp'] > current_time - timedelta(days=D_seq_len))]

        if (len(H1_slice) == H1_seq_len and 
            len(H4_slice) == H4_seq_len and 
            len(D_slice) == D_seq_len):

            future_prices = []
            valid_horizons = True
            for h in horizons:
                future_time = current_time + timedelta(hours=h)
                future_price = df_H1[df_H1['timestamp'] == future_time]['close']
                if future_price.empty:
                    valid_horizons = False
                    break
                future_prices.append(future_price.iloc[0])

            if valid_horizons:
                # Normaliser les séquences
                H1_seq = normalize_sequence(H1_slice[feature_cols].values, feature_cols, normalization_methods, stats_H1)
                H4_seq = normalize_sequence(H4_slice[feature_cols].values, feature_cols, normalization_methods, stats_H4)
                D_seq = normalize_sequence(D_slice[feature_cols].values, feature_cols, normalization_methods, stats_D)

                # Créer les records pour H1
                for t in range(H1_seq_len):
                    record = {
                        'group_id': seq_idx,
                        'time_idx': t,
                        'timestamp': H1_slice['timestamp'].iloc[t]
                    }
                    for f_idx, col in enumerate(feature_cols):
                        record[col] = H1_seq[t, f_idx]
                    if t == H1_seq_len - 1:
                        current_price = H1_slice['close'].iloc[-1]
                        for h_idx, h in enumerate(horizons):
                            record[f'Y_{h}h'] = np.tanh((future_prices[h_idx] - current_price) / current_price)
                    else:
                        for h in horizons:
                            record[f'Y_{h}h'] = np.nan
                    records_H1.append(record)

                # Créer les records pour H4
                for t in range(H4_seq_len):
                    record = {
                        'group_id': seq_idx,
                        'time_idx': t,
                        'timestamp': H4_slice['timestamp'].iloc[t]
                    }
                    for f_idx, col in enumerate(feature_cols):
                        record[col] = H4_seq[t, f_idx]
                    if t == H4_seq_len - 1:
                        current_price = H1_slice['close'].iloc[-1]  # Utiliser H1 pour cohérence
                        for h_idx, h in enumerate(horizons):
                            record[f'Y_{h}h'] = np.tanh((future_prices[h_idx] - current_price) / current_price)
                    else:
                        for h in horizons:
                            record[f'Y_{h}h'] = np.nan
                    records_H4.append(record)

                # Créer les records pour Daily
                for t in range(D_seq_len):
                    record = {
                        'group_id': seq_idx,
                        'time_idx': t,
                        'timestamp': D_slice['timestamp'].iloc[t]
                    }
                    for f_idx, col in enumerate(feature_cols):
                        record[col] = D_seq[t, f_idx]
                    if t == D_seq_len - 1:
                        current_price = H1_slice['close'].iloc[-1]
                        for h_idx, h in enumerate(horizons):
                            record[f'Y_{h}h'] = np.tanh((future_prices[h_idx] - current_price) / current_price)
                    else:
                        for h in horizons:
                            record[f'Y_{h}h'] = np.nan
                    records_D.append(record)

                seq_idx += 1

        current_time += timedelta(hours=24)

    # Convertir en DataFrames
    df_H1 = pd.DataFrame(records_H1)
    df_H4 = pd.DataFrame(records_H4)
    df_D = pd.DataFrame(records_D)
    
    # Retourner les tenseurs pour compatibilité avec l’ancien pipeline
    result = {
        'X_H1': np.array([df_H1[df_H1['group_id'] == i][feature_cols].values for i in range(seq_idx)], dtype=np.float32),
        'X_H4': np.array([df_H4[df_H4['group_id'] == i][feature_cols].values for i in range(seq_idx)], dtype=np.float32),
        'X_D': np.array([df_D[df_D['group_id'] == i][feature_cols].values for i in range(seq_idx)], dtype=np.float32),
        'Y': np.array([df_H1[df_H1['group_id'] == i][[f'Y_{h}h' for h in horizons]].iloc[-1].values for i in range(seq_idx)], dtype=np.float32),
        'timestamps': [df_H1[df_H1['group_id'] == i]['timestamp'].iloc[-1] for i in range(seq_idx)]
    }
    
    return df_H1, df_H4, df_D, result


def save_datasets(df_H1, df_H4, df_D, tensor_data, split_id, dataset_type, output_dir='datasets_TFT_npz_24H_large'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les DataFrames
    df_H1.to_csv(os.path.join(output_dir, f'split_{split_id}_{dataset_type}_H1.csv'), index=False)
    df_H4.to_csv(os.path.join(output_dir, f'split_{split_id}_{dataset_type}_H4.csv'), index=False)
    df_D.to_csv(os.path.join(output_dir, f'split_{split_id}_{dataset_type}_D.csv'), index=False)
    
    # Sauvegarder les tenseurs (compatibilité avec l’ancien pipeline)
    pkl_filename = os.path.join(output_dir, f'split_{split_id}_{dataset_type}.pkl')
    with open(pkl_filename, 'wb') as f:
        pickle.dump(tensor_data, f)

def process_split_and_save(split_id, df_splits, df_H1, df_H4, df_D):
    for df in [df_H1, df_H4, df_D, df_splits]:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

    train_H1, train_H4, train_D, val_H1, val_H4, val_D = extract_split_data(
        split_id, df_splits, df_H1, df_H4, df_D
    )

    feature_cols = [col for col in df_H1.columns if col not in ['timestamp', 'timeframe']]
    normalization_methods = {
        'open': 'minmax', 'high': 'minmax', 'low': 'minmax', 'close': 'minmax', 'volume': 'minmax',
        'momentum': 'zscore', 'range': 'zscore',
        'RSI': 'minmax',
        'MACD': 'zscore', 'MACD_signal': 'zscore', 'macd_histogram': 'zscore',
        'macd_divergence': 'zscore', 'macd_slope': 'zscore',
        'tenkan_sen': 'minmax', 'kijun_sen': 'minmax', 'senkou_span_a': 'minmax', 'senkou_span_b': 'minmax',
        'bollinger_mavg': 'minmax', 'bollinger_hband': 'minmax', 'bollinger_lband': 'minmax',
        'bollinger_width': 'zscore'
    }

    stats_H1 = compute_global_stats(train_H1, feature_cols, normalization_methods)
    stats_H4 = compute_global_stats(train_H4, feature_cols, normalization_methods)
    stats_D = compute_global_stats(train_D, feature_cols, normalization_methods)

    train_df_H1, train_df_H4, train_df_D, train_tensor_data = build_sequences_and_labels(
        train_H1, train_H4, train_D, stats_H1, stats_H4, stats_D, feature_cols, normalization_methods
    )
    val_df_H1, val_df_H4, val_df_D, val_tensor_data = build_sequences_and_labels(
        val_H1, val_H4, val_D, stats_H1, stats_H4, stats_D, feature_cols, normalization_methods
    )

    save_datasets(train_df_H1, train_df_H4, train_df_D, train_tensor_data, split_id, 'train')
    save_datasets(val_df_H1, val_df_H4, val_df_D, val_tensor_data, split_id, 'val')


if __name__ == "__main__":
    df_splits = pd.read_csv('Splits_temporels_BTC_LARGE.csv')

    df_H1 = pd.read_csv('btc_h1_final.csv')
    df_H4 = pd.read_csv('btc_h4_final.csv')
    df_D = pd.read_csv('btc_1d_final.csv')
    
    
    for df in [df_H1, df_H4, df_D]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for split_id in df_splits['split_id'].unique():
        process_split_and_save(split_id, df_splits, df_H1, df_H4, df_D)