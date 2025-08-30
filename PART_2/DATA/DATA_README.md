# DATA Folder

This directory contains all datasets required to train and evaluate the **CryptoTFT** model in Part 2.  
It is subdivided into two sub‑folders:

---

## 1. TRAINING

The **TRAINING** folder holds the data used for cross‑validated model training.  
Five splits (`split_1 … split_5`) are provided. Each split contains both a training and a validation set stored in multiple formats:

- **.csv files**: Human‑readable tabular files containing the features and targets.  
  - Suffix `_D.csv`: Daily resolution.  
  - Suffix `_H1.csv`: 1‑hour resolution.  
  - Suffix `_H4.csv`: 4‑hour resolution.  

- **.pkl files**: Python pickles containing preprocessed data structures used by PyTorch Lightning.  
  If you prefer working directly with the CSV files, you can ignore the pickles. We used the csv files for information. 

### Feature Columns

All CSVs include the same feature columns:

- **OHLCV**: `open, high, low, close, volume`  
- **Technical indicators**:  
  `RSI, MACD, MACD_signal, macd_histogram, macd_divergence, macd_slope, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, bollinger_mavg, bollinger_hband, bollinger_lband, bollinger_width`  
- **Targets** (only in training sets):  
  `Y_1h, Y_4h, Y_12h, Y_24h, Y_48h`  

### Usage Notes

- **Select the split**: e.g., use `split_1_train_*` and `split_1_val_*` for split 1. Do the same for splits 2–5 (if you do not want to keep 5 as a test split).  
- **Match resolutions**: The model expects three sequences per anchor point:  
  - 100 consecutive 1‑hour rows (`_H1`)  
  - 25 consecutive 4‑hour rows (`_H4`)  
  - 30 consecutive daily rows (`_D`)  
- **Combine features and targets**: The code automatically merges the three resolutions and scales them before training.The Data is already normalized. The lag is 24H between each anchor. 

---

## 2. TRADING

The **TRADING** folder contains the unseen test data used for the back‑test trading simulation.  

Files provided:  
- `btc_h1_test.csv` (1‑hour resolution)  
- `btc_h4_test.csv` (4‑hour resolution)  
- `btc_1d_test.csv` (daily resolution)  

These files correspond to the same feature set as the training data.  
**Important:** They do **not** include target columns, because they represent future data on which predictions are made.

### Usage Notes

- When running the trading script (`TRADING/Code_Trading.py`), point the `--datasets_dir` argument to this folder.  
- The script will automatically detect the timestamp column and process the features.  

---

## Notes

- Ensure that timestamps are in **UTC** and properly formatted (scripts handle automatic conversion and sorting).  
- Do **not** modify these files when submitting your work; they must remain unchanged for reproducibility.  
