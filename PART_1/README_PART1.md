# Part I — Classical & Keras Models (README)

This README documents **Part 1** of the technical work.  
It explains what is inside the two main folders and how to use them:

- **`DATA/`** → *training data for the models*  
- **`MODELS/`** → *all model scripts, Keras tuning artefacts and evaluation outputs*

> All Part‑1 models are implemented in **Keras/TensorFlow**. Best configurations are exported and the trained checkpoints are saved as `.keras` files.


---

## 1) Folder Overview 

```
C:\Users\rodolphe.lucas\OneDrive - ESTIA\Bureau\ESTIA3A\Cours-UK\THESIS\TECHNICAL_WORK\PART_1
├── DATA
│   └── btc_4H_onehot_thresh_0.47.csv
└── MODELS
    ├── CNN+LSTM\        # 1D CNN + LSTM classifier
    ├── FNN\             # Fully-Connected Neural Network
    ├── GRU_LSTM\        # GRU + LSTM 
    ├── LSTM\            #  LSTM classifier
    ├── ROLLING_WALK_FORWARD\  # Walk-forward evaluation scripts & reports
    ├── TCN\             # Temporal Convolutional Network
    ├── TRANSFORMERS_GRU\# Transformer-style encoder + GRU head
    └── XGBOOST\         # Gradient boosting baseline (non-neural)
```


---

## 2) DATA — Training Data for the Models

- File: **`DATA/btc_4H_onehot_thresh_0.47.csv`**  
  - Timeframe: **4‑hour** bars (H4).  
  - Targets: **one‑hot encoded** directional classes (threshold = **0.47**) for classification (e.g., *Buy / Hold / Sell*).  
  - Features: market variables and engineered indicators as prepared in your preprocessing pipeline.
- This single CSV is the **source** for Part‑1 training/evaluation across all models.

### Typical usage (Python)
Each model script loads this CSV and splits features/targets internally. If you need a quick test:


# Example: separate features/targets depending on your naming convention
# (Adjust if your label columns use different names.)
label_cols = [c for c in df.columns if "onehot" in c.lower() or c.lower().startswith(("label_", "y_"))]
X = df.drop(columns=label_cols)
y = df[label_cols]
print(X.shape, y.shape)
```

> 🔁 keep this CSV unchanged in submissions for strict reproducibility.


---

## 3) MODELS — All Models and Artefacts

Each subfolder contains:
- **Training script** (`*.py`) with the model definition (**Keras**),
- **Keras Tuner artefacts**: `best_hyperparameters*.json` (when applicable),
- **Best checkpoint**: `best_*_model.keras` (or `<model>_model.keras`),
- **Evaluation outputs**: loss curves, confusion matrices, text reports.

### Subfolders

#### `MODELS/CNN+LSTM/`
- **`1D_CNN_LSTM.py`**: 1D‑CNN feature extractor + LSTM classifier.
- **`best_hyperparameters.json`**: Keras Tuner best config.
- **`best_cnn_lstm_model.keras` / `cnn_lstm_model.keras`**: saved models.
- **`loss_plot.png`**, **`confusion_matrix_test_set.png`**: training curve & test diagnostics.

#### `MODELS/FNN/`
- **`FNN.py`**: Fully‑connected network baseline.
- **`best_hyperparameters (9).json`**, **`best_fnn_model.keras`**, **`fnn_model.keras`**: tuning + checkpoints.

#### `MODELS/GRU_LSTM/`
- **`GRU_LSTM_ATTENTION.py`**: GRU + (bi)LSTM with attention mechanism.
- **`best_hyperparameters (8).json`**, **`best_gru_lstm_model (3).keras`**, **`gru_lstm_model (4).keras`**.
- **`loss_plot (8).png`**, **`confusion_matrix_test_set (9).png`**, **`Test_Set_report (10).txt`**.

#### `MODELS/LSTM/`
- **`LSTM.py`**: vanilla LSTM classifier.
- **`best_hyperparameters.json`**, **`lstm_model.keras`** (and variants), **`loss_plot.png`**, **`confusion_matrix_test_set.png`**, **`Test_Set_report.txt`**.

#### `MODELS/ROLLING_WALK_FORWARD/`
- **`ROLLING_WALK_FORWARD.py`**: robust walk‑forward backtesting over moving windows.
- **`walk_forward_results.csv`**, **`f1_score_windows.png`**, **`F1.txt`**: aggregated results.
- **`Test_Set_-_window_*.txt`**: detailed per‑window classification reports.
- **`analyse.py`**: extra analysis/plots.

#### `MODELS/TCN/`
- **`TCN.py`**: Temporal Convolutional Network.
- **`best_hyperparameters (1).json`**, **`tcn_model.keras`**, **`Test_Set_report (1).txt`**.

#### `MODELS/TRANSFORMERS_GRU/`
- **`gru_transformers.py`**: Transformer‑style encoder with GRU head.
- **`best_transformer_gru_model.keras`**, **`transformer_gru_model.keras`**, **`Test_Set_report (11).txt`**.

#### `MODELS/XGBOOST/`
- **`XGBOOST.py`**: XGBoost baseline.
- **`best_xgboost_model.json`**, **`best_hyperparameters (8).json`**.
- **`BAR_PLOT.png`**, **`xgboost_feature_importance.png`**, **`xgboost_feature_importance_gain.csv`**, **`confusion_matrix_test_set (9).png`**, **`Test_Set_report (10).txt`**.


---

## 4) Where to Run

> Run in a TensorFlow/Keras environment (Google Collab is great). You can use GPUs.


---

5) Keras Tuning & Best Parameters

The best hyperparameters found by Keras Tuner are saved in each model folder (best_hyperparameters*.json).

The best checkpoints are saved under best_*_model.keras.

Test metrics and loss curves are exported (PNG/TXT/CSV).

To re-run training with the best hyperparameters, most scripts include a loading block; otherwise, copy the values from the JSON into the model definition.

7) License & Disclaimer

The models and scripts are provided for research purposes only. They do not constitute financial advice.