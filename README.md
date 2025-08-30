# A Foundational Deep Learning Approach to Cryptocurrency Market Forecasting

Reproducible crypto time-series project. 

Part 1: Keras baselines (FNN, LSTM, GRU, CNN-LSTM, TCN, Transformer-GRU, XGBoost) with Keras Tuner + rolling walk-forward. 

Part 2: multi-timeframe "CryptoTFT" (H1/H4/D1), split-based training, evaluation, and trading back-tests with risk management.

# Technical Work — MSc Thesis Project of RODOLPHE Lucas 462976

This repository contains the **technical deliverables** for the Cranfield MSc Thesis *"A Foundational Deep Learning Approach to Cryptocurrency Market Forecasting"*.


It is organised in two main parts:

---

## 📂 Repository Structure

```
TECHNICAL_WORK/
├── Thesis report
├── Poster             
│
├── PART_1/              # Baseline models (Keras/TensorFlow)
│   ├── DATA/            # Training dataset(s) for models
│   └── MODELS/          # Model definitions, tuning artefacts, checkpoints & reports
│       ├── CNN+LSTM/
│       ├── FNN/
│       ├── GRU_LSTM/
│       ├── LSTM/
│       ├── ROLLING_WALK_FORWARD/
│       ├── TCN/
│       ├── TRANSFORMERS_GRU/
│       └── XGBOOST/
│
└── PART_2/              # CryptoTFT model (PyTorch Lightning)
    ├── DATA/
    │   ├── TRADING/     # Test datasets for backtests
    │   └── TRAINING/    # Training/validation splits
    │       └── CODE/    # Preprocessing / data preparation scripts
    ├── TRADING/         # Trading backtest simulator
    └── TRAINING/        # Training pipeline + model outputs
        ├── SPLIT_1/
        ├── SPLIT_2/
        ├── SPLIT_3/
        ├── SPLIT_4/
        └── SPLIT_5_Evaluation/
```

---

## 📝 Documentation

- Each major folder (**PART_1**, **PART_2**, `DATA/`, `TRAINING/`, `TRADING/`, …) contains its own **README.md** with **detailed explanations** about:
  - Contents of the folder,
  - How to run the scripts,
  - Description of datasets, models, and outputs,
  - Reproducibility notes.

- **PART_1/** covers **baseline models** (FNN, LSTM, GRU+LSTM, CNN+LSTM, TCN, Transformer-GRU, XGBoost) trained with **Keras**.  
- **PART_2/** contains the **CryptoTFT** multi-horizon, multi-timeframe model trained with **PyTorch Lightning**, plus a trading simulator.

  
<img width="598" height="606" alt="image" src="https://github.com/user-attachments/assets/e2f01101-a0c0-4ec5-9fbb-58c142214d6d" />

---

## 📌 Reproducibility

- Datasets and code are provided for **end-to-end reproducibility**.  
- For each part:
  - **Training code** and **tuned hyperparameters** are included,
  - **Checkpoints** (`.keras`, `.ckpt`) are not provided but you can create them with the code given,
  - **Reports** and **plots** (loss curves, confusion matrices, equity curves, metrics) are stored alongside.  

---


## 📬 Contact

For any questions or clarifications, please contact:  
📧 **rodolphe.lucas@cranfield.ac.uk**

## 📌 Trading Simulator
<img width="2240" height="960" alt="Volatility_P90" src="https://github.com/user-attachments/assets/51b40709-30be-436f-ac73-951f98860c8b" />
