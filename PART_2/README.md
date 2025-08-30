# Part 2: Multi-Horizon Cryptocurrency Forecasting – Reproducibility Guide

This repository contains the code and resources for Part 2 of the thesis, which focuses on forecasting Bitcoin price movements across multiple time horizons (1h, 4h, 12h, 24h, and 48h) using a novel Crypto Temporal Fusion Transformer (CryptoTFT) model. The project includes training on historical data, validation across multiple splits, and evaluation in both quantitative and back-test trading settings.

---

## Repository Contents

The project is organized into three main directories:

- **DATA/**: Contains raw data for training and trading.  
  - `TRAINING/`: Training and validation splits (five folds) in `.csv` and `.pkl` formats, with features and targets at daily (`_D`), 4-hour (`_H4`), and 1-hour (`_H1`) resolutions.  
  - `TRADING/`: Test datasets for trading back-tests.  

- **TRAINING/**: Includes the training pipeline and model outputs.  
  - `Model_DL_PART2.py`: Implements the CryptoTFT model, training routine for splits 1–4, and evaluation on split 5.  
  - Subdirectories (`SPLIT_1` to `SPLIT_4`): Store checkpoints, predictions, metrics, and plots for each trained split.  
  - `SPLIT_5_Evaluation/`: Contains test evaluation results, including metrics, predictions, and plots.  

- **TRADING/**: Contains the trading simulation script.  
  - `Code_Trading.py`: Loads trained models, generates trading signals, executes trades with stop-loss/take-profit logic, and computes performance metrics.  

---

## Getting Started

### Prerequisites

The code requires **Python 3.8+**. We recommend using a virtual environment to manage dependencies. We used GPUs on Google Collab.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn torch pytorch_lightning
```

Additional plotting dependencies (`matplotlib`, `seaborn`) are included in the above command.

---

## Directory Structure

```
./
├── DATA/                    
│   ├── TRAINING/
      ├── CODE       #all the data needed to train the model  (splits)  
│   └── TRADING/            
├── TRAINING/                
│   ├── Model_DL_PART2.py    
│   ├── SPLIT_1/ …           
│   ├── …                    
│   └── SPLIT_5_Evaluation/  
│       ├── split_5_test_metrics_by_model.csv
│       ├── split_5_test_predictions_by_model.csv
│       └── <plots>          
├── TRADING/                 
│   └── Code_Trading.py      
└── README.md                
```

---

## Reproducing the Training

Run the training script:

```bash
python TRAINING/Model_DL_PART2.py
```

This will train models on splits 1–4 and evaluate on split 5, saving:  
- Checkpoints (`.ckpt`)  
- Predictions  
- Metrics  
- Loss curve plots  

Hyperparameters (learning rate, batch size, max epochs) can be modified in `Model_DL_PART2.py`.

---

## Reproducing the Trading Back-Test

Run the back-test script:

```bash
python TRADING/Code_Trading.py --help
```

Example:

```bash
python TRADING/Code_Trading.py     --models TRAINING/SPLIT_1/split_1.ckpt TRAINING/SPLIT_2/split_2.ckpt     --datasets_dir DATA/TRADING     --thr 0.1 --tp_horizon 24h --confirmation_threshold 0.6
```

Outputs include:  
- Trading signals  
- Trade summaries  
- Equity curves  
- Diagnostic plots  

---

## Data Overview

- **DATA/TRAINING/**:  
  - Five cross-validation folds (split_1 to split_5).  
  - Resolutions: `_D.csv` (daily), `_H1.csv` (1h), `_H4.csv` (4h).  
  - Includes target columns: `Y_1h, Y_4h, Y_12h, Y_24h, Y_48h`.  

- **DATA/TRADING/**:  
  - Test datasets: `btc_1d_test.csv`, `btc_h1_test.csv`, `btc_h4_test.csv`.  
  - Includes features (OHLCV, RSI, MACD, Bollinger Bands, Ichimoku, etc.).  

---

---

## Questions

Refer to subdirectory `README.md` files for more details or ask : 

rodolphe lucas : rodolphe.lucas@cranfield.ac.uk
