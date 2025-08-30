Part 2: Multi-Horizon Cryptocurrency Forecasting – Reproducibility Guide
This repository contains the code and resources for Part 2 of the thesis, which focuses on forecasting Bitcoin price movements across multiple time horizons (1h, 4h, 12h, 24h, and 48h) using a novel Crypto Temporal Fusion Transformer (CryptoTFT) model. The project includes training on historical data, validation across multiple splits, and evaluation in both quantitative and back-test trading settings.
Repository Contents
The project is organized into three main directories:

DATA/: Contains raw data for training and trading.
TRAINING/: Training and validation splits (five folds) in .csv and .pkl formats, with features and targets at daily (_D), 4-hour (_H4), and 1-hour (_H1) resolutions.
TRADING/: Test datasets for trading back-tests.


TRAINING/: Includes the training pipeline and model outputs.
Model_DL_PART2.py: Implements the CryptoTFT model, training routine for splits 1–4, and evaluation on split 5.
Subdirectories (SPLIT_1 to SPLIT_4): Store checkpoints, predictions, metrics, and plots for each trained split.
SPLIT_5_Evaluation: Contains test evaluation results, including metrics (split_5_test_metrics_by_model.csv), predictions (split_5_test_predictions_by_model.csv), and plots.


TRADING/: Contains the trading simulation script.
Code_Trading.py: Loads trained models, generates trading signals, executes trades with stop-loss/take-profit logic, and computes performance metrics.



For submission, compress the entire repository, including the thesis report (.tex or .docx), poster (.pptx), and any appendices. Ensure all files are named descriptively and free of corruption.
Getting Started
Prerequisites
The code requires Python 3.8+. We recommend using a virtual environment to manage dependencies. Install the required packages as follows:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn torch pytorch_lightning

Additional plotting dependencies (matplotlib, seaborn) are included in the above command.
Directory Structure
./
├── DATA/                    # Raw data for training and trading
│   ├── TRAINING/            # Training/validation splits
│   └── TRADING/             # Test data for trading back-test
├── TRAINING/                # Training pipeline and outputs
│   ├── Model_DL_PART2.py    # CryptoTFT model and training script
│   ├── SPLIT_1/ …           # Checkpoints, predictions, and plots for split 1
│   ├── …                    # Similar for splits 2–4
│   └── SPLIT_5_Evaluation/  # Test evaluation results
│       ├── split_5_test_metrics_by_model.csv
│       ├── split_5_test_predictions_by_model.csv
│       └── <plots>          # Loss curves and distribution plots
├── TRADING/                 # Trading simulation
│   └── Code_Trading.py      # Back-test trading script
└── README.md                # This file

Add your thesis report (.tex or .docx) and poster (.pptx) to the root directory before submission.
Reproducing the Training
The Model_DL_PART2.py script trains the CryptoTFT model on splits 1–4 and evaluates on split 5. To reproduce the training:

Verify Dataset Location:
The script expects training data in DATA/TRAINING/. Update base_dir and output_dir in Model_DL_PART2.py if your data or output paths differ.


Create Output Directories:
Ensure the TRAINING/ directory exists or modify output_dir to point to a valid location.


Run the Training Script:python TRAINING/Model_DL_PART2.py


This trains models on splits 1–4 and evaluates on split 5, saving checkpoints (.ckpt), predictions, metrics, and loss curve plots in TRAINING/.



Customizing Hyperparameters
The script uses default hyperparameters for learning rate, batch size, and maximum epochs. To experiment with different settings, modify the train_all_splits and evaluate_models_on_split5 function calls in Model_DL_PART2.py.
Reproducing the Trading Back-Test
The Code_Trading.py script simulates trading using trained models on a held-out test set. To run the back-test:

Ensure Test Data Availability:
Verify that btc_1d_test.csv, btc_h1_test.csv, and btc_h4_test.csv are in DATA/TRADING/. These files must include a timestamp column and the feature columns listed in the script.


Locate Checkpoints:
Place trained .ckpt files (e.g., TRAINING/SPLIT_1/split_1.ckpt) in a known directory. Update the load_models call in Code_Trading.py to point to the correct paths.


Run the Back-Test:python TRADING/Code_Trading.py --help


Review available command-line arguments for thresholds, take-profit, stop-loss, and other trading parameters. Example:

python TRADING/Code_Trading.py \
    --models TRAINING/SPLIT_1/split_1.ckpt TRAINING/SPLIT_2/split_2.ckpt \
    --datasets_dir DATA/TRADING \
    --thr 0.1 --tp_horizon 24h --confirmation_threshold 0.6


Outputs include a DataFrame of trading signals, trade summaries, equity curves, and diagnostic plots.



Data Overview
The DATA/ directory is divided into:

DATA/TRAINING/:
Contains five cross-validation folds (split_1 to split_5) in .csv and .pkl formats.
Each split includes training and validation sets at three resolutions:
*_D.csv: Daily samples with 30-day sequences.
*_H1.csv: 1-hour samples with 100-hour sequences.
*_H4.csv: 4-hour samples with 25-bar sequences.


.pkl files are serialized data for PyTorch Lightning (optional for CSV-based workflows).


DATA/TRADING/:
Contains test datasets for back-testing:
btc_1d_test.csv: Daily resolution.
btc_h1_test.csv: 1-hour resolution.
btc_h4_test.csv: 4-hour resolution.


Each file includes a timestamp column and features (OHLCV prices, RSI, MACD, Bollinger Bands, Ichimoku, etc.).
Training sets include target columns (Y_1h, Y_4h, Y_12h, Y_24h, Y_48h) for future returns.



Submission Notes

Report and Poster:
Include your thesis report (.tex or .docx, with figures and bibliography) and poster (.pptx) in the root directory before submission.


Reproducibility:
Avoid including large simulation outputs unless required. The provided checkpoints, scripts, and data are sufficient to regenerate results.


File Naming:
Use descriptive file names. For directories with many files, include a README.md (e.g., DATA/README.md, TRAINING/README.md) to describe contents.


Questions:
Refer to subdirectory README.md files for additional details or troubleshooting.


