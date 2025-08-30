# TRAINING Folder

This folder contains the model definition, training routines, and all artefacts produced by training the **CryptoTFT** model on the cross‑validation splits.  

---

## Files

- **Model_DL_PART2.py**  
  Main script defining the CryptoTFT model architecture and implementing the training/evaluation pipeline.  
  - Trains models on splits 1–4  
  - Evaluates them on split 5  

## Model Architecture Summary

The CryptoTFT model is composed of several blocks and projection heads, as summarised below:

| Name         | Type               | Params | Mode  |
|--------------|--------------------|--------|-------|
| tft_h1       | SimpleTFTBlock     | 1.2 M  | train |
| tft_h4       | SimpleTFTBlock     | 167 K  | train |
| tft_d        | SimpleTFTBlock     | 167 K  | train |
| agg_h1       | TemporalAggregator | 256    | train |
| agg_h4       | TemporalAggregator | 128    | train |
| agg_d        | TemporalAggregator | 128    | train |
| proj_h1      | Linear             | 262 K  | train |
| proj_h4      | Linear             | 131 K  | train |
| proj_d       | Linear             | 131 K  | train |
| gate         | WeightedSumGate    | 98.8 K | train |
| shared       | Sequential         | 98.7 K | train |
| heads        | ModuleList         | 20.8 K | train |
| mse_loss     | MSELoss            | 0      | train |
| mae_loss     | L1Loss             | 0      | train |
| other params | n/a                | 5      | n/a   |

This modular design allows for per‑timeframe processing (H1, H4, Daily), temporal aggregation, gated fusion, and horizon‑specific output heads.
---

## Subdirectories

### SPLIT_1 … SPLIT_4

Each split folder contains the outputs of training a model on the corresponding fold:

- `loss_curves_split_*.png`: Training and validation loss over epochs.  
- `split_*.ckpt`: PyTorch Lightning checkpoint for the trained model (load with `LightningModule.load_from_checkpoint`).  
- `split_*_predictions.csv`: Model predictions on its validation set.  
- `split_*_targets.csv`: Ground‑truth targets on the validation set.  

➡️ These outputs are generated automatically by running `Model_DL_PART2.py`. Do **not** modify them manually.

---

### SPLIT_5_Evaluation

After training models on splits 1–4, the script evaluates them on split 5 and aggregates their predictions.  
This folder contains:

- `split_5_test_metrics_by_model.csv`: Performance metrics (MSE, etc.) for each model on each horizon.  
- `split_5_test_predictions_by_model.csv`: Raw predictions for each model on the test set.  
- `.png` figures: Histograms, scatter plots, and MSE curves across horizons.  
- `Code_To_Plot_Results.py`: Helper script to regenerate the figures.

---

## How to Train

1. Ensure that training data are located in `../DATA/TRAINING`.  
   - If stored elsewhere, edit `base_dir` in `Model_DL_PART2.py`.  

2. Prepare an output directory for checkpoints and plots.  
   - Default: `TRAINING/` (or `/content/output/` if running in Colab).  
   - Can be changed in the script via `output_dir`.  

3. From the project root, run:

```bash
python TRAINING/Model_DL_PART2.py
```

The script will:  
- Train four models on splits 1–4  
- Save checkpoints and plots into `SPLIT_1 … SPLIT_4`  
- Evaluate models on split 5 and save results in `SPLIT_5_Evaluation`  

Once training is complete, you can inspect checkpoints and metrics, or load them into `TRADING/Code_Trading.py` for trading simulations.

---

## Notes

- The script uses **early stopping** and **checkpointing** via PyTorch Lightning.  
- Hyperparameters (learning rate, batch size, epochs) can be changed in `Model_DL_PART2.py` (`train_all_splits` and `evaluate_models_on_split5`).  
- The variable `use_uncertainty` controls whether heteroscedastic loss (Kendall’s uncertainty weighting) is used.  
  - `use_uncertainty=False`: standard MSE loss.  
