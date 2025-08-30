"""
CODE of PART2
Work of Rodolphe Lucas 462976
Proposition of a new approach for Forecasting Time Series
"""
from __future__ import annotations
import os
import math
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


##################################
###### NEURAL NETWORK COMPONENTS ##
##################################

class AttnPool1d(nn.Module):
    """
    Simple attention pooling module with a learned context vector.
    Purpose: Captures the relative importance of time steps in a sequence.
    """
    def __init__(self, hidden_size: int):
        """
        Initialize the attention pooling layer.
        
        Args:
            hidden_size (int): Size of the hidden dimension.
        """
        super().__init__()
        self.context = nn.Parameter(torch.randn(hidden_size))  # Learnable context vector
        self.tanh = nn.Tanh()  # Activation for score computation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attention pooling.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
        
        Returns:
            torch.Tensor: Pooled output of shape [batch_size, hidden_size].
        """
        B, T, H = x.shape
        ctx = self.context.view(1, 1, H).expand(B, T, H)  # Expand context vector
        scores = torch.sum(self.tanh(x) * ctx, dim=-1) / math.sqrt(H)  # Compute attention scores
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # Normalize scores to weights
        return torch.sum(w * x, dim=1)  # Weighted sum of input sequence

class SimpleTFTBlock(nn.Module):
    """
    Lightweight LSTM-based encoder with Gated Linear Unit (GLU).
    Purpose: Processes sequential data while preserving temporal information with low dropout.
    """
    def __init__(self, input_size: int, hidden_size: int, lstm_layers: int = 1, dropout: float = 0.15):
        """
        Initialize the pseudo TFT block.
        
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Size of the hidden state.
            lstm_layers (int): Number of LSTM layers (default: 1).
            dropout (float): Dropout rate (default: 0.15).
        """
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)  # Project input to hidden size
        self.lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True)  # LSTM encoder
        self.glu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),  # Linear layer for GLU
            nn.GLU(dim=-1),  # Gated Linear Unit
        )
        self.dropout = nn.Dropout(dropout)  # Apply dropout
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)  # Orthogonal initialization for LSTM weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TFT block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_size].
        
        Returns:
            torch.Tensor: Encoded output of shape [batch_size, seq_len, hidden_size].
        """
        x = self.input_proj(x)  # Project input
        h, _ = self.lstm(x)  # Pass through LSTM
        h = self.glu(h)  # Apply GLU
        return self.dropout(h)  # Apply dropout

class TemporalAggregator(nn.Module):
    """
    Aggregates sequence data using last, mean, max, and attention pooling.
    Purpose: Captures comprehensive sequence information without losing key features.
    """
    def __init__(self, hidden_size: int):
        """
        Initialize the temporal aggregator.
        
        Args:
            hidden_size (int): Size of the input hidden dimension.
        """
        super().__init__()
        self.attn = AttnPool1d(hidden_size)  # Attention pooling module

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal aggregation.
        
        Args:
            seq (torch.Tensor): Input sequence of shape [batch_size, seq_len, hidden_size].
        
        Returns:
            torch.Tensor: Aggregated output of shape [batch_size, 4 * hidden_size].
        """
        last = seq[:, -1, :]  # Last time step
        mean = seq.mean(dim=1)  # Mean across time steps
        maxv = seq.max(dim=1).values  # Max across time steps
        attn = self.attn(seq)  # Attention-pooled output
        return torch.cat([last, mean, maxv, attn], dim=-1)  # Concatenate all aggregations

class WeightedSumGate(nn.Module):
    """
    Learned gating mechanism to fuse three embeddings (H1, H4, D).
    Purpose: Provides a stable and efficient alternative to multi-head attention for three tokens.
    """
    def __init__(self, emb_dim: int = 256, hidden: int = 128, n_tokens: int = 3, dropout: float = 0.1):
        """
        Initialize the weighted sum gate.
        
        Args:
            emb_dim (int): Embedding dimension (default: 256).
            hidden (int): Hidden layer size (default: 128).
            n_tokens (int): Number of input tokens (default: 3 for H1, H4, D).
            dropout (float): Dropout rate (default: 0.1).
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.net = nn.Sequential(
            nn.Linear(emb_dim * n_tokens, hidden),  # Project concatenated embeddings
            nn.ReLU(),  # Activation
            nn.Dropout(dropout),  # Dropout
            nn.Linear(hidden, n_tokens),  # Output weights for each token
        )
        self.drop = nn.Dropout(dropout)  # Final dropout

    def forward(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for weighted sum gating.
        
        Args:
            tokens (List[torch.Tensor]): List of embeddings, each of shape [batch_size, emb_dim].
        
        Returns:
            torch.Tensor: Fused embedding of shape [batch_size, emb_dim].
        """
        x = torch.cat(tokens, dim=-1)  # Concatenate tokens
        logits = self.net(x)  # Compute gating weights
        w = torch.softmax(logits, dim=-1)  # Normalize weights
        stacked = torch.stack(tokens, dim=1)  # Stack tokens
        fused = torch.sum(w.unsqueeze(-1) * stacked, dim=1)  # Weighted sum
        return self.drop(fused)  # Apply dropout

################################
###### MODEL ARCHITECTURE #####
################################

class CryptoTFTModel(pl.LightningModule):
    """
    Main Temporal Fusion Transformer model for cryptocurrency price prediction.
    Combines multi-timeframe encoders, temporal aggregators, gating, and multi-horizon heads.
    """
    def __init__(
        self,
        feature_cols: List[str],
        target_cols: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        use_uncertainty: bool = False,
        horizon_weights: Optional[List[float]] = None,
        dropout_mlp: float = 0.3,
    ) -> None:
        """
        Initialize the Crypto TFT model.
        
        Args:
            feature_cols (List[str]): List of input feature column names.
            target_cols (List[str]): List of target column names for prediction horizons.
            lr (float): Learning rate (default: 1e-3).
            weight_decay (float): Weight decay for AdamW optimizer (default: 1e-4).
            warmup_epochs (int): Number of warmup epochs for learning rate (default: 5).
            use_uncertainty (bool): Use uncertainty-based loss (Kendall's method) (default: False).
            horizon_weights (Optional[List[float]]): Weights for each horizon's loss (default: None).
            dropout_mlp (float): Dropout rate for MLP layers (default: 0.3).
        """
        super().__init__()
        self.save_hyperparameters()

        in_size = len(feature_cols)
        self.n_targets = len(target_cols)

        # Encoders for each timeframe
        self.tft_h1 = SimpleTFTBlock(input_size=in_size, hidden_size=256, lstm_layers=2, dropout=0.15)
        self.tft_h4 = SimpleTFTBlock(input_size=in_size, hidden_size=128, lstm_layers=1, dropout=0.15)
        self.tft_d = SimpleTFTBlock(input_size=in_size, hidden_size=128, lstm_layers=1, dropout=0.15)

        # Aggregators to reduce sequence outputs
        self.agg_h1 = TemporalAggregator(256)
        self.agg_h4 = TemporalAggregator(128)
        self.agg_d = TemporalAggregator(128)
        self.proj_h1 = nn.Linear(256 * 4, 256)  # Project aggregated H1 to 256
        self.proj_h4 = nn.Linear(128 * 4, 256)  # Project aggregated H4 to 256
        self.proj_d = nn.Linear(128 * 4, 256)   # Project aggregated D to 256

        # Gating to fuse timeframe embeddings
        self.gate = WeightedSumGate(emb_dim=256, hidden=128, n_tokens=3, dropout=0.1)

        # Shared trunk for feature processing
        self.shared = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_mlp),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_mlp),
        )

        # Prediction heads for each horizon
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)) for _ in range(self.n_targets)]
        )

        # Loss functions
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.mae_loss = nn.L1Loss(reduction="mean")

        # Horizon weights for loss aggregation
        if horizon_weights is None:
            horizon_weights = [1.0] * self.n_targets
        w = torch.tensor(horizon_weights, dtype=torch.float32)
        self.register_buffer("horizon_weights", w)

        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(self.n_targets))  # Learnable log-variances for uncertainty
        else:
            self.log_vars = None

        self.target_cols = target_cols  # Store target column names for logging

    def forward(self, x_h1: torch.Tensor, x_h4: torch.Tensor, x_d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x_h1 (torch.Tensor): H1 timeframe input [batch_size, seq_len_h1, input_size].
            x_h4 (torch.Tensor): H4 timeframe input [batch_size, seq_len_h4, input_size].
            x_d (torch.Tensor): Daily timeframe input [batch_size, seq_len_d, input_size].
        
        Returns:
            torch.Tensor: Predictions for all horizons [batch_size, n_targets].
        """
        h1_seq = self.tft_h1(x_h1)  # Encode H1 sequence
        h4_seq = self.tft_h4(x_h4)  # Encode H4 sequence
        d_seq = self.tft_d(x_d)     # Encode daily sequence

        h1 = self.proj_h1(self.agg_h1(h1_seq))  # Aggregate and project H1
        h4 = self.proj_h4(self.agg_h4(h4_seq))  # Aggregate and project H4
        d = self.proj_d(self.agg_d(d_seq))      # Aggregate and project daily

        fused = self.gate([h1, h4, d])          # Fuse timeframe embeddings
        z = self.shared(fused)                  # Process through shared trunk

        outs = [head(z) for head in self.heads]  # Generate predictions for each horizon
        y = torch.cat(outs, dim=-1)              # Concatenate predictions
        return y

    def _loss_per_horizon(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MSE and MAE losses for each prediction horizon.
        
        Args:
            y_pred (torch.Tensor): Predicted values [batch_size, n_targets].
            y_true (torch.Tensor): True values [batch_size, n_targets].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: MSE and MAE losses per horizon [n_targets].
        """
        mse_list, mae_list = [], []
        for i in range(self.n_targets):
            mse_list.append(self.mse_loss(y_pred[:, i], y_true[:, i]))
            mae_list.append(self.mae_loss(y_pred[:, i], y_true[:, i]))
        return torch.stack(mse_list), torch.stack(mae_list)

    def _aggregate_loss(self, mse_per_h: torch.Tensor) -> torch.Tensor:
        """
        Aggregate per-horizon MSE losses, optionally using uncertainty-based weighting.
        
        Args:
            mse_per_h (torch.Tensor): MSE losses for each horizon [n_targets].
        
        Returns:
            torch.Tensor: Aggregated loss scalar.
        """
        if self.use_uncertainty:
            var = torch.exp(self.log_vars)  # Convert log-variances to variances
            return torch.sum(mse_per_h / (2.0 * var) + 0.5 * self.log_vars)  # Kendall's uncertainty loss
        else:
            return torch.sum(self.horizon_weights * mse_per_h)  # Weighted sum of MSEs

    def _metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
        """
        Compute global metrics for predictions (MSE, MAE, RMSE, R2, MAPE, directional accuracy).
        
        Args:
            y_pred (torch.Tensor): Predicted values [batch_size, n_targets].
            y_true (torch.Tensor): True values [batch_size, n_targets].
        
        Returns:
            dict: Dictionary of metric values.
        """
        mse = self.mse_loss(y_pred, y_true)
        mae = self.mae_loss(y_pred, y_true)
        rmse = torch.sqrt(mse + 1e-12)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2) + 1e-8
        r2 = 1 - ss_res / ss_tot
        mape = torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + 1e-2))) * 100.0  #ERROR DO NOT TAKE INTO ACCOUNT
        directional = torch.mean((torch.sign(y_pred) == torch.sign(y_true)).float())
        return dict(mse=mse, mae=mae, rmse=rmse, r2=r2, mape=mape, directional=directional)

    def _metrics_per_horizon_full(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
        """
        Compute metrics for each prediction horizon.
        
        Args:
            y_pred (torch.Tensor): Predicted values [batch_size, n_targets].
            y_true (torch.Tensor): True values [batch_size, n_targets].
        
        Returns:
            dict: Metrics per horizon, keyed by horizon index.
        """
        eps = 1e-8
        out = {}
        for i in range(self.n_targets):
            y = y_true[:, i]
            p = y_pred[:, i]
            mse = torch.mean((p - y) ** 2)
            mae = torch.mean(torch.abs(p - y))
            rmse = torch.sqrt(mse + eps)
            var = torch.sum((y - y.mean()) ** 2) + eps
            r2 = 1.0 - torch.sum((y - p) ** 2) / var
            mape = torch.mean(torch.abs((y - p) / (torch.abs(y) + 1e-2))) * 100.0  # ERROR ON MAPE DO NOT TAKE INTO ACCOUNT
            directional = torch.mean((torch.sign(p) == torch.sign(y)).float())
            out[i] = {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "dir": directional}
        return out

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.
        
        Args:
            batch: Tuple of (inputs, targets) where inputs are (x_h1, x_h4, x_d).
            batch_idx: Index of the batch.
        
        Returns:
            torch.Tensor: Training loss.
        """
        (x_h1, x_h4, x_d), y = batch
        y_pred = self(x_h1, x_h4, x_d)
        mse_per_h, mae_per_h = self._loss_per_horizon(y_pred, y)
        loss = self._aggregate_loss(mse_per_h)

        metrics = self._metrics(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mse", metrics["mse"], on_epoch=True, prog_bar=True)
        self.log("train_mae", metrics["mae"], on_epoch=True, prog_bar=True)
        self.log("train_rmse", metrics["rmse"], on_epoch=True, prog_bar=False)
        self.log("train_r2", metrics["r2"], on_epoch=True, prog_bar=False)
        self.log("train_mape", metrics["mape"], on_epoch=True, prog_bar=False)
        self.log("train_directional", metrics["directional"], on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.
        
        Args:
            batch: Tuple of (inputs, targets) where inputs are (x_h1, x_h4, x_d).
            batch_idx: Index of the batch.
        """
        (x_h1, x_h4, x_d), y = batch
        y_pred = self(x_h1, x_h4, x_d)
        mse_per_h, mae_per_h = self._loss_per_horizon(y_pred, y)
        loss = self._aggregate_loss(mse_per_h)

        metrics = self._metrics(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_mse", metrics["mse"], prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_mae", metrics["mae"], prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_rmse", metrics["rmse"], prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_r2", metrics["r2"], prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_mape", metrics["mape"], prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_directional", metrics["directional"], prog_bar=False, on_epoch=True, on_step=False)

        # Log per-horizon metrics only for the first batch to avoid duplication
        if batch_idx == 0:
            perh = self._metrics_per_horizon_full(y_pred, y)
            for i, col in enumerate(self.target_cols):
                self.log(f"val_mse_{col}", perh[i]["mse"], prog_bar=False, on_epoch=True, on_step=False)
                self.log(f"val_mae_{col}", perh[i]["mae"], prog_bar=False, on_epoch=True, on_step=False)
                self.log(f"val_rmse_{col}", perh[i]["rmse"], prog_bar=False, on_epoch=True, on_step=False)
                self.log(f"val_r2_{col}", perh[i]["r2"], prog_bar=False, on_epoch=True, on_step=False)
                self.log(f"val_mape_{col}", perh[i]["mape"], prog_bar=False, on_epoch=True, on_step=False)
                self.log(f"val_dir_{col}", perh[i]["dir"], prog_bar=False, on_epoch=True, on_step=False)

        hit_ratio = torch.mean((torch.sign(y_pred) == torch.sign(y)).float())
        self.log("val_hit_ratio", hit_ratio, prog_bar=False)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        
        Returns:
            dict: Optimizer and scheduler configuration.
        """
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        total_epochs = self.trainer.max_epochs if self.trainer is not None else 100
        warmup_epochs = int(min(self.hparams.warmup_epochs, max(1, total_epochs - 1)))
        if warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
            cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_epochs - warmup_epochs))
            sched = optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_epochs))
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

################################
###### DATA HANDLING ##########
################################

class MultiTimeframeDataset(Dataset):
    """
    Dataset for multi-timeframe data (H1, H4, D) grouped by group_id.
    Assumes fixed sequence lengths: H1=100, H4=25, D=30.
    """
    def __init__(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame, df_d: pd.DataFrame,
                 feature_cols: List[str], target_cols: List[str]):
        """
        Initialize the dataset.
        
        Args:
            df_h1 (pd.DataFrame): H1 timeframe data.
            df_h4 (pd.DataFrame): H4 timeframe data.
            df_d (pd.DataFrame): Daily timeframe data.
            feature_cols (List[str]): List of feature column names.
            target_cols (List[str]): List of target column names.
        """
        self.groups = df_h1["group_id"].unique()
        self.df_h1 = df_h1
        self.df_h4 = df_h4
        self.df_d = df_d
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        for g in self.groups:
            assert len(df_h1[df_h1["group_id"] == g]) == 100, f"Invalid H1 sequence length for group {g}"
            assert len(df_h4[df_h4["group_id"] == g]) == 25, f"Invalid H4 sequence length for group {g}"
            assert len(df_d[df_d["group_id"] == g]) == 30, f"Invalid D sequence length for group {g}"

    def __len__(self) -> int:
        """Return the number of groups in the dataset."""
        return len(self.groups)

    def __getitem__(self, idx: int):
        """
        Retrieve a single data sample.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Tuple: (inputs, targets) where inputs are (x_h1, x_h4, x_d) tensors.
        """
        g = self.groups[idx]
        x_h1 = torch.tensor(self.df_h1[self.df_h1["group_id"] == g][self.feature_cols].values, dtype=torch.float32)
        x_h4 = torch.tensor(self.df_h4[self.df_h4["group_id"] == g][self.feature_cols].values, dtype=torch.float32)
        x_d = torch.tensor(self.df_d[self.df_d["group_id"] == g][self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(self.df_h1[self.df_h1["group_id"] == g][self.target_cols].iloc[-1].values, dtype=torch.float32)
        return (x_h1, x_h4, x_d), y

@torch.no_grad()
def predict_model(model: pl.LightningModule, val_loader: DataLoader, output_dir: str, split_id: int):
    """
    Generate predictions for a validation/test DataLoader and save to CSV.
    
    Args:
        model (pl.LightningModule): Trained model.
        val_loader (DataLoader): DataLoader for validation/test data.
        output_dir (str): Directory to save predictions and targets.
        split_id (int): Split identifier for file naming.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Predictions and true targets.
    """
    model.eval()
    predictions, targets = [], []
    device = model.device
    for batch in val_loader:
        (x_h1, x_h4, x_d), y = batch
        y_pred = model(x_h1.to(device), x_h4.to(device), x_d.to(device))
        predictions.append(y_pred.cpu().numpy())
        targets.append(y.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(predictions, columns=[
        "Y_1h_pred", "Y_4h_pred", "Y_12h_pred", "Y_24h_pred", "Y_48h_pred"
    ]).to_csv(f"{output_dir}/split_{split_id}_predictions.csv", index=False)
    pd.DataFrame(targets, columns=[
        "Y_1h", "Y_4h", "Y_12h", "Y_24h", "Y_48h"
    ]).to_csv(f"{output_dir}/split_{split_id}_targets.csv", index=False)
    return predictions, targets

################################
###### METRIC CALCULATIONS ##
################################

def _metrics_numpy(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Compute global metrics (MSE, MAE, RMSE, R2, MAPE, directional accuracy) using NumPy.
    
    Args:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.
    
    Returns:
        Dict[str, float]: Dictionary of metric values.
    """
    eps = 1e-8
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse + eps))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2) + eps)
    r2 = float(1.0 - ss_res / ss_tot)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-2))) * 100.0)
    directional = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return dict(mse=mse, mae=mae, rmse=rmse, r2=r2, mape=mape, directional=directional)

def _metrics_per_h_numpy(y_pred: np.ndarray, y_true: np.ndarray, cols: List[str]) -> Dict[str, float]:
    """
    Compute metrics for each prediction horizon using NumPy.
    
    Args:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.
        cols (List[str]): List of target column names.
    
    Returns:
        Dict[str, float]: Metrics per horizon, prefixed with metric type and column name.
    """
    eps = 1e-8
    out = {}
    for i, c in enumerate(cols):
        y = y_true[:, i]
        p = y_pred[:, i]
        mse = float(np.mean((p - y) ** 2))
        mae = float(np.mean(np.abs(p - y)))
        rmse = float(np.sqrt(mse + eps))
        var = float(np.sum((y - y.mean()) ** 2) + eps)
        r2 = float(1.0 - np.sum((y - p) ** 2) / var)
        mape = float(np.mean(np.abs((y - p) / (np.abs(y) + 1e-2))) * 100.0)
        directional = float(np.mean(np.sign(p) == np.sign(y)))
        out.update({
            f"mse_{c}": mse,
            f"mae_{c}": mae,
            f"rmse_{c}": rmse,
            f"r2_{c}": r2,
            f"mape_{c}": mape,
            f"dir_{c}": directional,
        })
    return out

def plot_loss_curves(csv_path: str, output_dir: str, split_id: int):
    """
    Plot and save training and validation loss curves.
    
    Args:
        csv_path (str): Path to the metrics CSV file.
        output_dir (str): Directory to save the plot.
        split_id (int): Split identifier for file naming.
    """
    metrics = pd.read_csv(csv_path)
    train_loss = metrics['train_loss_epoch'].dropna()
    val_loss = metrics['val_loss'].dropna()
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves for Split {split_id}')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f'loss_curves_split_{split_id}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve to {save_path}")


################################
### TRAINING AND EVALUATION ##
################################


def train_all_splits(base_dir: str, output_dir: str, feature_cols: List[str], target_cols: List[str],
                     use_uncertainty: bool = False,
                     horizon_weights: Optional[List[float]] = None,
                     max_epochs: int = 200,
                     batch_size: int = 32) -> None:
    """
    Train models on splits 1 to 4 and save results.
    
    Args:
        base_dir (str): Directory containing input CSV files.
        output_dir (str): Directory to save outputs (checkpoints, metrics, plots).
        feature_cols (List[str]): List of feature column names.
        target_cols (List[str]): List of target column names.
        use_uncertainty (bool): Use uncertainty-based loss (default: False).
        horizon_weights (Optional[List[float]]): Weights for each horizon (default: None).
        max_epochs (int): Maximum number of training epochs (default: 100).
        batch_size (int): Batch size for training (default: 32).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for split_id in range(1, 5):
        print(f"Training split {split_id}...")
        # Load data for the current split
        train_h1 = pd.read_csv(f"{base_dir}/split_{split_id}_train_H1.csv")
        train_h4 = pd.read_csv(f"{base_dir}/split_{split_id}_train_H4.csv")
        train_d = pd.read_csv(f"{base_dir}/split_{split_id}_train_D.csv")
        val_h1 = pd.read_csv(f"{base_dir}/split_{split_id}_val_H1.csv")
        val_h4 = pd.read_csv(f"{base_dir}/split_{split_id}_val_H4.csv")
        val_d = pd.read_csv(f"{base_dir}/split_{split_id}_val_D.csv")

        # Create datasets and data loaders
        train_ds = MultiTimeframeDataset(train_h1, train_h4, train_d, feature_cols, target_cols)
        val_ds = MultiTimeframeDataset(val_h1, val_h4, val_d, feature_cols, target_cols)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

        # Initialize model
        model = CryptoTFTModel(
            feature_cols=feature_cols,
            target_cols=target_cols,
            lr=1e-3,
            weight_decay=1e-4,
            warmup_epochs=5,
            use_uncertainty=use_uncertainty,
            horizon_weights=horizon_weights,
            dropout_mlp=0.3,
        )

        # Configure callbacks
        ckpt_cb = ModelCheckpoint(
            monitor="val_mse",
            mode="min",
            filename=f"split{split_id}-best-{{epoch:02d}}-{{val_mse:.4f}}",
            save_top_k=1,
        )
        early = EarlyStopping(
            monitor="val_loss" if use_uncertainty else "val_mse",
            mode="min",
            patience=120,
            min_delta=1e-6,
            strict=False,
        )
        csv_logger = CSVLogger(
            save_dir=f"{output_dir}/split_{split_id}",
            name="csv_logs"
        )

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs if max_epochs is not None else 300,
            min_epochs=150,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[early, ckpt_cb],
            default_root_dir=f"{output_dir}/split_{split_id}",
            logger=csv_logger,
            log_every_n_steps=8,
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Plot and save loss curves
        metrics_csv = os.path.join(csv_logger.log_dir, "metrics.csv")
        if not os.path.exists(metrics_csv) and hasattr(csv_logger, "experiment"):
            try:
                metrics_csv = csv_logger.experiment.metrics_file_path
            except Exception:
                pass
        plot_loss_curves(metrics_csv, output_dir, split_id)

        # Load best model for predictions
        best_path = ckpt_cb.best_model_path
        best_model = CryptoTFTModel.load_from_checkpoint(
            best_path,
            feature_cols=feature_cols,
            target_cols=target_cols,
            lr=1e-3,
            weight_decay=1e-4,
            warmup_epochs=5,
            use_uncertainty=use_uncertainty,
            horizon_weights=horizon_weights,
            dropout_mlp=0.3,
        )

        # Generate and save predictions
        predictions, targets = predict_model(best_model, val_loader, output_dir, split_id)

        # Collect and log metrics
        metrics = trainer.callback_metrics
        row = {
            "split_id": split_id,
            "val_mse": float(metrics.get("val_mse", torch.tensor(0.0)).item()),
            "val_mae": float(metrics.get("val_mae", torch.tensor(0.0)).item()),
            "val_mape": float(metrics.get("val_mape", torch.tensor(0.0)).item()),
            "val_directional": float(metrics.get("val_directional", torch.tensor(0.0)).item()),
            "val_rmse": float(metrics.get("val_rmse", torch.tensor(0.0)).item()),
            "val_r2": float(metrics.get("val_r2", torch.tensor(0.0)).item()),
            "val_hit_ratio": float(metrics.get("val_hit_ratio", torch.tensor(0.0)).item()),
        }
        for col in target_cols:
            for m in ["mse", "mae", "rmse", "r2", "mape", "dir"]:
                key = f"val_{m}_{col}"
                if key in metrics:
                    row[key] = float(metrics[key].item())
        results.append(row)

        print(f"Results for Split {split_id}:")
        for k, v in row.items():
            if k != "split_id":
                print(f"{k}: {v:.4f}")

    # Save aggregated results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/all_splits_results.csv", index=False)
    print(f"Average results across all splits:")
    print(results_df.mean(numeric_only=True))

def _temporal_inner_split(df, group_col="group_id", val_ratio=0.1):
    """
    Split a DataFrame temporally into training and validation sets based on group_id.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        group_col (str): Column name for grouping (default: "group_id").
        val_ratio (float): Fraction of groups for validation (default: 0.1).
    
    Returns:
        Callable: Function to split a DataFrame into training and validation sets.
    """
    groups = np.sort(df[group_col].unique())
    n = len(groups)
    n_val = max(1, int(round(n * val_ratio)))
    train_groups = groups[: n - n_val]
    val_groups = groups[n - n_val:]

    def split_one(d):
        tr = d[d[group_col].isin(train_groups)].copy()
        va = d[d[group_col].isin(val_groups)].copy()
        return tr, va

    return split_one

def train_and_test_split5(base_dir, output_dir, feature_cols, target_cols,
                          use_uncertainty=False, horizon_weights=None,
                          max_epochs=200, batch_size=32, inner_val_ratio=0.1):
    """
    Train a model on split 5 training data with early stopping on an inner validation set,
    then evaluate on the split 5 test set.
    
    Args:
        base_dir (str): Directory containing input CSV files.
        output_dir (str): Directory to save outputs.
        feature_cols (List[str]): List of feature column names.
        target_cols (List[str]): List of target column names.
        use_uncertainty (bool): Use uncertainty-based loss (default: False).
        horizon_weights (Optional[List[float]]): Weights for each horizon (default: None).
        max_epochs (int): Maximum number of training epochs (default: 200).
        batch_size (int): Batch size for training (default: 32).
        inner_val_ratio (float): Fraction of training data for inner validation (default: 0.1).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Predictions and true targets for the test set.
    """
    # Load split 5 data
    train_h1 = pd.read_csv(f"{base_dir}/split_5_train_H1.csv")
    train_h4 = pd.read_csv(f"{base_dir}/split_5_train_H4.csv")
    train_d = pd.read_csv(f"{base_dir}/split_5_train_D.csv")
    test_h1 = pd.read_csv(f"{base_dir}/split_5_val_H1.csv")
    test_h4 = pd.read_csv(f"{base_dir}/split_5_val_H4.csv")
    test_d = pd.read_csv(f"{base_dir}/split_5_val_D.csv")

    # Create inner training and validation splits
    split_one = _temporal_inner_split(train_h1, group_col="group_id", val_ratio=inner_val_ratio)
    train_h1_tr, train_h1_va = split_one(train_h1)
    split_one_h4 = _temporal_inner_split(train_h4, val_ratio=inner_val_ratio)
    train_h4_tr, train_h4_va = split_one_h4(train_h4)
    split_one_d = _temporal_inner_split(train_d, val_ratio=inner_val_ratio)
    train_d_tr, train_d_va = split_one_d(train_d)

    # Create datasets and data loaders
    train_ds = MultiTimeframeDataset(train_h1_tr, train_h4_tr, train_d_tr, feature_cols, target_cols)
    val_ds = MultiTimeframeDataset(train_h1_va, train_h4_va, train_d_va, feature_cols, target_cols)
    test_ds = MultiTimeframeDataset(test_h1, test_h4, test_d, feature_cols, target_cols)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # Initialize model
    model = CryptoTFTModel(
        feature_cols=feature_cols,
        target_cols=target_cols,
        lr=1e-3, weight_decay=1e-4, warmup_epochs=5,
        use_uncertainty=use_uncertainty, horizon_weights=horizon_weights, dropout_mlp=0.3,
    )

    # Configure callbacks
    ckpt_cb = ModelCheckpoint(monitor="val_mse", mode="min",
                              filename="split5-inner-best-{epoch:02d}-{val_mse:.4f}",
                              save_top_k=1)
    early = EarlyStopping(
        monitor="val_loss" if use_uncertainty else "val_mse",
        mode="min",
        patience=120,
        min_delta=1e-6,
        strict=False
    )
    logger = CSVLogger(save_dir=f"{output_dir}/split_5_test_run", name="csv_logs")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=300,
        min_epochs=150,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early, ckpt_cb],
        default_root_dir=f"{output_dir}/split_5_test_run",
        logger=logger,
        log_every_n_steps=8,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Plot and save loss curves
    metrics_csv = os.path.join(logger.experiment.metrics_file_path)
    plot_loss_curves(metrics_csv, output_dir, 5)

    # Load best model for predictions
    best_path = ckpt_cb.best_model_path
    best_model = CryptoTFTModel.load_from_checkpoint(
        best_path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        lr=1e-3, weight_decay=1e-4, warmup_epochs=5,
        use_uncertainty=use_uncertainty, horizon_weights=horizon_weights, dropout_mlp=0.3,
    )

    # Evaluate on test set
    preds, targs = predict_model(best_model, test_loader, output_dir, split_id=5)

    # Compute and save test metrics
    test_global = _metrics_numpy(preds, targs)
    test_per_h = _metrics_per_h_numpy(preds, targs, target_cols)
    metrics_out = {**test_global}
    for k, v in test_per_h.items():
        metrics_out[f"test_{k}"] = v
    pd.DataFrame([metrics_out]).to_csv(os.path.join(output_dir, "split_5_test_metrics.csv"), index=False)

    return preds, targs

################################
###### ENSEMBLE EVALUATION ####
################################

import glob

def _find_best_ckpt(output_dir: str, split_id: int) -> str:
    """
    Find the best checkpoint file for a given split.
    
    Args:
        output_dir (str): Directory containing checkpoints.
        split_id (int): Split identifier.
    
    Returns:
        str: Path to the best checkpoint file.
    
    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    pattern = os.path.join(output_dir, f"split_{split_id}", "**", "checkpoints", f"split{split_id}-best-*.ckpt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No checkpoint found for split {split_id} in {output_dir}.")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def evaluate_models_on_split5(
    base_dir: str,
    output_dir: str,
    feature_cols: List[str],
    target_cols: List[str],
    use_uncertainty: bool,
    horizon_weights: Optional[List[float]],
    batch_size: int = 32,
) -> None:
    """
    Evaluate models from splits 1-4 on the split 5 test set and compute ensemble predictions.
    Saves per-model and ensemble metrics, predictions, and distribution histograms.
    
    Args:
        base_dir (str): Directory containing input CSV files.
        output_dir (str): Directory to save outputs.
        feature_cols (List[str]): List of feature column names.
        target_cols (List[str]): List of target column names.
        use_uncertainty (bool): Use uncertainty-based loss.
        horizon_weights (Optional[List[float]]): Weights for each horizon.
        batch_size (int): Batch size for evaluation (default: 32).
    """
    # Load split 5 test data
    test_h1 = pd.read_csv(f"{base_dir}/split_5_val_H1.csv")
    test_h4 = pd.read_csv(f"{base_dir}/split_5_val_H4.csv")
    test_d = pd.read_csv(f"{base_dir}/split_5_val_D.csv")

    test_ds = MultiTimeframeDataset(test_h1, test_h4, test_d, feature_cols, target_cols)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    preds_list = []
    model_names = []

    # Load validation metrics for weighting (inverse MSE)
    results_csv = os.path.join(output_dir, "all_splits_results.csv")
    val_weights = None
    if os.path.exists(results_csv):
        df_res = pd.read_csv(results_csv)
        df_res = df_res[df_res["split_id"].isin([1, 2, 3, 4])]
        if len(df_res) == 4:
            inv = 1.0 / (df_res.set_index("split_id")["val_mse"] + 1e-8)
            w = inv / inv.sum()
            val_weights = np.array([w.loc[1], w.loc[2], w.loc[3], w.loc[4]], dtype=float)

    # Evaluate models from splits 1-4
    for sid in range(1, 5):
        ckpt = _find_best_ckpt(output_dir, sid)
        model = CryptoTFTModel.load_from_checkpoint(
            ckpt,
            feature_cols=feature_cols,
            target_cols=target_cols,
            lr=1e-3, weight_decay=1e-4, warmup_epochs=5,
            use_uncertainty=use_uncertainty, horizon_weights=horizon_weights, dropout_mlp=0.3,
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        preds, targs = [], []
        with torch.no_grad():
            for (x_h1, x_h4, x_d), y in test_loader:
                p = model(x_h1.to(device), x_h4.to(device), x_d.to(device)).cpu().numpy()
                preds.append(p)
                targs.append(y.numpy())
        P = np.concatenate(preds, axis=0)
        Y = np.concatenate(targs, axis=0)

        preds_list.append(P)
        model_names.append(f"split_{sid}")

    # Compute ensemble predictions
    stack = np.stack(preds_list, axis=0)  # [4, N, H]
    ens_mean = stack.mean(axis=0)
    ens_median = np.median(stack, axis=0)
    if val_weights is None:
        val_weights = np.ones(4, dtype=float) / 4.0
    ens_weighted = np.tensordot(val_weights, stack, axes=(0, 0))  # [N, H]

    # Compute metrics for individual models and ensembles
    rows = []
    for name, P in zip(model_names, preds_list):
        row = {"model": name, **_metrics_numpy(P, Y)}
        row.update({f"{k}": v for k, v in _metrics_per_h_numpy(P, Y, target_cols).items()})
        rows.append(row)

    for name, P in [
        ("ensemble_mean", ens_mean),
        ("ensemble_median", ens_median),
        ("ensemble_val_weighted", ens_weighted),
    ]:
        row = {"model": name, **_metrics_numpy(P, Y)}
        row.update({f"{k}": v for k, v in _metrics_per_h_numpy(P, Y, target_cols).items()})
        rows.append(row)

    # Save metrics
    met_path = os.path.join(output_dir, "split_5_test_metrics_by_model.csv")
    pd.DataFrame(rows).to_csv(met_path, index=False)

    # Save predictions in wide format
    out = {c: Y[:, i] for i, c in enumerate(target_cols)}
    for name, P in zip(model_names, preds_list):
        for i, c in enumerate(target_cols):
            out[f"{c}_pred_{name}"] = P[:, i]
    for i, c in enumerate(target_cols):
        out[f"{c}_pred_ens_mean"] = ens_mean[:, i]
        out[f"{c}_pred_ens_median"] = ens_median[:, i]
        out[f"{c}_pred_ens_weighted"] = ens_weighted[:, i]

    pred_wide = pd.DataFrame(out)
    pred_wide.to_csv(os.path.join(output_dir, "split_5_test_predictions_by_model.csv"), index=False)

    # Generate and save distribution histograms
    for i, col in enumerate(target_cols):
        plt.figure(figsize=(10, 6))
        plt.hist(Y[:, i], bins=50, alpha=0.5, label='True')
        plt.hist(ens_mean[:, i], bins=50, alpha=0.5, label='Predicted (Ensemble Mean)')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution for {col}: True vs Predicted')
        plt.legend()
        plt.grid(True)
        hist_path = os.path.join(output_dir, f'hist_distribution_{col}.png')
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved histogram to {hist_path}")

    print(f"\n[OK] Evaluation of 4 models on split 5 completed.\n- Metrics: {met_path}\n- Predictions: split_5_test_predictions_by_model.csv")

#####################
### MAIN EXECUTION ##
######################

if __name__ == "__main__":
    """
    Main execution block for training and evaluating the Crypto TFT model.
    Trains models on splits 1-4, evaluates on split 5, and computes ensemble predictions.
    """
    # Ensure reproducibility
    pl.seed_everything(42, workers=True)

    # Define paths and columns
    base_dir = "/content/datasets_npz_24_large"
    output_dir = "/content/output/"
    feature_cols = [
        "open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_signal",
        "macd_histogram", "macd_divergence", "macd_slope", "tenkan_sen", "kijun_sen",
        "senkou_span_a", "senkou_span_b", "bollinger_mavg", "bollinger_hband",
        "bollinger_lband", "bollinger_width",
    ]
    target_cols = ["Y_1h", "Y_4h", "Y_12h", "Y_24h", "Y_48h"]

    # Configuration
    use_uncertainty = True  # Enable Kendall's uncertainty-based loss
    horizon_weights = None  # Use default equal weights for horizons

    # Phase A: Train and log splits 1-4
    train_all_splits(
        base_dir=base_dir,
        output_dir=output_dir,
        feature_cols=feature_cols,
        target_cols=target_cols,
        use_uncertainty=use_uncertainty,
        horizon_weights=horizon_weights,
        max_epochs=200,
        batch_size=32,
    )

    # Phase B: Evaluate models on split 5 and compute ensembles
    evaluate_models_on_split5(
        base_dir=base_dir,
        output_dir=output_dir,
        feature_cols=feature_cols,
        target_cols=target_cols,
        use_uncertainty=use_uncertainty,
        horizon_weights=horizon_weights,
        batch_size=32,
    )