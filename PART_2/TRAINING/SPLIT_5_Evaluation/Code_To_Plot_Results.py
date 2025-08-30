import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data (adjust paths if necessary)
metrics_df = pd.read_csv('split_5_test_metrics_by_model (5).csv')
preds_df = pd.read_csv('split_5_test_predictions_by_model (4).csv')

# Configure plot style
plt.style.use('ggplot')  # Alternative style if seaborn is not installed
sns.set_palette("husl")  # Modern color palette (requires seaborn)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True  # Add grid for better readability

# 1. Graphique des métriques par horizon (example MSE)
horizons = ['Y_1h', 'Y_4h', 'Y_12h', 'Y_24h', 'Y_48h']
mse_horizon = [f'mse_{h}' for h in horizons]
fig, ax = plt.subplots(figsize=(12, 6))
metrics_df_melted = metrics_df.melt(id_vars=['model'], value_vars=mse_horizon,
                                    var_name='Horizon', value_name='MSE')
sns.barplot(x='Horizon', y='MSE', hue='model', data=metrics_df_melted)
ax.set_title('MSE by Horizon and Model', fontsize=14, pad=15)
ax.set_xlabel('Horizon', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
for i, v in enumerate(metrics_df_melted['MSE']):
    ax.text(i % 5, v, f'{v:.6f}', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=45)
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('mse_by_horizon_plot.png', bbox_inches='tight')
plt.close()

# 2. Predictions vs True Values for all horizons
horizons = ['Y_1h', 'Y_4h', 'Y_12h', 'Y_24h', 'Y_48h']
for horizon in horizons:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(preds_df[horizon], preds_df[f'{horizon}_pred_ens_mean'], alpha=0.5, label='Ensemble Mean', s=50, edgecolor='w', linewidth=0.5)
    ax.plot([preds_df[horizon].min(), preds_df[horizon].max()], [preds_df[horizon].min(), preds_df[horizon].max()], 'r--', label='Perfect Prediction', linewidth=1.5)
    ax.set_xlabel(f'True Values ({horizon})', fontsize=12)
    ax.set_ylabel(f'Predictions ({horizon})', fontsize=12)
    ax.set_title(f'Predictions vs True Values ({horizon} - Ensemble Mean)', fontsize=14, pad=15)
    ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'predictions_vs_true_{horizon}.png', bbox_inches='tight')
    plt.close()

# 3. Distribution of predictions vs true values for all horizons
for horizon in horizons:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(preds_df[horizon], bins=50, alpha=0.5, label='True Values', color='blue', density=True)
    ax.hist(preds_df[f'{horizon}_pred_ens_mean'], bins=50, alpha=0.5, label='Predictions', color='orange', density=True)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of {horizon}: True vs Predictions', fontsize=14, pad=15)
    ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'distribution_{horizon}.png', bbox_inches='tight')
    plt.close()

# 4. Summary table
print("\nGlobal Metrics Table:")
print(metrics_df[['model', 'mse', 'mae', 'rmse', 'r2', 'mape', 'directional']].round(4))
print("\nMean of Global Metrics:")
print(metrics_df[['mse', 'mae', 'rmse', 'r2', 'mape', 'directional']].mean().round(4))

print("Graphs saved: mse_by_horizon_plot.png, " + ", ".join([f'predictions_vs_true_{h}.png' for h in horizons]) + ", " + ", ".join([f'distribution_{h}.png' for h in horizons]))