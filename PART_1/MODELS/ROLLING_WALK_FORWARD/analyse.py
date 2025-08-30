import pandas as pd
import numpy as np
from datetime import datetime

# Load the CSV file
df = pd.read_csv('walk_forward_results.csv')

# Select numerical columns for statistical analysis
numerical_cols = df.select_dtypes(include=[np.number]).columns

# Calculate statistics: mean, median, max, min for each numerical column
stats = {
    'Mean': df[numerical_cols].mean(),
    'Median': df[numerical_cols].median(),
    'Max': df[numerical_cols].max(),
    'Min': df[numerical_cols].min()
}

# Convert stats to a DataFrame for better formatting
stats_df = pd.DataFrame(stats, index=numerical_cols)

# Find the window with the highest F1-Score
best_window = df.loc[df['F1-Score'].idxmax()]

# Create a clean report as a string for terminal output
report = f"""
Data Analysis Report
===================
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Statistical Summary
------------------
{stats_df.to_string(float_format='%.6f')}

Best Window for F1-Score
------------------------
The window with the highest F1-Score is '{best_window['Window']}' with the following metrics:
- F1-Score: {best_window['F1-Score']:.3f}
- Precision: {best_window['Precision']:.3f}
- Recall: {best_window['Recall']:.3f}
- Buy Precision: {best_window['Buy_Precision']:.3f}
- Buy Recall: {best_window['Buy_Recall']:.3f}
- Buy F1-Score: {best_window['Buy_F1-Score']:.3f}
- Buy Support: {best_window['Buy_Support']:.0f}
- Hold Precision: {best_window['Hold_Precision']:.3f}
- Hold Recall: {best_window['Hold_Recall']:.3f}
- Hold F1-Score: {best_window['Hold_F1-Score']:.3f}
- Hold Support: {best_window['Hold_Support']:.0f}
- Sell Precision: {best_window['Sell_Precision']:.3f}
- Sell Recall: {best_window['Sell_Recall']:.3f}
- Sell F1-Score: {best_window['Sell_F1-Score']:.3f}
- Sell Support: {best_window['Sell_Support']:.0f}
"""

# Print the report to the terminal
print(report)