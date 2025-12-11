# utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to get a list of columns by type
def get_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols

# Function to generate the combined plot for numeric data
def plot_numeric_data(df: pd.DataFrame):
    numeric_cols, _ = get_column_types(df)
    
    if not numeric_cols:
        raise ValueError("No numeric columns found.")
    
    # Set up the figure and axes for the plots
    num_plots = len(numeric_cols)
    # Determine grid layout: 2 columns, rows needed
    n_rows = (num_plots + 1) // 2 
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten() # Flatten the array for easy iteration

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        
    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    return fig

# Function to generate the combined plot for categorical data
def plot_categorical_data(df: pd.DataFrame):
    _, categorical_cols = get_column_types(df)
    
    if not categorical_cols:
        raise ValueError("No categorical columns found.")
        
    # Set up the figure and axes for the plots
    num_plots = len(categorical_cols)
    n_rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        # Count plot for categorical data
        sns.countplot(y=col, data=df, ax=axes[i], order=df[col].value_counts().index)
        axes[i].set_title(f'Counts of {col}', fontsize=10)
        
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    return fig

# The summarize_data function is not used by app.py, but included for completeness:
def summarize_data(df: pd.DataFrame):
    # This could return the output of DataProfiler from stats_engine.py
    return df.describe(include='all')