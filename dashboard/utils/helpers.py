"""
Dashboard Utilities
===================
Helper functions for the Beijing Air Quality Dashboard
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_experiment_results(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load experiment results from dashboard_summary.json
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        Dictionary with experiment results or None if not found
    """
    summary_file = exp_dir / "dashboard_summary.json"
    
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_parquet_safe(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Safely load a parquet file
    
    Args:
        file_path: Path to parquet file
        
    Returns:
        DataFrame or None if file doesn't exist
    """
    if file_path.exists():
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    return None


def format_number(value: float, precision: int = 2) -> str:
    """
    Format number with thousands separator
    
    Args:
        value: Number to format
        precision: Decimal places
        
    Returns:
        Formatted string
    """
    if isinstance(value, (int, float)):
        return f"{value:,.{precision}f}"
    return str(value)


def calculate_percentage_change(old_value: float, new_value: float) -> str:
    """
    Calculate percentage change
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Formatted percentage string with + or -
    """
    if old_value == 0:
        return "N/A"
    
    change = ((new_value - old_value) / old_value) * 100
    sign = "+" if change > 0 else ""
    return f"{sign}{change:.1f}%"


def get_color_scale(metric: str) -> str:
    """
    Get appropriate color scale for a metric
    
    Args:
        metric: Name of the metric
        
    Returns:
        Plotly color scale name
    """
    # Lower is better
    if any(x in metric.lower() for x in ['rmse', 'mae', 'error', 'loss']):
        return 'RdYlGn_r'
    
    # Higher is better
    if any(x in metric.lower() for x in ['accuracy', 'f1', 'precision', 'recall', 'r2', 'auc']):
        return 'Blues'
    
    # Neutral
    return 'Viridis'


def create_comparison_table(
    configs: list,
    metrics: list,
    values: list
) -> pd.DataFrame:
    """
    Create a comparison table from experiment results
    
    Args:
        configs: List of configuration names
        metrics: List of metric names
        values: List of lists with metric values per config
        
    Returns:
        DataFrame with comparison table
    """
    data = {'Configuration': configs}
    for i, metric in enumerate(metrics):
        data[metric] = [vals[i] for vals in values]
    
    return pd.DataFrame(data)


def highlight_best_values(df: pd.DataFrame, columns: list, higher_is_better: bool = True):
    """
    Highlight best values in a dataframe
    
    Args:
        df: DataFrame to style
        columns: Columns to highlight
        higher_is_better: Whether higher values are better
        
    Returns:
        Styled DataFrame
    """
    def highlight_max(s):
        is_max = s == s.max() if higher_is_better else s == s.min()
        return ['background-color: #ccfbf1' if v else '' for v in is_max]
    
    return df.style.apply(highlight_max, subset=columns)


# Color palette for visualizations (Ocean Blue theme)
COLORS = {
    'primary': '#0ea5e9',      # Sky blue
    'secondary': '#06b6d4',    # Cyan
    'accent': '#0284c7',       # Blue
    'dark': '#0369a1',         # Dark blue
    'success': '#14b8a6',      # Teal
    'light': '#e0f2fe',        # Light blue
    'gray': '#64748b'          # Slate gray
}


def get_gradient_colors(n: int) -> list:
    """
    Generate n colors in ocean blue gradient
    
    Args:
        n: Number of colors needed
        
    Returns:
        List of hex colors
    """
    if n <= 5:
        return [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                COLORS['dark'], COLORS['success']][:n]
    
    # Generate gradient for more colors
    import plotly.express as px
    return px.colors.sample_colorscale('Blues', [i/(n-1) for i in range(n)])
