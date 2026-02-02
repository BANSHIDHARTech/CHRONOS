"""
CHRONOS Visualization Styling Module

Provides consistent visual styling for all CHRONOS plots,
including color palettes, fonts, and figure utilities.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator
import seaborn as sns
from pathlib import Path
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FIGURES_DIR
from src.utils.regime_utils import REGIME_CHARACTERISTICS


# ============================================================================
# DEFAULT FIGURE PARAMETERS
# ============================================================================

DEFAULT_DPI = 150
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10

# ============================================================================
# REGIME COLOR PALETTE
# ============================================================================

# Extract hex colors from REGIME_CHARACTERISTICS
REGIME_COLORS = {
    0: REGIME_CHARACTERISTICS[0]['hex_color'],  # Euphoria - #00C853 (Green)
    1: REGIME_CHARACTERISTICS[1]['hex_color'],  # Complacency - #FFD600 (Yellow)
    2: REGIME_CHARACTERISTICS[2]['hex_color'],  # Capitulation - #D50000 (Red)
}

# Regime names for labels
REGIME_NAMES = {
    0: 'Euphoria',
    1: 'Complacency',
    2: 'Capitulation'
}

# Background transparency for regime spans
REGIME_ALPHA = 0.3

# ============================================================================
# CRISIS HIGHLIGHTING
# ============================================================================

CRISIS_COLOR = '#FF6B6B'
CRISIS_ALPHA = 0.2

# ============================================================================
# ASSET COLORS
# ============================================================================

ASSET_COLORS = {
    'SPY': '#2E7D32',  # Dark green
    'TLT': '#1565C0',  # Dark blue
    'GLD': '#F9A825',  # Gold
}

# ============================================================================
# GENERAL COLORS
# ============================================================================

CHRONOS_GREEN = '#00C853'
BENCHMARK_BLUE = '#1976D2'
POSITIVE_COLOR = '#4CAF50'
NEGATIVE_COLOR = '#F44336'


# ============================================================================
# STYLE APPLICATION
# ============================================================================

def apply_chronos_style():
    """
    Apply consistent CHRONOS styling to matplotlib.
    Sets default rcParams for all plots.
    """
    # Set seaborn style first
    sns.set_style('whitegrid')
    
    # Custom rcParams
    plt.rcParams.update({
        # Figure settings
        'figure.dpi': DEFAULT_DPI,
        'savefig.dpi': DEFAULT_DPI,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        
        # Font settings
        'font.size': TICK_FONTSIZE,
        'axes.titlesize': TITLE_FONTSIZE,
        'axes.labelsize': LABEL_FONTSIZE,
        'xtick.labelsize': TICK_FONTSIZE,
        'ytick.labelsize': TICK_FONTSIZE,
        'legend.fontsize': TICK_FONTSIZE,
        
        # Font weight
        'axes.titleweight': 'bold',
        
        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        
        # Lines
        'lines.linewidth': 2,
        
        # Layout
        'figure.autolayout': False,
    })
    
    # Set seaborn palette with regime colors
    sns.set_palette([REGIME_COLORS[0], REGIME_COLORS[1], REGIME_COLORS[2]])


def save_figure(fig, filename: str, output_dir: str = None):
    """
    Save figure to the specified directory with consistent settings.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the file (with extension)
        output_dir: Output directory path (defaults to FIGURES_DIR)
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    fig.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight', facecolor='white')
    
    print(f"Saved figure: {filepath}")
    return str(filepath)


def format_date_axis(ax, rotation: int = 45):
    """
    Format x-axis with date formatting for time series plots.
    
    Args:
        ax: Matplotlib axes object
        rotation: Rotation angle for date labels
    """
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')


def create_figure(figsize: tuple = (14, 7)):
    """
    Create a figure with CHRONOS styling applied.
    
    Args:
        figsize: Tuple of (width, height) in inches
        
    Returns:
        Tuple of (fig, ax)
    """
    apply_chronos_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def format_percentage_axis(ax, axis: str = 'y'):
    """
    Format axis labels as percentages.
    
    Args:
        ax: Matplotlib axes object
        axis: 'x' or 'y' to specify which axis
    """
    from matplotlib.ticker import FuncFormatter
    
    def pct_formatter(x, pos):
        return f'{x:.0%}' if abs(x) < 1 else f'{x:.0f}%'
    
    if axis == 'y':
        ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))


# Apply styling on import
apply_chronos_style()
