"""
Transition Matrix Heatmap Module

Visualizes regime transition probabilities using seaborn heatmaps.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Union, Dict
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.styles import (
    REGIME_COLORS, REGIME_NAMES,
    save_figure, apply_chronos_style,
    TITLE_FONTSIZE
)


def plot_transition_matrix(
    transition_matrix: Union[np.ndarray, pd.DataFrame],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap visualization of regime transition probabilities.
    
    Args:
        transition_matrix: 3x3 transition probability matrix
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    apply_chronos_style()
    
    # Convert to DataFrame with regime names
    if isinstance(transition_matrix, np.ndarray):
        regime_labels = [REGIME_NAMES[i] for i in range(3)]
        df = pd.DataFrame(
            transition_matrix,
            index=regime_labels,
            columns=regime_labels
        )
    else:
        df = transition_matrix.copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        df,
        annot=True,
        fmt='.2%',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Transition Probability'},
        ax=ax,
        linewidths=0.5,
        linecolor='white',
        square=True
    )
    
    # Formatting
    ax.set_xlabel("To Regime", fontsize=12)
    ax.set_ylabel("From Regime", fontsize=12)
    ax.set_title(
        "Regime Transition Probability Matrix",
        fontsize=TITLE_FONTSIZE,
        fontweight='bold'
    )
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Add interpretation text
    fig.text(
        0.5, 0.02,
        "Diagonal values = Regime persistence | Off-diagonal = Transition probabilities",
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_regime_persistence(
    avg_durations: Dict[int, float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create horizontal bar chart showing average regime durations.
    
    Args:
        avg_durations: Dictionary mapping regime ID to average duration in days
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    apply_chronos_style()
    
    # Prepare data
    regime_names = [REGIME_NAMES[i] for i in sorted(avg_durations.keys())]
    durations = [avg_durations[i] for i in sorted(avg_durations.keys())]
    colors = [REGIME_COLORS[i] for i in sorted(avg_durations.keys())]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(regime_names))
    bars = ax.barh(y_pos, durations, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    for bar, duration in zip(bars, durations):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{duration:.1f} days',
            va='center',
            fontsize=11,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(regime_names)
    ax.set_xlabel("Average Duration (Trading Days)", fontsize=12)
    ax.set_title(
        "Average Regime Duration",
        fontsize=TITLE_FONTSIZE,
        fontweight='bold'
    )
    
    # Extend x-axis to fit labels
    ax.set_xlim(0, max(durations) * 1.3)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_transition_flow(
    transition_matrix: Union[np.ndarray, pd.DataFrame],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a flow diagram showing regime transitions.
    
    Shows arrows between regimes sized by transition probability.
    
    Args:
        transition_matrix: 3x3 transition probability matrix
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    apply_chronos_style()
    
    # Convert to numpy if needed
    if isinstance(transition_matrix, pd.DataFrame):
        transition_matrix = transition_matrix.values
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define node positions (circle layout)
    angles = np.linspace(0, 2 * np.pi, 4)[:-1] - np.pi / 2  # Start from top
    radius = 0.35
    center = (0.5, 0.5)
    
    positions = {}
    for i, angle in enumerate(angles):
        positions[i] = (
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle)
        )
    
    # Draw nodes
    for regime_id, (x, y) in positions.items():
        circle = plt.Circle(
            (x, y), 0.1,
            color=REGIME_COLORS[regime_id],
            ec='white',
            linewidth=2,
            zorder=3
        )
        ax.add_patch(circle)
        
        # Add regime name
        ax.text(
            x, y,
            REGIME_NAMES[regime_id],
            ha='center', va='center',
            fontsize=10,
            fontweight='bold',
            color='white' if regime_id != 1 else 'black',
            zorder=4
        )
        
        # Add self-loop probability
        self_prob = transition_matrix[regime_id, regime_id]
        offset_angle = angles[regime_id] - np.pi / 4
        label_x = center[0] + (radius + 0.15) * np.cos(angles[regime_id])
        label_y = center[1] + (radius + 0.15) * np.sin(angles[regime_id])
        ax.text(
            label_x, label_y,
            f'{self_prob:.0%}',
            ha='center', va='center',
            fontsize=9,
            color=REGIME_COLORS[regime_id],
            fontweight='bold',
            zorder=4
        )
    
    # Draw transition arrows
    from matplotlib.patches import FancyArrowPatch
    
    for from_regime in range(3):
        for to_regime in range(3):
            if from_regime == to_regime:
                continue
            
            prob = transition_matrix[from_regime, to_regime]
            if prob < 0.01:  # Skip very small probabilities
                continue
            
            start = positions[from_regime]
            end = positions[to_regime]
            
            # Calculate arrow positions (outside circles)
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            arrow_start = (
                start[0] + 0.11 * dx / dist,
                start[1] + 0.11 * dy / dist
            )
            arrow_end = (
                end[0] - 0.11 * dx / dist,
                end[1] - 0.11 * dy / dist
            )
            
            # Arrow width based on probability
            arrow_width = max(0.5, prob * 5)
            
            arrow = FancyArrowPatch(
                arrow_start, arrow_end,
                connectionstyle="arc3,rad=0.2",
                arrowstyle=f"simple,head_width={arrow_width*2},head_length={arrow_width}",
                color='gray',
                alpha=0.5 + prob * 0.5,
                linewidth=arrow_width,
                zorder=2
            )
            ax.add_patch(arrow)
            
            # Add probability label
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            offset = 0.08 if from_regime < to_regime else -0.08
            
            # Perpendicular offset
            perp_x = -dy / dist * offset
            perp_y = dx / dist * offset
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(
        "Regime Transition Flow",
        fontsize=TITLE_FONTSIZE,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


if __name__ == '__main__':
    # Test with synthetic transition matrix
    transition_matrix = np.array([
        [0.85, 0.10, 0.05],  # Euphoria -> Euphoria/Complacency/Capitulation
        [0.15, 0.75, 0.10],  # Complacency -> Euphoria/Complacency/Capitulation
        [0.05, 0.20, 0.75],  # Capitulation -> Euphoria/Complacency/Capitulation
    ])
    
    fig1 = plot_transition_matrix(transition_matrix)
    
    avg_durations = {0: 15.5, 1: 22.3, 2: 8.7}
    fig2 = plot_regime_persistence(avg_durations)
    
    fig3 = plot_transition_flow(transition_matrix)
    
    plt.show()
