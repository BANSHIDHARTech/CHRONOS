"""
Interactive Plotly Charts for CHRONOS Dashboard

Provides interactive Plotly versions of key charts for better UX,
with hover tooltips, zoom/pan capabilities, and responsive design.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.styles import REGIME_COLORS, REGIME_NAMES, ASSET_COLORS


# ============================================================================
# REGIME CHART
# ============================================================================

def create_interactive_regime_chart(
    prices: pd.Series,
    regimes: pd.Series,
    dates: pd.DatetimeIndex,
    title: str = "CHRONOS Regime-Based Market Analysis",
    confidence: pd.Series = None
) -> go.Figure:
    """
    Create an interactive regime chart with Plotly.
    
    Args:
        prices: Price series data
        regimes: Regime labels (0, 1, 2)
        dates: DatetimeIndex
        title: Chart title
        confidence: Optional confidence scores per date
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add regime background shapes
    shapes = _create_regime_shapes(dates, regimes)
    
    # Prepare hover text
    hover_template = "<b>Date</b>: %{x|%Y-%m-%d}<br>"
    hover_template += "<b>Price</b>: $%{y:,.2f}<br>"
    
    if confidence is not None:
        hover_template += "<b>Regime</b>: %{customdata[0]}<br>"
        hover_template += "<b>Confidence</b>: %{customdata[1]:.1%}<extra></extra>"
        custom_data = list(zip(
            [REGIME_NAMES.get(r, 'Unknown') for r in regimes],
            confidence.values if isinstance(confidence, pd.Series) else confidence
        ))
    else:
        hover_template += "<b>Regime</b>: %{customdata}<extra></extra>"
        custom_data = [REGIME_NAMES.get(r, 'Unknown') for r in regimes]
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='#1a1a2e', width=2),
        customdata=custom_data,
        hovertemplate=hover_template
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#1a1a2e'),
            x=0.5
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='Price ($)',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        shapes=shapes,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    # Add regime legend annotations
    _add_regime_legend(fig)
    
    return fig


def _create_regime_shapes(dates: pd.DatetimeIndex, regimes: pd.Series) -> List[Dict]:
    """Create Plotly shapes for regime backgrounds."""
    shapes = []
    
    if len(dates) == 0:
        return shapes
    
    regimes_arr = regimes.values if isinstance(regimes, pd.Series) else regimes
    current_regime = regimes_arr[0]
    start_idx = 0
    
    for i in range(1, len(regimes_arr)):
        if regimes_arr[i] != current_regime:
            shapes.append({
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': dates[start_idx],
                'x1': dates[i - 1],
                'y0': 0,
                'y1': 1,
                'fillcolor': REGIME_COLORS.get(current_regime, '#808080'),
                'opacity': 0.2,
                'layer': 'below',
                'line': {'width': 0}
            })
            current_regime = regimes_arr[i]
            start_idx = i
    
    # Add final shape
    shapes.append({
        'type': 'rect',
        'xref': 'x',
        'yref': 'paper',
        'x0': dates[start_idx],
        'x1': dates[-1],
        'y0': 0,
        'y1': 1,
        'fillcolor': REGIME_COLORS.get(current_regime, '#808080'),
        'opacity': 0.2,
        'layer': 'below',
        'line': {'width': 0}
    })
    
    return shapes


def _add_regime_legend(fig: go.Figure):
    """Add regime legend annotations to figure."""
    annotations = []
    x_positions = [0.15, 0.5, 0.85]
    
    for i, (regime_id, regime_name) in enumerate(REGIME_NAMES.items()):
        annotations.append(dict(
            x=x_positions[i],
            y=-0.12,
            xref='paper',
            yref='paper',
            text=f'■ {regime_name}',
            showarrow=False,
            font=dict(color=REGIME_COLORS[regime_id], size=12)
        ))
    
    fig.update_layout(annotations=annotations)


# ============================================================================
# PERFORMANCE CHART
# ============================================================================

def create_interactive_performance_chart(
    chronos_values: pd.Series,
    benchmark_values: pd.Series,
    dates: pd.DatetimeIndex,
    chronos_name: str = "CHRONOS Portfolio",
    benchmark_name: str = "SPY Benchmark"
) -> go.Figure:
    """
    Create an interactive performance comparison chart.
    
    Args:
        chronos_values: CHRONOS portfolio values
        benchmark_values: Benchmark values
        dates: DatetimeIndex
        chronos_name: Name for CHRONOS line
        benchmark_name: Name for benchmark line
        
    Returns:
        Plotly Figure object
    """
    # Normalize to starting value of 100
    chronos_norm = (chronos_values / chronos_values.iloc[0]) * 100
    benchmark_norm = (benchmark_values / benchmark_values.iloc[0]) * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=['Cumulative Performance', 'Drawdown']
    )
    
    # Cumulative performance
    fig.add_trace(go.Scatter(
        x=dates,
        y=chronos_norm,
        mode='lines',
        name=chronos_name,
        line=dict(color='#00C853', width=2.5),
        hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_norm,
        mode='lines',
        name=benchmark_name,
        line=dict(color='#1976D2', width=2, dash='dash'),
        hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Calculate drawdowns
    chronos_dd = _calculate_drawdown(chronos_values)
    benchmark_dd = _calculate_drawdown(benchmark_values)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=chronos_dd * 100,
        mode='lines',
        name=f'{chronos_name} DD',
        line=dict(color='#00C853', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.2)',
        showlegend=False,
        hovertemplate='Drawdown: %{y:.2f}%<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_dd * 100,
        mode='lines',
        name=f'{benchmark_name} DD',
        line=dict(color='#1976D2', width=1.5, dash='dash'),
        showlegend=False,
        hovertemplate='Drawdown: %{y:.2f}%<extra></extra>'
    ), row=2, col=1)
    
    fig.update_layout(
        title=dict(
            text='Portfolio Performance vs Benchmark',
            font=dict(size=18, color='#1a1a2e'),
            x=0.5
        ),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=40, t=100, b=60)
    )
    
    fig.update_yaxes(title_text='Portfolio Value (Normalized)', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    
    return fig


def _calculate_drawdown(values: pd.Series) -> pd.Series:
    """Calculate drawdown series."""
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    return drawdown


# ============================================================================
# ALLOCATION CHART
# ============================================================================

def create_interactive_allocation_chart(
    results_df: pd.DataFrame,
    weight_columns: List[str] = None
) -> go.Figure:
    """
    Create an interactive stacked area chart for portfolio allocations.
    
    Args:
        results_df: DataFrame with weight columns
        weight_columns: List of weight column names (defaults to SPY, TLT, GLD)
        
    Returns:
        Plotly Figure object
    """
    if weight_columns is None:
        weight_columns = ['weight_SPY', 'weight_TLT', 'weight_GLD']
    
    # Filter existing columns
    existing_cols = [col for col in weight_columns if col in results_df.columns]
    
    if not existing_cols:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No allocation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#666')
        )
        return fig
    
    dates = results_df.index if isinstance(results_df.index, pd.DatetimeIndex) else pd.to_datetime(results_df.index)
    
    fig = go.Figure()
    
    colors = {
        'weight_SPY': ASSET_COLORS['SPY'],
        'weight_TLT': ASSET_COLORS['TLT'],
        'weight_GLD': ASSET_COLORS['GLD']
    }
    
    labels = {
        'weight_SPY': 'SPY (Equities)',
        'weight_TLT': 'TLT (Bonds)',
        'weight_GLD': 'GLD (Gold)'
    }
    
    for col in existing_cols:
        fig.add_trace(go.Scatter(
            x=dates,
            y=results_df[col] * 100,
            mode='lines',
            name=labels.get(col, col),
            line=dict(width=0),
            stackgroup='allocation',
            fillcolor=colors.get(col, '#808080'),
            hovertemplate='<b>%{fullData.name}</b><br>Weight: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Portfolio Allocation Over Time',
            font=dict(size=18, color='#1a1a2e'),
            x=0.5
        ),
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='Allocation (%)',
            range=[0, 100],
            ticksuffix='%'
        ),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig


# ============================================================================
# TRANSITION MATRIX HEATMAP
# ============================================================================

def create_interactive_transition_heatmap(
    transition_matrix: np.ndarray,
    regime_names: List[str] = None
) -> go.Figure:
    """
    Create an interactive heatmap for regime transition probabilities.
    
    Args:
        transition_matrix: 3x3 transition probability matrix
        regime_names: Names for each regime
        
    Returns:
        Plotly Figure object
    """
    if regime_names is None:
        regime_names = ['Euphoria', 'Complacency', 'Capitulation']
    
    # Create text annotations
    text_matrix = [[f'{val:.1%}' for val in row] for row in transition_matrix]
    
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=regime_names,
        y=regime_names,
        text=text_matrix,
        texttemplate='%{text}',
        textfont=dict(size=14, color='white'),
        colorscale='YlOrRd',
        hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Regime Transition Probability Matrix',
            font=dict(size=18, color='#1a1a2e'),
            x=0.5
        ),
        xaxis=dict(title='To Regime', side='bottom'),
        yaxis=dict(title='From Regime', autorange='reversed'),
        template='plotly_white',
        margin=dict(l=100, r=40, t=80, b=80)
    )
    
    return fig


# ============================================================================
# ROLLING METRICS CHART
# ============================================================================

def create_rolling_metrics_chart(
    results_df: pd.DataFrame,
    window: int = 13
) -> go.Figure:
    """
    Create an interactive chart showing rolling Sharpe ratio and volatility.
    
    Args:
        results_df: DataFrame with return data
        window: Rolling window in weeks
        
    Returns:
        Plotly Figure object
    """
    dates = results_df.index if isinstance(results_df.index, pd.DatetimeIndex) else pd.to_datetime(results_df.index)
    
    # Calculate rolling metrics
    if 'weekly_return' in results_df.columns:
        returns = results_df['weekly_return']
    elif 'return' in results_df.columns:
        returns = results_df['return']
    else:
        returns = results_df['portfolio_value'].pct_change()
    
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(52)  # Annualized
    rolling_mean = returns.rolling(window=window).mean() * 52  # Annualized
    rolling_sharpe = rolling_mean / rolling_vol
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=['Rolling Sharpe Ratio', 'Rolling Volatility']
    )
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=rolling_sharpe,
        mode='lines',
        name='Rolling Sharpe',
        line=dict(color='#00C853', width=2),
        hovertemplate='Sharpe: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Add zero line for Sharpe
    fig.add_hline(y=0, line_dash='dash', line_color='#808080', row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=rolling_vol * 100,
        mode='lines',
        name='Rolling Volatility',
        line=dict(color='#E65100', width=2),
        fill='tozeroy',
        fillcolor='rgba(230, 81, 0, 0.1)',
        hovertemplate='Volatility: %{y:.1f}%<extra></extra>'
    ), row=2, col=1)
    
    fig.update_layout(
        title=dict(
            text=f'Rolling Metrics ({window}-Week Window)',
            font=dict(size=18, color='#1a1a2e'),
            x=0.5
        ),
        template='plotly_white',
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    fig.update_yaxes(title_text='Sharpe Ratio', row=1, col=1)
    fig.update_yaxes(title_text='Volatility (%)', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    
    return fig


# ============================================================================
# MONTHLY RETURNS HEATMAP
# ============================================================================

def create_monthly_returns_heatmap(results_df: pd.DataFrame) -> go.Figure:
    """
    Create a monthly returns heatmap.
    
    Args:
        results_df: DataFrame with portfolio values
        
    Returns:
        Plotly Figure object
    """
    # Ensure datetime index
    if not isinstance(results_df.index, pd.DatetimeIndex):
        results_df.index = pd.to_datetime(results_df.index)
    
    # Calculate daily returns
    if 'portfolio_value' in results_df.columns:
        daily_returns = results_df['portfolio_value'].pct_change()
    else:
        daily_returns = results_df.iloc[:, 0].pct_change()
    
    # Resample to monthly returns
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create pivot table
    monthly_returns_df = monthly_returns.to_frame('return')
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month
    
    pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')
    
    # Create month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create text annotations
    text_matrix = [[f'{val:.1%}' if pd.notna(val) else '' for val in row] for row in pivot.values]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=month_labels[:pivot.shape[1]],
        y=pivot.index.astype(str),
        text=text_matrix,
        texttemplate='%{text}',
        textfont=dict(size=11),
        colorscale='RdYlGn',
        zmid=0,
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Monthly Returns Heatmap',
            font=dict(size=18, color='#1a1a2e'),
            x=0.5
        ),
        xaxis=dict(title='Month'),
        yaxis=dict(title='Year', autorange='reversed'),
        template='plotly_white',
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig
