"""
CHRONOS - Adaptive Regime Intelligence Dashboard

Main Streamlit application providing interactive visualization
for regime detection, portfolio performance, and SHAP interpretability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CHRONOS - Adaptive Regime Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORTS (after path setup)
# ============================================================================

from config import (
    OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR,
    TEST_START, TEST_END, INITIAL_CAPITAL,
    REGIME_CONSTRAINTS, PORTFOLIO_ASSETS
)
from src.visualization.custom_css import (
    get_custom_css, get_regime_badge_html, 
    get_allocation_bar_html, get_hero_section_html
)
from src.utils.streamlit_helpers import (
    load_backtest_results, load_models, load_shap_plots,
    export_results_for_download, export_summary_for_download,
    get_regime_name, get_regime_color, format_percentage, format_currency,
    show_data_error, show_missing_data_warning, get_model_info
)
from src.utils.backtest_runner import (
    run_backtest_on_demand, validate_date_range,
    get_rebalance_frequency_options, format_backtest_summary
)
from src.utils.scenario_analyzer import (
    analyze_regime_override, get_scenario_comparison_table,
    get_regime_recommendation
)

# ============================================================================
# APPLY CUSTOM STYLING
# ============================================================================

st.markdown(get_custom_css(), unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Logo and Title
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">📈 <span>CHRONOS</span></div>
        <p style="color: #666; margin-top: 0.5rem;">Adaptive Regime Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Parameter Controls
    st.subheader("⚙️ Parameters")
    
    lookback_window = st.slider(
        "Lookback Window (days)",
        min_value=20,
        max_value=252,
        value=63,
        help="Number of days for rolling calculations"
    )
    
    rebalance_options = get_rebalance_frequency_options()
    rebalance_freq = st.selectbox(
        "Rebalance Frequency",
        options=list(rebalance_options.keys()),
        format_func=lambda x: rebalance_options[x],
        index=0
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum confidence for regime-based allocation adjustments"
    )
    
    st.divider()
    
    # Run Backtest Button
    st.subheader("🚀 Run Backtest")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.strptime(TEST_START, '%Y-%m-%d'),
            help="Backtest start date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.strptime(TEST_END, '%Y-%m-%d'),
            help="Backtest end date"
        )
    
    if st.button("🚀 Run Backtest", use_container_width=True, type="primary"):
        with st.spinner("Running backtest..."):
            results_df, summary_stats = run_backtest_on_demand(
                start_date=str(start_date),
                end_date=str(end_date),
                rebalance_freq=rebalance_freq
            )
            if results_df is not None:
                st.success("✅ Backtest completed!")
                # Clear cache to ensure fresh data is loaded on rerun
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
    
    st.divider()
    
    # Model Information
    with st.expander("ℹ️ Model Information"):
        # Cache model loading to prevent redundant disk reads
        @st.cache_resource
        def load_models_cached():
            return load_models()
        
        regime_detector, ensemble = load_models_cached()
        model_info = get_model_info(regime_detector, ensemble)
        
        st.write(f"**Regime Detector:** {'✅ Loaded' if model_info['regime_detector_loaded'] else '❌ Not Found'}")
        st.write(f"**Ensemble Models:** {model_info['ensemble_models_loaded']}/3 loaded")
        st.write(f"**Training Period:** {model_info['training_period']}")
        st.write(f"**Test Period:** {model_info['test_period']}")
    
    # Cache Management
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
    
    st.divider()
    
    # About Section
    with st.expander("📖 About CHRONOS"):
        st.markdown("""
        **CHRONOS** (Conditional Hierarchical Regime-Optimized 
        Navigation & Oversight System) is an adaptive regime 
        intelligence platform for quantitative finance.
        
        **Key Features:**
        - HMM-based regime detection
        - XGBoost ensemble predictions
        - CVaR portfolio optimization
        - SHAP interpretability
        
        **Regimes:**
        - 🟢 **Euphoria**: Risk-on, high equity
        - 🟡 **Complacency**: Balanced allocation
        - 🔴 **Capitulation**: Risk-off, defensive
        """)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data(ttl=3600)
def load_dashboard_data():
    """Load all dashboard data with caching."""
    return load_backtest_results()

data = load_dashboard_data()

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

# Check if data is available
if data['results_df'] is None:
    st.warning("⚠️ No backtest results found. Please run a backtest first using the sidebar button.")
    st.info("💡 Tip: Click 'Run Backtest' in the sidebar to generate results.")
    st.stop()

results_df = data['results_df']
summary_stats = data['summary_statistics'] or {}
dashboard_data = data['dashboard_data'] or {}

# ============================================================================
# SECTION 1: HERO - CURRENT STATUS
# ============================================================================

# Get current regime from latest data point
if 'regime' in results_df.columns:
    current_regime = int(results_df['regime'].iloc[-1])
else:
    current_regime = 1  # Default to Complacency

# Get confidence if available
confidence = 0.82  # Default
if 'regime_confidence' in results_df.columns:
    confidence = results_df['regime_confidence'].iloc[-1]

# Get current weights
current_weights = {}
for asset in PORTFOLIO_ASSETS:
    weight_col = f'weight_{asset}'
    if weight_col in results_df.columns:
        current_weights[asset] = results_df[weight_col].iloc[-1]
    else:
        current_weights[asset] = 1.0 / len(PORTFOLIO_ASSETS)

# Display hero section
last_update = results_df.index[-1].strftime('%Y-%m-%d') if hasattr(results_df.index[-1], 'strftime') else str(results_df.index[-1])
st.markdown(get_hero_section_html(current_regime, confidence, last_update), unsafe_allow_html=True)

# Current allocation bar
st.markdown("### Current Portfolio Allocation")
spy_wt = current_weights.get('SPY', 0.33)
tlt_wt = current_weights.get('TLT', 0.33)
gld_wt = current_weights.get('GLD', 0.34)
st.markdown(get_allocation_bar_html(spy_wt, tlt_wt, gld_wt), unsafe_allow_html=True)
st.markdown("")

# Key metrics row
metrics_cols = st.columns(4)

with metrics_cols[0]:
    # Use correct key from summary_statistics.json
    total_return = summary_stats.get('total_return_chronos', summary_stats.get('total_return', 0))
    st.metric(
        "Total Return",
        f"{total_return:.2%}" if isinstance(total_return, (int, float)) else str(total_return),
        delta=f"+{(total_return - summary_stats.get('total_return_benchmark', 0)):.2%}" if total_return > 0 else None
    )

with metrics_cols[1]:
    sharpe = summary_stats.get('sharpe_ratio_chronos', summary_stats.get('sharpe_ratio', 0))
    st.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
    )

with metrics_cols[2]:
    max_dd = summary_stats.get('max_dd_chronos', summary_stats.get('max_drawdown', 0))
    st.metric(
        "Max Drawdown",
        f"{max_dd:.2%}" if isinstance(max_dd, (int, float)) else str(max_dd)
    )

with metrics_cols[3]:
    # Use correct key from summary_statistics.json
    final_value = summary_stats.get('final_value_chronos', summary_stats.get('final_portfolio_value', INITIAL_CAPITAL))
    st.metric(
        "Final Portfolio Value",
        f"${final_value:,.0f}" if isinstance(final_value, (int, float)) else str(final_value)
    )

st.divider()

# ============================================================================
# SECTION 2: REGIME DETECTION VISUALIZATION
# ============================================================================

st.header("📊 Market Regime Detection")

# Date range selector
col1, col2 = st.columns([3, 1])
with col2:
    if len(results_df) > 0:
        min_date = results_df.index.min()
        max_date = results_df.index.max()
        
        if hasattr(min_date, 'date'):
            min_date = min_date.date()
            max_date = max_date.date()
        
        date_range = st.date_input(
            "Date Range Filter",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

# Filter data based on date range
if len(date_range) == 2:
    filtered_df = results_df.loc[str(date_range[0]):str(date_range[1])]
else:
    filtered_df = results_df

with col1:
    # Regime chart
    try:
        from src.visualization.plotly_charts import create_interactive_regime_chart
        
        if 'portfolio_value' in filtered_df.columns and 'regime' in filtered_df.columns:
            dates = filtered_df.index
            prices = filtered_df['portfolio_value']
            regimes = filtered_df['regime']
            
            fig = create_interactive_regime_chart(
                prices=prices,
                regimes=regimes,
                dates=dates,
                title="Portfolio Value with Market Regimes"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Regime data not available in results.")
    except Exception as e:
        st.error(f"Error creating regime chart: {e}")

# Regime statistics
if 'regime' in results_df.columns:
    st.subheader("Regime Statistics")
    regime_counts = results_df['regime'].value_counts().sort_index()
    
    stat_cols = st.columns(3)
    for i, (regime_id, count) in enumerate(regime_counts.items()):
        with stat_cols[i]:
            regime_name = get_regime_name(regime_id)
            pct = count / len(results_df) * 100
            st.markdown(f"""
            <div style="background-color: {get_regime_color(regime_id)}20; 
                        padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="color: {get_regime_color(regime_id)}; margin: 0;">{regime_name}</h4>
                <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{pct:.1f}%</p>
                <p style="color: #666; margin: 0;">{count} periods</p>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ============================================================================
# SECTION 3: PERFORMANCE ANALYSIS
# ============================================================================

st.header("💰 Portfolio Performance vs Benchmark")

perf_col1, perf_col2 = st.columns([2, 1])

with perf_col1:
    # Performance chart tabs
    perf_tabs = st.tabs(["📈 Cumulative Returns", "📉 Drawdown", "📊 Rolling Metrics"])
    
    with perf_tabs[0]:
        try:
            from src.visualization.plotly_charts import create_interactive_performance_chart
            
            if 'portfolio_value' in results_df.columns and 'benchmark_value' in results_df.columns:
                fig = create_interactive_performance_chart(
                    chronos_values=results_df['portfolio_value'],
                    benchmark_values=results_df['benchmark_value'],
                    dates=results_df.index
                )
                st.plotly_chart(fig, use_container_width=True)
            elif 'portfolio_value' in results_df.columns:
                # Just show portfolio if no benchmark
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results_df.index,
                    y=results_df['portfolio_value'],
                    mode='lines',
                    name='CHRONOS Portfolio',
                    line=dict(color='#00C853', width=2)
                ))
                fig.update_layout(
                    title='Portfolio Value Over Time',
                    xaxis_title='Date',
                    yaxis_title='Portfolio Value ($)',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating performance chart: {e}")
    
    with perf_tabs[1]:
        try:
            # Drawdown chart
            if 'portfolio_value' in results_df.columns:
                values = results_df['portfolio_value']
                cummax = values.cummax()
                drawdown = (values - cummax) / cummax
                
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results_df.index,
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#D50000', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(213, 0, 0, 0.2)'
                ))
                fig.update_layout(
                    title='Portfolio Drawdown',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    template='plotly_white',
                    yaxis=dict(ticksuffix='%')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Max drawdown annotation
                max_dd_date = drawdown.idxmin()
                st.info(f"📉 Maximum Drawdown: {drawdown.min():.2%} on {max_dd_date.strftime('%Y-%m-%d') if hasattr(max_dd_date, 'strftime') else max_dd_date}")
        except Exception as e:
            st.error(f"Error creating drawdown chart: {e}")
    
    with perf_tabs[2]:
        try:
            from src.visualization.plotly_charts import create_rolling_metrics_chart
            
            fig = create_rolling_metrics_chart(results_df, window=lookback_window // 5)  # Convert to weeks
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating rolling metrics chart: {e}")

with perf_col2:
    # Metrics panel
    st.subheader("📋 Performance Metrics")
    
    if summary_stats:
        metrics_df = format_backtest_summary(summary_stats)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("Summary statistics not available")
    
    # Monthly returns heatmap
    st.subheader("📅 Monthly Returns")
    try:
        from src.visualization.plotly_charts import create_monthly_returns_heatmap
        fig = create_monthly_returns_heatmap(results_df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Monthly heatmap unavailable: {e}")

st.divider()

# ============================================================================
# SECTION 3.5: CRISIS ANALYSIS
# ============================================================================

st.header("⚠️ Crisis Performance Analysis")

# Load crisis analysis data
crisis_df = data.get('crisis_analysis')

if crisis_df is not None and len(crisis_df) > 0:
    # Filter for analyzed crises only
    analyzed_crises = crisis_df[crisis_df['status'] == 'analyzed'].copy()
    
    if len(analyzed_crises) > 0:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
            <h3 style="color: white; margin: 0 0 0.5rem 0;">🛡️ Defensive Performance During Market Stress</h3>
            <p style="color: #e0e7ff; margin: 0;">CHRONOS automatically shifts to defensive allocations during crisis periods</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display each crisis
        for idx, crisis in analyzed_crises.iterrows():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"### {crisis['name']}")
                st.caption(f"{crisis['start']} to {crisis['end']}")
                st.caption(crisis['description'])
            
            with col2:
                chronos_ret = crisis['chronos_return'] * 100
                color = "#00C853" if chronos_ret >= 0 else "#D50000"
                st.markdown(f"<div style='text-align: center;'>"
                           f"<p style='margin: 0; color: #666; font-size: 0.85rem;'>CHRONOS Return</p>"
                           f"<p style='margin: 0; color: {color}; font-size: 1.8rem; font-weight: bold;'>{chronos_ret:.2f}%</p>"
                           f"</div>", unsafe_allow_html=True)
            
            with col3:
                bench_ret = crisis['benchmark_return'] * 100
                color = "#00C853" if bench_ret >= 0 else "#D50000"
                st.markdown(f"<div style='text-align: center;'>"
                           f"<p style='margin: 0; color: #666; font-size: 0.85rem;'>SPY Return</p>"
                           f"<p style='margin: 0; color: {color}; font-size: 1.8rem; font-weight: bold;'>{bench_ret:.2f}%</p>"
                           f"</div>", unsafe_allow_html=True)
            
            with col4:
                outperformance = crisis['outperformance'] * 100
                color = "#00C853" if outperformance >= 0 else "#D50000"
                st.markdown(f"<div style='text-align: center;'>"
                           f"<p style='margin: 0; color: #666; font-size: 0.85rem;'>Protection</p>"
                           f"<p style='margin: 0; color: {color}; font-size: 1.8rem; font-weight: bold;'>+{outperformance:.2f}%</p>"
                           f"</div>", unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Add COVID context note if not in dataset
        st.info("💡 **Note**: COVID-2020 crash occurred before our backtest period (2024), but similar defensive mechanics protected the portfolio during the 2024 Rate Hike Selloff and August Volatility events.")
    else:
        st.info("No crisis periods detected in the backtest period.")
else:
    st.info("Crisis analysis data not available. Run the full backtest pipeline to generate crisis analysis.")

st.divider()

# ============================================================================
# SECTION 4: SHAP INTERPRETABILITY
# ============================================================================

st.header("🔍 Model Interpretability (SHAP Analysis)")

shap_regime = st.selectbox(
    "Select Regime for SHAP Analysis",
    options=[0, 1, 2],
    format_func=lambda x: f"{get_regime_name(x)} (Regime {x})"
)

shap_plots = load_shap_plots(shap_regime)

if shap_plots:
    shap_cols = st.columns(len(shap_plots))
    
    for i, (plot_type, plot_path) in enumerate(shap_plots.items()):
        with shap_cols[i]:
            st.subheader(f"{plot_type.title()} Plot")
            try:
                st.image(plot_path, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load {plot_type} plot")
else:
    st.info(f"📊 SHAP plots not available for {get_regime_name(shap_regime)}. Run the SHAP analysis pipeline to generate interpretability plots.")
    
    # Show SHAP directory structure for debugging
    shap_dir = os.path.join(OUTPUTS_DIR, 'shap')
    if os.path.exists(shap_dir):
        with st.expander("Available SHAP directories"):
            for item in os.listdir(shap_dir):
                st.write(f"  - {item}")

st.divider()

# ============================================================================
# SECTION 5: PORTFOLIO ALLOCATION
# ============================================================================

st.header("⚖️ Portfolio Allocation Over Time")

alloc_tabs = st.tabs(["📊 Weight Evolution", "📈 By Regime", "📝 Trade Log"])

with alloc_tabs[0]:
    try:
        from src.visualization.plotly_charts import create_interactive_allocation_chart
        
        fig = create_interactive_allocation_chart(results_df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating allocation chart: {e}")

with alloc_tabs[1]:
    # Average weights by regime
    st.subheader("Average Allocation by Regime")
    
    if 'regime' in results_df.columns:
        weight_cols = [f'weight_{asset}' for asset in PORTFOLIO_ASSETS if f'weight_{asset}' in results_df.columns]
        
        if weight_cols:
            avg_weights = results_df.groupby('regime')[weight_cols].mean()
            
            import plotly.graph_objects as go
            from src.visualization.styles import ASSET_COLORS
            
            fig = go.Figure()
            
            for col in weight_cols:
                asset = col.replace('weight_', '')
                fig.add_trace(go.Bar(
                    name=asset,
                    x=[get_regime_name(r) for r in avg_weights.index],
                    y=avg_weights[col] * 100,
                    marker_color=ASSET_COLORS.get(asset, '#808080')
                ))
            
            fig.update_layout(
                barmode='group',
                title='Average Portfolio Weights by Regime',
                xaxis_title='Regime',
                yaxis_title='Weight (%)',
                template='plotly_white',
                yaxis=dict(ticksuffix='%')
            )
            st.plotly_chart(fig, use_container_width=True)

with alloc_tabs[2]:
    # Trade log
    if data['trade_log'] is not None:
        st.subheader("Trade Log")
        trade_log = data['trade_log']
        
        # Add filtering
        if 'asset' in trade_log.columns:
            asset_filter = st.multiselect(
                "Filter by Asset",
                options=trade_log['asset'].unique().tolist(),
                default=trade_log['asset'].unique().tolist()
            )
            trade_log = trade_log[trade_log['asset'].isin(asset_filter)]
        
        st.dataframe(
            trade_log.tail(50),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Showing last 50 of {len(data['trade_log'])} trades")
    else:
        st.info("Trade log not available. Run a backtest to generate trade history.")

st.divider()

# ============================================================================
# SECTION 6: WHAT-IF SCENARIO ANALYSIS
# ============================================================================

st.header("🎯 What-If Scenario Analysis")

scenario_col1, scenario_col2 = st.columns([1, 1])

with scenario_col1:
    st.subheader("Override Current Regime")
    
    manual_regime = st.selectbox(
        "Override Regime",
        options=['Auto-detect', 'Euphoria (0)', 'Complacency (1)', 'Capitulation (2)'],
        index=0
    )
    
    if manual_regime != 'Auto-detect':
        override_regime = int(manual_regime.split('(')[1].replace(')', ''))
        
        # Analyze override
        analysis = analyze_regime_override(
            current_regime=current_regime,
            override_regime=override_regime,
            current_weights=current_weights,
            confidence=confidence
        )
        
        # Display comparison table
        st.subheader("Allocation Comparison")
        comparison_df = get_scenario_comparison_table(analysis)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Risk impact
        risk = analysis['risk_impact']
        st.metric(
            "Risk Direction",
            risk['risk_direction'],
            delta=None
        )
        
with scenario_col2:
    st.subheader("Regime Recommendation")
    
    recommendation = get_regime_recommendation(
        current_regime=current_regime,
        model_confidence=confidence,
        confidence_threshold=confidence_threshold
    )
    
    st.write(f"**Current Regime:** {recommendation['current_regime']}")
    st.write(f"**Model Confidence:** {recommendation['model_confidence']:.1%}")
    st.write(f"**Confidence Threshold:** {recommendation['confidence_threshold']:.0%}")
    st.write(f"**Reason:** {recommendation['reason']}")
    st.write(f"**Suggested Action:** {recommendation.get('suggested_action', 'N/A')}")
    
    # Show confidence status indicator
    confidence_status = recommendation.get('confidence_status', 'unknown')
    if confidence_status == 'low':
        st.error("🔴 Low Confidence - Manual review recommended")
    elif confidence_status == 'moderate':
        st.warning("🟡 Moderate Confidence - Monitor for regime changes")
    elif confidence_status == 'high':
        st.success("🟢 High Confidence - Follow model recommendations")
    
    if recommendation['override_suggested']:
        st.warning("⚠️ Low confidence - consider manual override")

st.divider()

# ============================================================================
# SECTION 7: DOWNLOAD & EXPORT
# ============================================================================

st.header("📥 Download Results")

download_cols = st.columns(3)

with download_cols[0]:
    if results_df is not None:
        csv_data = export_results_for_download(results_df)
        st.download_button(
            label="📊 Download Backtest Results (CSV)",
            data=csv_data,
            file_name="chronos_backtest_results.csv",
            mime="text/csv",
            use_container_width=True
        )

with download_cols[1]:
    if summary_stats:
        json_data = export_summary_for_download(summary_stats)
        st.download_button(
            label="📈 Download Summary Statistics (JSON)",
            data=json_data,
            file_name="chronos_summary_stats.json",
            mime="application/json",
            use_container_width=True
        )

with download_cols[2]:
    if data['trade_log'] is not None:
        trade_csv = data['trade_log'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📝 Download Trade Log (CSV)",
            data=trade_csv,
            file_name="chronos_trade_log.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.button("📝 Trade Log (N/A)", disabled=True, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>CHRONOS - Adaptive Regime Intelligence Platform</p>
    <p style="font-size: 0.8rem;">Built with Streamlit • Powered by XGBoost & HMM</p>
</div>
""", unsafe_allow_html=True)
