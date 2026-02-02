"""
Custom CSS Styling for CHRONOS Streamlit Dashboard

Provides custom CSS for regime badges, metric cards, headers,
and other visual elements with CHRONOS branding.
"""


def get_custom_css() -> str:
    """
    Return custom CSS for the CHRONOS dashboard.
    
    Returns:
        CSS string to be injected via st.markdown
    """
    return """
    <style>
    /* ============================================
       CHRONOS DASHBOARD CUSTOM STYLES
       ============================================ */
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1a1a2e;
        font-weight: 700;
        border-bottom: 3px solid #00C853;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: #16213e;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #0f3460;
        font-weight: 500;
    }
    
    /* ============================================
       REGIME BADGES
       ============================================ */
    
    .regime-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .regime-euphoria {
        background: linear-gradient(135deg, #00C853 0%, #69F0AE 100%);
        color: white;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .regime-complacency {
        background: linear-gradient(135deg, #FFD600 0%, #FFEE58 100%);
        color: #333;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.3);
    }
    
    .regime-capitulation {
        background: linear-gradient(135deg, #D50000 0%, #FF5252 100%);
        color: white;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* ============================================
       METRIC CARDS
       ============================================ */
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #00C853;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card.positive {
        border-left-color: #00C853;
    }
    
    .metric-card.negative {
        border-left-color: #D50000;
    }
    
    .metric-card.neutral {
        border-left-color: #FFD600;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .metric-delta.positive {
        color: #00C853;
    }
    
    .metric-delta.negative {
        color: #D50000;
    }
    
    /* ============================================
       STATUS INDICATORS
       ============================================ */
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-dot.active {
        background-color: #00C853;
    }
    
    .status-dot.warning {
        background-color: #FFD600;
    }
    
    .status-dot.error {
        background-color: #D50000;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* ============================================
       ALLOCATION DISPLAY
       ============================================ */
    
    .allocation-bar {
        display: flex;
        height: 30px;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .allocation-segment {
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .allocation-spy {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
    }
    
    .allocation-tlt {
        background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%);
    }
    
    .allocation-gld {
        background: linear-gradient(135deg, #F9A825 0%, #FDD835 100%);
    }
    
    /* ============================================
       HERO SECTION
       ============================================ */
    
    .hero-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #00C853, #69F0AE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 1.5rem;
    }
    
    /* ============================================
       DOWNLOAD BUTTONS
       ============================================ */
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00C853 0%, #00E676 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 200, 83, 0.4);
    }
    
    /* ============================================
       SIDEBAR STYLING
       ============================================ */
    
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid #00C853;
        margin-bottom: 1rem;
    }
    
    .sidebar-logo {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .sidebar-logo span {
        color: #00C853;
    }
    
    /* ============================================
       DATA TABLES
       ============================================ */
    
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe th {
        background-color: #1a1a2e !important;
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .dataframe td {
        font-size: 0.95rem;
    }
    
    /* ============================================
       TABS STYLING
       ============================================ */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00C853;
        color: white;
    }
    
    /* ============================================
       EXPANDER STYLING
       ============================================ */
    
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1a1a2e;
    }
    
    /* ============================================
       TOOLTIPS AND INFO
       ============================================ */
    
    .info-tooltip {
        display: inline-block;
        width: 18px;
        height: 18px;
        background-color: #e0e0e0;
        border-radius: 50%;
        text-align: center;
        line-height: 18px;
        font-size: 12px;
        color: #666;
        cursor: help;
        margin-left: 4px;
    }
    
    /* ============================================
       CONFIDENCE METER
       ============================================ */
    
    .confidence-meter {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .confidence-bar {
        flex: 1;
        height: 10px;
        background-color: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #00C853, #69F0AE);
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    
    .confidence-value {
        font-weight: 700;
        font-size: 1.1rem;
        color: #1a1a2e;
        min-width: 50px;
    }
    
    /* ============================================
       RESPONSIVE ADJUSTMENTS
       ============================================ */
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.8rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .regime-badge {
            font-size: 1rem;
            padding: 0.4rem 1rem;
        }
    }
    </style>
    """


def get_regime_badge_html(regime_id: int, confidence: float = None) -> str:
    """
    Generate HTML for a regime badge.
    
    Args:
        regime_id: Regime identifier (0, 1, or 2)
        confidence: Optional confidence score (0-1)
        
    Returns:
        HTML string for the regime badge
    """
    regime_classes = {
        0: 'regime-euphoria',
        1: 'regime-complacency',
        2: 'regime-capitulation'
    }
    
    regime_names = {
        0: 'EUPHORIA',
        1: 'COMPLACENCY',
        2: 'CAPITULATION'
    }
    
    regime_class = regime_classes.get(regime_id, '')
    regime_name = regime_names.get(regime_id, 'UNKNOWN')
    
    confidence_text = f" ({confidence:.0%})" if confidence is not None else ""
    
    return f'<span class="regime-badge {regime_class}">{regime_name}{confidence_text}</span>'


def get_metric_card_html(value: str, label: str, delta: str = None, is_positive: bool = None) -> str:
    """
    Generate HTML for a metric card.
    
    Args:
        value: The metric value to display
        label: The metric label
        delta: Optional delta/change value
        is_positive: Whether the delta is positive (for coloring)
        
    Returns:
        HTML string for the metric card
    """
    card_class = "metric-card"
    if is_positive is not None:
        card_class += " positive" if is_positive else " negative"
    
    delta_html = ""
    if delta is not None:
        delta_class = "positive" if is_positive else "negative"
        delta_icon = "↑" if is_positive else "↓"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_icon} {delta}</div>'
    
    return f"""
    <div class="{card_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


def get_allocation_bar_html(spy_pct: float, tlt_pct: float, gld_pct: float) -> str:
    """
    Generate HTML for an allocation bar visualization.
    
    Args:
        spy_pct: SPY allocation percentage (0-1)
        tlt_pct: TLT allocation percentage (0-1)
        gld_pct: GLD allocation percentage (0-1)
        
    Returns:
        HTML string for the allocation bar
    """
    spy_width = spy_pct * 100
    tlt_width = tlt_pct * 100
    gld_width = gld_pct * 100
    
    return f"""
    <div class="allocation-bar">
        <div class="allocation-segment allocation-spy" style="width: {spy_width}%">
            SPY {spy_pct:.0%}
        </div>
        <div class="allocation-segment allocation-tlt" style="width: {tlt_width}%">
            TLT {tlt_pct:.0%}
        </div>
        <div class="allocation-segment allocation-gld" style="width: {gld_width}%">
            GLD {gld_pct:.0%}
        </div>
    </div>
    """


def get_confidence_meter_html(confidence: float) -> str:
    """
    Generate HTML for a confidence meter.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        HTML string for the confidence meter
    """
    fill_width = confidence * 100
    
    return f"""
    <div class="confidence-meter">
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {fill_width}%"></div>
        </div>
        <div class="confidence-value">{confidence:.0%}</div>
    </div>
    """


def get_hero_section_html(current_regime: int, confidence: float, last_update: str) -> str:
    """
    Generate HTML for the hero section.
    
    Args:
        current_regime: Current regime identifier
        confidence: Regime confidence score
        last_update: Last update timestamp string
        
    Returns:
        HTML string for the hero section
    """
    regime_badge = get_regime_badge_html(current_regime, confidence)
    
    return f"""
    <div class="hero-section">
        <div class="hero-title">CHRONOS</div>
        <div class="hero-subtitle">Adaptive Regime Intelligence Platform</div>
        <div style="margin-top: 1rem;">
            <span style="color: rgba(255,255,255,0.7); margin-right: 0.5rem;">Current Market Regime:</span>
            {regime_badge}
        </div>
        <div style="margin-top: 1rem; font-size: 0.85rem; color: rgba(255,255,255,0.6);">
            Last updated: {last_update}
        </div>
    </div>
    """
