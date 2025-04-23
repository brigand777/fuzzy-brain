import streamlit as st
import streamlit.components.v1 as components
from utils.plots import add_interactivity

import base64

def inject_tooltip_css():
    tooltip_css = """
    <style>
    .info-tooltip {
      display: inline-block;
      position: relative;
      vertical-align: super;
      margin-left: 6px;
      font-size: 0.75em;
      font-family: inherit;
      color: inherit;
    }

    .info-tooltip-icon {
      cursor: help;
      font-weight: bold;
      font-size: 0.8em;
      line-height: 1;
      color: inherit;
    }

    .info-tooltip-box {
      visibility: hidden;
      opacity: 0;
      width: 220px;
      background-color: #FFF8DC; /* 🟡 vintage paper yellow */
      color: #000000;             /* black text */
      font-style: italic;         /* ✒️ handwritten look */
      font-family: 'Georgia', serif;  /* optional script-style font */
      text-align: left;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 0.7em;
      position: absolute;
      z-index: 10;
      bottom: 125%;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: auto;
      transition: opacity 0.3s ease-in-out, visibility 0s linear 0.3s;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }

    .info-tooltip:hover .info-tooltip-box {
      visibility: visible;
      opacity: 1;
      transition-delay: 0s, 0s;
    }
    </style>
    """
    st.markdown(tooltip_css, unsafe_allow_html=True)
def set_global_font_style():
    import streamlit as st

    st.markdown("""
    <style>
    html, body, .stApp {
        font-size: 16px !important;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .info-tooltip-box {
        font-size: 0.85em !important;
    }
    </style>
    """, unsafe_allow_html=True)

import streamlit as st
import streamlit.components.v1 as components
import base64

import streamlit as st
import streamlit.components.v1 as components
import base64

import base64

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def add_info_icon(term: str, short_description: str, glossary_url: str = None, mascot_path: str = "Portfolio Tuner/App/assets/headshot.png") -> str:
    mascot_b64 = image_to_base64(mascot_path)

    tooltip_content = f"<strong>{term}</strong><br>{short_description}" if term else short_description
    if glossary_url:
        tooltip_content += f'<br><a href="{glossary_url}" target="_blank" style="color:#1F77B4;">Read more</a>'

    return f"""
<style>
.info-tooltip {{
    position: relative;
    display: inline-block;
}}
.info-tooltip-icon {{
    cursor: pointer;
    font-size: 1rem;
    opacity: 0.7;
    margin-left: 4px;
    vertical-align: middle;
}}
.info-tooltip-box {{
    visibility: hidden;
    background-color: #FAF3D3;
    color: #1A1A1A;
    border: 1px solid #D6C899;
    border-radius: 8px;
    padding: 10px 10px 24px 10px;
    position: absolute;
    z-index: 9999;
    bottom: 120%;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.85rem;
    min-width: 220px;
    max-width: 300px;
    box-shadow: 2px 4px 12px rgba(0,0,0,0.3);
    opacity: 0;
    transition: opacity 0.3s ease;
    white-space: normal;
}}
.info-tooltip:hover .info-tooltip-box {{
    visibility: visible;
    opacity: 1;
}}
.info-tooltip-box img {{
    position: absolute;
    bottom: 6px;
    right: 6px;
    width: 18px;
    height: 18px;
    opacity: 0.4;
}}
</style>

<span class="info-tooltip">
    <span class="info-tooltip-icon">ℹ️</span>
    <div class="info-tooltip-box">
        {tooltip_content}
        <img src="data:image/png;base64,{mascot_b64}" alt="icon" />
    </div>
</span>
"""

# This is your info icon (uploaded) as base64
#info_icon_b64 = "iVBORw0KGgoAAAANSUhEUgAAAA8AAAAQCAYAAAC0tH7LAAA..."

def render_final_tooltip_heading(title: str, short_description: str, term: str = "", glossary_url: str = None):
    tooltip_html = f"<strong>{term}</strong><br>{short_description}" if term else short_description
    if glossary_url:
        tooltip_html += f'<br><a href="{glossary_url}" target="_blank" style="color:#1F77B4;">Read more</a>'

    html = f"""
    <style>
    .tooltip-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #F5F5F5;
    }}
    .info-tooltip {{
        position: relative;
        display: inline-block;
    }}
    .info-tooltip-icon {{
        width: 20px;
        height: 20px;
        opacity: 0.8;
        cursor: pointer;
    }}
    .info-tooltip-box {{
        visibility: hidden;
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #FAF3D3;
        color: #1A1A1A;
        border: 1px solid #D6C899;
        border-radius: 8px;
        padding: 10px 10px 24px 10px;
        box-shadow: 2px 4px 12px rgba(0,0,0,0.3);
        font-size: 0.85rem;
        z-index: 9999;
        min-width: 220px;
        max-width: 300px;
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    .info-tooltip:hover .info-tooltip-box {{
        visibility: visible;
        opacity: 1;
    }}
    .info-tooltip-box img {{
        position: absolute;
        bottom: 6px;
        right: 6px;
        width: 18px;
        height: 18px;
        opacity: 0.4;
    }}
    </style>

    <div class="tooltip-header">
        {title}
        <span class="info-tooltip">
            <img src="data:image/png;base64,{info_icon_b64}" class="info-tooltip-icon" />
            <div class="info-tooltip-box">
                {tooltip_html}
                <img src="data:image/png;base64,{info_icon_b64}" alt="icon"/>
            </div>
        </span>
    </div>
    """

    components.html(html, height=100)

import base64

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def section_heading(title, short_description, term=None, glossary_url=None, level=2, mascot_path="Portfolio Tuner/App/assets/headshot.png"):
    mascot_b64 = image_to_base64(mascot_path)

    if term:
        tooltip_html = f"<strong>{term}</strong><br>{short_description}"
    else:
        tooltip_html = short_description

    if glossary_url:
        tooltip_html += f'<br><a href="{glossary_url}" target="_blank" style="color:#1F77B4;">Read more</a>'

    tag = f"h{level}"

    html = f"""
<style>
/* Tooltip styles */
.info-tooltip {{
    position: relative;
    display: inline-block;
    line-height: 1;
}}

.info-tooltip-icon {{
    cursor: pointer;
    font-size: 1rem;
    opacity: 1;
    margin-left: 0;
    vertical-align: middle;
}}

.info-tooltip-box {{
    visibility: hidden;
    background-color: #FAF3D3;
    color: #1A1A1A;
    border: 1px solid #D6C899;
    border-radius: 8px;
    padding: 10px 10px 24px 10px;
    position: absolute;
    z-index: 9999;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.85rem;
    min-width: 220px;
    max-width: 300px;
    box-shadow: 2px 4px 12px rgba(0,0,0,0.3);
    opacity: 0;
    transition: opacity 0.3s ease;
}}

.info-tooltip:hover .info-tooltip-box {{
    visibility: visible;
    opacity: 1;
}}

.info-tooltip-box img {{
    position: absolute;
    bottom: 6px;
    right: 6px;
    width: 28px;
    height: 28px;
    opacity: 1;
}}
</style>

<div style="display: inline-flex; align-items: center; gap: 0;">
    <{tag} style="margin: 0;">{title}</{tag}>
    <span class="info-tooltip">
        <span class="info-tooltip-icon">ℹ️</span>
        <div class="info-tooltip-box">
            {tooltip_html}
            <img src="data:image/png;base64,{mascot_b64}" alt="icon" />
        </div>
    </span>
</div>
"""
    return html

def vintage_dropdown(title: str, content: str):
    """Displays a dropdown styled like a vintage tooltip: yellow background, italic serif font."""
    styled_html = f"""
    <style>
    .vintage-box {{
        background-color: #FFF8DC;
        color: #000;
        font-style: italic;
        font-family: 'Lora', serif;
        border-radius: 6px;
        padding: 12px;
        font-size: 0.95em;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }}
    </style>
    <div class="vintage-box">{content}</div>
    """
    with st.expander(title):
        st.markdown(styled_html, unsafe_allow_html=True)


def chart_with_tooltip(
    title: str,
    short_desc: str,
    chart_func,
    term: str = "",
    glossary_url: str = None,
    interactive: bool = False,
    x_field: str = None,
    y_field: str = None,
    *args, **kwargs
    ):
    import streamlit as st
    from utils.plots import add_interactivity

    # --- Tooltip content handling ---
    tooltip_content = f"<strong>{term}</strong><br>{short_desc}" if term else short_desc
    if glossary_url:
        tooltip_content += f'<br><a href="{glossary_url}" target="_blank" style="color:#1F77B4;">Read more</a>'

    tooltip_html = f"""
    <style>
    .tooltip-inline {{
      display: inline-block;
      position: relative;
      vertical-align: super;
      margin-left: 6px;
      font-size: 0.75em;
      font-family: inherit;
      color: inherit;
    }}

    .tooltip-icon {{
      cursor: help;
      font-weight: bold;
      font-size: 0.7em;
      line-height: 1;
      color: inherit;
    }}

    .tooltip-text-wrapper {{
      visibility: hidden;
      opacity: 0;
      width: 220px;
      background-color: #FFF8DC;
      color: #000;
      font-style: italic;
      font-family: 'Merriweather', serif;
      text-align: left;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 0.65em;
      position: absolute;
      z-index: 10;
      bottom: 125%;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: auto;
      transition: opacity 0.3s ease-in-out, visibility 0s linear 0.3s;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }}

    .tooltip-inline:hover .tooltip-text-wrapper {{
      visibility: visible;
      opacity: 1;
      transition-delay: 0s, 0s;
    }}
    </style>

    <h3 style="margin-bottom: 0.5rem;">
      <span>{title}</span>
      <span class="tooltip-inline">
        <span class="tooltip-icon">ℹ️</span>
        <div class="tooltip-text-wrapper">
          {tooltip_content}
        </div>
      </span>
    </h3>
    """

    st.markdown(tooltip_html, unsafe_allow_html=True)

    # ✅ Draw chart
    chart = chart_func(*args, **kwargs)
    if interactive and x_field and y_field:
        chart = add_interactivity(chart, x_field=x_field, y_field=y_field)

    if "plotly" in str(type(chart)).lower():
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.altair_chart(chart, use_container_width=True)

def add_glossary_term(term, short, long, show_more=True):
    """Displays a glossary term with an anchor link for navigation."""
    anchor = term.lower().replace(" ", "-")  # e.g. "Sharpe Ratio" → "sharpe-ratio"
    
    st.markdown(f'<h3 id="{anchor}">🔹 {term}</h3>', unsafe_allow_html=True)
    st.markdown(f"*{short}*")
    
    if show_more:
        with st.expander("Show more"):
            st.markdown(long)


# Optional: Store glossary items in a list for looping
GLOSSARY_TERMS = [
    {
        "term": "Sharpe Ratio",
        "short": "Measures return relative to risk.",
        "long": """The Sharpe Ratio is a risk-adjusted return metric calculated as the portfolio's excess return (over the risk-free rate) divided by its standard deviation. 
A higher Sharpe Ratio indicates more return per unit of risk — typically values >1.0 are considered good.

🔗 [Investopedia – Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)"""
    },
    {
        "term": "Volatility",
        "short": "How much an asset's price moves.",
        "long": """Volatility represents the degree of variation in asset prices over time, usually measured by standard deviation. 
High volatility means prices swing more dramatically, indicating higher risk and uncertainty.

🔗 [Investopedia – Volatility](https://www.investopedia.com/terms/v/volatility.asp)"""
    },
    {
        "term": "Max Drawdown",
        "short": "Largest portfolio drop from peak to trough.",
        "long": """Max Drawdown quantifies the largest observed loss from a peak to a trough in portfolio value, before a new high is achieved. 
It’s a key risk metric to understand the worst-case scenario over a given time period.

🔗 [Investopedia – Drawdown](https://www.investopedia.com/terms/d/drawdown.asp)"""
    },
    {
        "term": "Mean-Variance Optimization (MVO)",
        "short": "Classic method to find best risk-adjusted portfolio.",
        "long": """MVO is based on Modern Portfolio Theory and seeks to construct a portfolio with the highest expected return for a given level of risk (or the lowest risk for a given return). 
It relies heavily on historical means and covariances, and can be sensitive to estimation errors.

🔗 [Medium – MVO Explained](https://medium.com/@everettminshall/mean-variance-optimization-a-beginners-guide-dd1a9ddda758)"""
    },
    {
        "term": "Hierarchical Risk Parity (HRP)",
        "short": "Diversifies using hierarchical clustering of assets.",
        "long": """HRP avoids inverting the covariance matrix (unlike MVO) by first clustering assets by similarity and then allocating weights recursively based on intra-cluster variances. 
It offers better out-of-sample stability and is less sensitive to noisy data.

🔗 [HRP Paper by Marcos López de Prado (PDF)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)"""
    },
    {
        "term": "Monte Carlo Simulation",
        "short": "Simulates many future price paths using randomness.",
        "long": """Monte Carlo simulation generates thousands of potential future portfolio outcomes based on statistical properties (e.g., volatility, distribution shape). 
It helps estimate the range of potential returns and assess risk under uncertainty.

🔗 [Investopedia – Monte Carlo Simulation in Finance](https://www.investopedia.com/terms/m/montecarlosimulation.asp)"""
    },
    {
        "term": "Rolling Sharpe Ratio",
        "short": "Sharpe ratio over a moving window.",
        "long": """Rolling Sharpe Ratio tracks how a portfolio’s risk-adjusted return evolves over time. 
It’s calculated using a fixed window (e.g., 30 days) to capture temporal variation in performance and risk.

🔗 [QuantStart – Rolling Sharpe Ratio Explained](https://www.quantstart.com/articles/annualised-rolling-sharpe-ratio-in-qstrader/)"""
    },
    {
        "term": "Backtest",
        "short": "Testing a strategy on historical data.",
        "long": """Backtesting evaluates how a strategy would have performed in the past using historical price data. 
It helps determine robustness, drawdowns, and risk metrics before applying the strategy live.

🔗 [Investopedia – Backtesting](https://www.investopedia.com/terms/b/backtesting.asp)"""
    },
    {
        "term": "Rebalancing",
        "short": "Adjusting portfolio to target weights.",
        "long": """Rebalancing is the process of periodically adjusting the portfolio to match the target asset weights. 
It ensures risk exposure stays consistent but may incur transaction costs.

🔗 [Vanguard – What Is Rebalancing?](https://investor.vanguard.com/investor-resources-education/portfolio-management/rebalancing-your-portfolio)"""
    },
    {
        "term": "Correlation Heatmap",
        "short": "Visualizes relationships between assets.",
        "long": """A correlation heatmap shows how assets move relative to each other. 
High positive correlation means assets move together; low or negative values suggest diversification potential.

🔗 [Seaborn – Correlation Heatmaps](https://investor.vanguard.com/investor-resources-education/portfolio-management/rebalancing-your-portfolio)"""
    },
    {
    "term": "Benchmark Portfolio",
    "short": "Reference portfolio used for performance comparison.",
    "long": """A benchmark portfolio is a predefined set of assets used to evaluate the performance of an investment strategy or portfolio. 
In crypto, common benchmarks include BTC, ETH, or a market-cap weighted index of top coins.

Comparing to a benchmark helps assess whether active strategies are adding value or underperforming a passive approach.

🔗 [Investopedia – Benchmark Definition](https://www.investopedia.com/terms/b/benchmark.asp)"""
}

]
