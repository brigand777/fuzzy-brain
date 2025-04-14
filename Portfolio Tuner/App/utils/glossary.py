import streamlit as st
import streamlit.components.v1 as components
from utils.plots import add_interactivity

def add_info_icon(term: str, short_description: str, glossary_url: str = None):
    """Displays a compact info icon with hover tooltip and optional glossary link."""
    tooltip_html = f"""
    <style>
    .tooltip {{
      position: relative;
      display: inline-block;
      cursor: help;
    }}

    .tooltip .tooltiptext {{
      visibility: hidden;
      width: 240px;
      background-color: #333;
      color: #fff;
      text-align: left;
      border-radius: 6px;
      padding: 8px;
      position: absolute;
      z-index: 1;
      top: -5px;
      left: 110%;
      opacity: 0;
      transition: opacity 0.3s;
      font-size: 0.85rem;
    }}

    .tooltip:hover .tooltiptext {{
      visibility: visible;
      opacity: 1;
    }}
    </style>

    <div class="tooltip">❓
      <span class="tooltiptext">
        <strong>{term}</strong><br>{short_description}
        {'<br><a href="' + glossary_url + '" target="_blank" style="color:#1F77B4;">Read more</a>' if glossary_url else ''}
      </span>
    </div>
    """
    components.html(tooltip_html, height=30)


def chart_with_tooltip(
    title: str,
    term: str,
    short_desc: str,
    chart_func,
    glossary_url: str = None,
    interactive: bool = False,
    x_field: str = None,
    y_field: str = None,
    *args, **kwargs
    ):

    # ✅ Show title using Streamlit styling
    col1, col2 = st.columns([0.97, 0.03])
    with col1:
        st.markdown(f"### {title}")
    tooltip_html = f"""
    <style>
    .tooltip-inline {{
      display: inline-block;
      position: relative;
      vertical-align: super;
      margin-left: 6px;
      font-size: 0.75em;  /* smaller icon */
      font-family: inherit;
      color: inherit;
    }}

    .tooltip-icon {{
      cursor: help;
      font-weight: bold;
      font-size: 1em;  /* matches superscript scale */
      line-height: 1;
      color: inherit;
    }}

    .tooltip-text-wrapper {{
      visibility: hidden;
      opacity: 0;
      width: 220px;
      background-color: #333;
      color: #fff;
      text-align: left;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 0.8em;  /* smaller tooltip text */
      position: absolute;
      z-index: 10;
      bottom: 125%;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: auto;
      transition: opacity 0.3s ease-in-out, visibility 0s linear 0.3s;
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
        <span class="tooltip-icon">❓</span>
        <div class="tooltip-text-wrapper">
          <strong>{term}</strong><br>{short_desc}
          {'<br><a href="' + glossary_url + '" target="_blank" style="color:#1F77B4;">Read more</a>' if glossary_url else ''}
        </div>
      </span>
    </h3>
    """

    st.markdown(tooltip_html, unsafe_allow_html=True)




    # ✅ Now draw chart
    chart = chart_func(*args, **kwargs)
    if interactive and x_field and y_field:
        chart = add_interactivity(chart, x_field=x_field, y_field=y_field)

    if "plotly" in str(type(chart)).lower():
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.altair_chart(chart, use_container_width=True)

def chart_with_tooltips(
    title: str,
    term: str,
    short_desc: str,
    chart_func,
    glossary_url: str = None,
    interactive: bool = False,
    x_field: str = None,
    y_field: str = None,
    *args, **kwargs
    ):

    # ✅ Show title using Streamlit styling
    col1, col2 = st.columns([0.97, 0.03])
    with col1:
        st.markdown(f"### {title}")
    tooltip_html = f"""
    <style>
    .tooltip-inline {{
      position: relative;
      display: inline-block;
    }}

    .tooltip-icon {{
      cursor: help;
      font-weight: bold;
      padding: 2px;
    }}

    .tooltip-text-wrapper {{
      visibility: hidden;
      opacity: 0;
      width: 250px;
      background-color: #333;
      color: #fff;
      text-align: left;
      border-radius: 6px;
      padding: 8px;
      position: absolute;
      z-index: 1;
      bottom: 125%;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: auto;
      
      /* This controls fade-out AND fade-in */
      transition: opacity 0.5s ease-in-out, visibility 0s linear 1s;
    }}

    .tooltip-inline:hover .tooltip-text-wrapper {{
      visibility: visible;
      opacity: 1;
      pointer-events: auto;

      /* Cancel the hide delay on hover */
      transition-delay: 0s, 0s;
    }}
    </style>

    <div class="tooltip-inline">
      <span class="tooltip-icon">❓</span>
      <div class="tooltip-text-wrapper">
        <strong>{term}</strong><br>{short_desc}
        {'<br><a href="' + glossary_url + '" target="_blank" style="color:#1F77B4;">Read more</a>' if glossary_url else ''}
      </div>
    </div>
    """


    with col2:
        st.markdown(tooltip_html, unsafe_allow_html=True)



    # ✅ Now draw chart
    chart = chart_func(*args, **kwargs)
    if interactive and x_field and y_field:
        chart = add_interactivity(chart, x_field=x_field, y_field=y_field)

    if "plotly" in str(type(chart)).lower():
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.altair_chart(chart, use_container_width=True)

def add_glossary_term(term, short, long, show_more=True):
    """Displays a glossary term with an expandable explanation."""
    st.markdown(f"### 🔹 {term}")
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

🔗 [Portfolio Optimizer – MVO Explained](https://portfoliooptimizer.io/docs/optimization/mvo/)"""
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

🔗 [QuantInsti – Monte Carlo Simulation in Finance](https://blog.quantinsti.com/monte-carlo-simulation-python/)"""
    },
    {
        "term": "Rolling Sharpe Ratio",
        "short": "Sharpe ratio over a moving window.",
        "long": """Rolling Sharpe Ratio tracks how a portfolio’s risk-adjusted return evolves over time. 
It’s calculated using a fixed window (e.g., 30 days) to capture temporal variation in performance and risk.

🔗 [Medium – Rolling Sharpe Ratio Explained](https://medium.com/@mddanishyusuf/rolling-sharpe-ratio-in-python-f2bfc39a3eb2)"""
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

🔗 [Morningstar – What Is Rebalancing?](https://www.morningstar.com/retirement/what-is-portfolio-rebalancing)"""
    },
    {
        "term": "Correlation Heatmap",
        "short": "Visualizes relationships between assets.",
        "long": """A correlation heatmap shows how assets move relative to each other. 
High positive correlation means assets move together; low or negative values suggest diversification potential.

🔗 [Towards Data Science – Correlation Heatmaps](https://towardsdatascience.com/correlation-heatmaps-in-python-558b96c2a4b8)"""
    }
]
