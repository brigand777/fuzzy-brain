import altair as alt
import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta


def plot_portfolio_absolute_value(
    data, selected_assets, start, end, portfolio_df,
    x_axis_font_size=15,
    y_axis_font_size=15
    ):
    filtered_data = data[selected_assets].loc[start:end]
    amounts = portfolio_df.set_index("Asset").loc[selected_assets]["Amount"]
    dollar_values = filtered_data.multiply(amounts, axis=1)
    portfolio_value = dollar_values.sum(axis=1)

    df = portfolio_value.reset_index()
    df.columns = ["Date", "Portfolio Value"]

    # Area background
    area = alt.Chart(df).mark_area(
        color="#f5c518",
        opacity=0.25,
        interpolate="monotone"
    ).encode(
        x=alt.X("Date:T", axis=alt.Axis(title="Date", labelFontSize=x_axis_font_size)),
        y=alt.Y("Portfolio Value:Q", axis=alt.Axis(title="Value ($)", format="$,.0f", labelFontSize=y_axis_font_size))
    )

    # Line chart with tooltip
    line = alt.Chart(df).mark_line(
        color="#f5c518",
        strokeWidth=3
    ).encode(
        x="Date:T",
        y="Portfolio Value:Q",
        tooltip=[
            alt.Tooltip("Date:T", title="Date"),
            alt.Tooltip("Portfolio Value:Q", format="$,.2f", title="Value")
        ]
    )

    base_chart = area + line

    # Add interactivity (rule + selectors, no floating text)
    interactive_chart = add_interactivity(
        base_chart=base_chart,
        line_chart = line,
        df=df,
        x_field="Date",
        y_field="Portfolio Value"
    ).properties(
        width=1000,
        height=500,
        title="Portfolio Value Over Time"
    )

    return interactive_chart

# ----- Metric Calculation Function -----
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def calculate_portfolio_metrics(price_data: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    """
    Calculate portfolio-level performance metrics using actual asset weights.

    Parameters:
        price_data (pd.DataFrame): Price history of assets (columns = asset symbols).
        portfolio_df (pd.DataFrame): Portfolio holdings with columns ['Asset', 'Amount'].

    Returns:
        dict: Dictionary of portfolio metrics.
    """
    returns = price_data.pct_change().dropna()

    # --- Compute weights from portfolio holdings and latest prices ---
    latest_prices = price_data.iloc[-1]
    portfolio_df = portfolio_df.copy()

    portfolio_df["Price"] = portfolio_df["Asset"].map(latest_prices)
    portfolio_df["Value"] = portfolio_df["Amount"] * portfolio_df["Price"]
    total_value = portfolio_df["Value"].sum()

    weights = (portfolio_df.set_index("Asset")["Value"] / total_value).reindex(price_data.columns).fillna(0)

    # --- Portfolio returns and cumulative value over time ---
    portfolio_returns = returns.dot(weights)
    portfolio_value = (1 + portfolio_returns).cumprod()

    # --- Metrics ---
    cumulative_return = 100 * (portfolio_value.iloc[-1] - 1)
    volatility = 100 * portfolio_returns.std() * np.sqrt(365.0)
    sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
    
    # Max drawdown on portfolio value
    rolling_max = portfolio_value.cummax()
    drawdown = portfolio_value / rolling_max - 1
    max_drawdown = 100 * drawdown.min()

    # Calmar = annual return / abs(max drawdown)
    annual_return = portfolio_returns.mean() * 365
    calmar_ratio = 100 * (annual_return / abs(drawdown.min()))

    # Historical VaR (95%)
    var_95 = portfolio_returns.quantile(0.05)

    return {
        "Cumulative Returns": cumulative_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Value at Risk (95%)": abs(var_95) * 100  # as positive %
    }

import plotly.graph_objects as go

# ---- Single Gauge using Plotly ----
import plotly.graph_objects as go

def plot_single_gauge(title: str, value: float, metric_name: str = None, title_font_size: int = 18) -> go.Figure:
    label_to_metric = {
        "cumulative returns": "cumulative",
        "volatility": "volatility",
        "sharpe ratio": "sharpe",
        "max drawdown": "drawdown",
        "calmar ratio": "calmar",
        "value at risk (95%)": "var"
    }

    normalized_label = title.strip().lower()
    metric_key = metric_name.lower() if metric_name else label_to_metric.get(normalized_label, normalized_label)

    # Define which metrics should show %
    percent_metrics = {"cumulative", "volatility", "drawdown", "var"}

    # Metric-specific settings
    metric_settings = {
        "sharpe":     {"range": [-1, 3], "threshold": 1.5},
        "calmar":     {"range": [0, 5], "threshold": 2},
        "drawdown":   {"range": [-100, 0], "threshold": -60},
        "cumulative": {"range": [0, 100], "threshold": 10},
        "var":        {"range": [0, 20], "threshold": 5},
        "volatility": {"range": [0, 70], "threshold": 50}
    }

    settings = metric_settings.get(metric_key, {"range": [0, 100], "threshold": 50})
    min_val, max_val = settings["range"]
    threshold = settings["threshold"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            'suffix': "%" if metric_key in percent_metrics else "",
            'font': {'color': 'white'}
        },
        title={'text': title, 'font': {'size': title_font_size, 'color': 'white'}},
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickmode': 'array',
                'tickvals': [],  # 👈 no tick labels
                'tickwidth': 0
            },
            'bar': {'color': "lightgray", 'thickness': 0.35},
            'bgcolor': "black",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [],
            'threshold': {
                'line': {'color': "gray", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=200,
        margin=dict(t=25, b=25, l=20, r=20)
    )

    return fig


# ---- Layout for Multiple Gauges ----
def plot_gauge_charts(metrics: dict):
    return [plot_single_gauge(name, value) for name, value in metrics.items()]

# ----- Correlation Heatmap Plotting -----
import plotly.express as px
import numpy as np
import plotly.express as px
import numpy as np

def plot_correlation_heatmap(price_data: pd.DataFrame):
    returns = price_data.pct_change().dropna()
    corr = returns.corr()

    # Format to 2 decimal places (including the diagonal)
    corr_display = np.round(corr, 2)

    red_blue_scale = [
        [0.0, "rgba(0,0,120,1)"],
        [0.1, "rgba(30,30,160,1)"],
        [0.2, "rgba(70,70,200,1)"],
        [0.3, "rgba(120,120,240,1)"],
        [0.4, "rgba(200,200,255,1)"],
        [0.5, "rgba(255,255,255,1)"],
        [0.6, "rgba(255,200,200,1)"],
        [0.7, "rgba(240,120,120,1)"],
        [0.8, "rgba(200,70,70,1)"],
        [0.9, "rgba(160,30,30,1)"],
        [1.0, "rgba(120,0,0,1)"]
    ]

    fig = px.imshow(
        corr_display,
        text_auto=True,
        aspect="auto",
        title="📈 Correlation Heatmap of Portfolio Holdings",
        color_continuous_scale=red_blue_scale,
        zmin=-1, zmax=1,
        labels=dict(color="Correlation")
    )

    fig.update_layout(
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11))
    )

    return fig

# ----- Master Dashboard Plotter -----
from utils.metrics import calculate_portfolio_metrics  # Make sure this is imported

def plot_portfolio_dashboard(price_data: pd.DataFrame, selected_assets: list, portfolio_df: pd.DataFrame, date_range: tuple = None):
    if not selected_assets:
        return None, None

    asset_data = price_data[selected_assets]

    # Use custom date range if provided
    if date_range is None:
        end_date = asset_data.index.max()
        start_date = end_date - pd.Timedelta(days=100)
    else:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    start_date = pd.to_datetime(start_date).tz_localize("UTC") if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date).tz_localize("UTC") if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date)

    filtered_data = asset_data[(asset_data.index >= start_date) & (asset_data.index <= end_date)]

    if filtered_data.empty:
        return None, None

    # ✅ NEW: Portfolio-aware metrics using weighted returns
    metrics = calculate_portfolio_metrics(filtered_data, portfolio_df)

    # 🧭 Gauge charts
    needle_fig = plot_gauge_charts(metrics)

    # 🔥 Optional: Heatmap for diversification insight
    heatmap_fig = plot_correlation_heatmap(filtered_data)

    return needle_fig, heatmap_fig

def plot_historical_assets(data, selected_assets, portfolio_df=None):
    import altair as alt
    import streamlit as st
    import pandas as pd

    if not selected_assets:
        st.warning("No assets selected to plot.")
        return

    st.markdown("### 📊 Historical Asset Plots")

    # Time range input
    date_range = st.date_input(
        "Select Time Range",
        value=[data.index.min().date(), data.index.max().date()],
        min_value=data.index.min().date(),
        max_value=data.index.max().date()
    )

    if len(date_range) != 2:
        st.warning("Please select a valid time range.")
        return

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    if data.index.tz is not None:
        start_date = start_date.tz_localize(data.index.tz) if start_date.tzinfo is None else start_date.tz_convert(data.index.tz)
        end_date = end_date.tz_localize(data.index.tz) if end_date.tzinfo is None else end_date.tz_convert(data.index.tz)

    try:
        asset_data = data.loc[start_date:end_date, selected_assets]
    except Exception as e:
        st.error(f"Error loading selected assets: {e}")
        return

    portfolio_data = None
    if portfolio_df is not None and "Asset" in portfolio_df.columns:
        portfolio_assets = portfolio_df["Asset"].dropna().unique()
        portfolio_assets = [a for a in portfolio_assets if a in data.columns]

        if portfolio_assets:
            portfolio_data = data.loc[start_date:end_date, portfolio_assets]

    # Toggle checkboxes if portfolio is given
    if portfolio_data is not None:
        st.markdown("#### Select What to Plot:")
        col1, col2 = st.columns(2)
        with col1:
            show_assets = st.checkbox("Show Selected Assets", value=True)
        with col2:
            show_portfolio = st.checkbox("Show Portfolio", value=True)
    else:
        show_assets = True
        show_portfolio = False

    combined_data = pd.DataFrame(index=data.loc[start_date:end_date].index)

    if show_assets:
        combined_data = combined_data.join(asset_data, how="outer")

    if show_portfolio and portfolio_data is not None and not portfolio_df.empty:
        if "Amount" not in portfolio_df.columns:
            st.warning("Portfolio data does not include an 'Amount' column.")
        else:
            valid_assets = [a for a in portfolio_df["Asset"] if a in portfolio_data.columns]
            amounts = portfolio_df.set_index("Asset").loc[valid_assets, "Amount"].fillna(0)

            total_amount = amounts.sum()
            if total_amount == 0:
                st.warning("Total portfolio amount is zero. Cannot compute weights.")
            else:
                weights = amounts / total_amount
                portfolio_series = (portfolio_data[valid_assets] * weights).sum(axis=1)
                portfolio_series.name = "Portfolio"
                combined_data = combined_data.join(portfolio_series)

    combined_data = combined_data.dropna(how="all", axis=1)

    if combined_data.empty:
        st.warning("No data available to plot.")
        return

    # Create origin labels (for styling)
    origin_labels = []
    for col in combined_data.columns:
        if col == "Portfolio":
            origin_labels.append("Portfolio")
        else:
            origin_labels.append("Assets")

    column_metadata = pd.DataFrame({
        "Asset": combined_data.columns,
        "Origin": origin_labels
    })

    # --- Compute cumulative returns (full data)
    cumulative = combined_data.pct_change().fillna(0).add(1).cumprod()
    cumulative.index.name = "date"
    cumulative = cumulative.reset_index()

    # --- Then downsample for plotting
    num_days = max((end_date - start_date).days, 1)
    downsample_interval = max(1, num_days // 365)
    cumulative_downsampled = cumulative.iloc[::downsample_interval]

    # --- Melt and plot cumulative returns
    melted = cumulative_downsampled.melt(id_vars="date", var_name="Asset", value_name="Cumulative Return")
    merged = pd.merge(melted, column_metadata, on="Asset", how="left")

    st.markdown("#### Cumulative Returns")
    cum_chart = alt.Chart(merged).mark_line().encode(
        x="date:T",
        y=alt.Y("Cumulative Return:Q", title="Cumulative Return"),
        color="Asset:N",
        strokeDash=alt.condition(
            "datum.Origin === 'Portfolio'",
            alt.value([5, 5]),
            alt.value([1, 0])
        )
    ).properties(height=300)

    st.altair_chart(add_interactivity(cum_chart, "date", "Cumulative Return"), use_container_width=True)


    # Downsample
    num_days = max((end_date - start_date).days, 1)
    downsample_interval = max(1, num_days // 365)
    downsampled = combined_data.iloc[::downsample_interval]

    if not downsampled.empty:
        st.markdown("#### Daily Returns (Downsampled)")
        returns_chart = plot_asset_returns(downsampled, downsampled.columns.tolist())
        st.altair_chart(add_interactivity(returns_chart, "date", "Daily Return (%)"), use_container_width=True)

        st.markdown("#### Prices (Downsampled)")
        price_chart = plot_asset_prices(downsampled, downsampled.columns.tolist())
        st.altair_chart(add_interactivity(price_chart, "date", "Price"), use_container_width=True)
    else:
        st.warning("No data available in the selected range.")



def plot_portfolio_allocation_3d(portfolio_df: pd.DataFrame, title: str = "Portfolio Allocation") -> None:
    """
    Vertical bar chart showing asset allocations by value percentage.
    """
    if portfolio_df.empty or "Asset" not in portfolio_df.columns:
        st.warning("No portfolio data available to plot.")
        return

    df = portfolio_df.copy()
    if "Percent" not in df.columns:
        if "Amount" in df.columns and "Price" in df.columns:
            df["Value"] = df["Amount"] * df["Price"]
            total_value = df["Value"].sum()
            df["Percent"] = df["Value"] / total_value * 100 if total_value > 0 else 0
        elif "Amount" in df.columns:
            total_amount = df["Amount"].sum()
            df["Percent"] = df["Amount"] / total_amount * 100 if total_amount > 0 else 0
        else:
            st.warning("No allocation data available to plot.")
            return

    df = df[df["Percent"] > 0]
    if df.empty:
        st.warning("All portfolio holdings are 0.")
        return

    max_index = df["Percent"].idxmax()

    # Color palette
    base_color = "rgb(100, 143, 255)"
    highlight_color = "rgb(255, 140, 0)"
    bar_colors = [highlight_color if i == max_index else base_color for i in df.index]

    fig = go.Figure(data=[go.Bar(
        x=df["Asset"],
        y=df["Percent"],
        marker=dict(color=bar_colors),
        hovertemplate="%{x}: %{y:.2f}%"
    )])

    fig.update_layout(
        title=title,
        yaxis_title="Allocation (%)",
        xaxis_title="Asset",
        margin=dict(l=40, r=20, t=50, b=40),
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)
def add_interactivity(
    base_chart,
    x_field,
    y_field,
    df=None,
    tooltip_fields=None,
    rule_color="lightgray",
    rule_opacity=0.6,
    show_rule=True,
    show_text=False,
    text_dx=10,
    text_dy=-30,
    dot_color=None,
    line_chart=None  # ✅ optional dedicated line chart
    ):
    if df is None:
        df = getattr(base_chart, 'data', None)
        if df is None:
            raise TypeError("You must provide `df` or ensure `base_chart` has `.data` set.")

    nearest = alt.selection_single(
        fields=[x_field],
        nearest=True,
        on="mouseover",
        empty="none",
        clear="mouseout"
    )

    selectors = alt.Chart(df).mark_point(size=0, opacity=0).encode(
        x=x_field,
        y=y_field
    ).add_selection(nearest)

    rule = alt.Chart(df).mark_rule(
        strokeDash=[4, 4],
        color=rule_color
    ).encode(
        x=x_field,
        opacity=alt.condition(nearest, alt.value(rule_opacity), alt.value(0))
    ) if show_rule else alt.Chart(df)

    # 🟡 Resolve dot color
    color_encoding = getattr(base_chart.encoding, "color", None)
    line_color_attr = getattr(line_chart.mark, "color", None) if line_chart else None

    if dot_color:
        dot_color_expr = alt.value(dot_color)
        dot_kwargs = {'color': dot_color}
    elif color_encoding:
        dot_color_expr = color_encoding
        dot_kwargs = {}
    elif line_color_attr:
        dot_color_expr = alt.value(line_color_attr)
        dot_kwargs = {'color': line_color_attr}
    else:
        dot_color_expr = alt.value(rule_color)
        dot_kwargs = {'color': rule_color}

    dot = alt.Chart(df).mark_point(
        filled=True,
        size=60,
        **dot_kwargs
    ).encode(
        x=x_field,
        y=y_field,
        color=dot_color_expr,
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    if show_text:
        text = alt.Chart(df).mark_text(
            align="left",
            dx=text_dx,
            dy=text_dy,
            fontSize=13,
            fontWeight="bold",
            color="white"
        ).encode(
            x=x_field,
            y=y_field,
            text=alt.condition(
                nearest,
                alt.Text(f"{y_field}:Q", format="$,.2f"),
                alt.value("")
            )
        )
        overlay = selectors + rule + dot + text
    else:
        overlay = selectors + rule + dot

    if tooltip_fields:
        base_chart = base_chart.encode(
            tooltip=[
                alt.Tooltip(f"{f}:Q", format="$,.2f" if "value" in f.lower() else "") for f in tooltip_fields
            ]
        )

    return (base_chart + overlay).interactive()

def plot_cumulative_returns(results_dict):
    cumul_df_list = []
    for method, res in results_dict.items():
        temp = res["cumulative"].reset_index()
        temp.columns = ["date", "cumulative"]
        # Adjust cumulative returns to start at 0 (subtract 1)
        temp["cumulative"] = temp["cumulative"] - 1
        temp["Method"] = method
        cumul_df_list.append(temp)
    
    cumul_df = pd.concat(cumul_df_list)

    chart = alt.Chart(cumul_df).mark_line().encode(
        x="date:T",
        y=alt.Y("cumulative:Q", title="Cumulative Return (starting at 0)"),
        color="Method:N"
    ).properties(
        width=700,
        height=400,
        title="Cumulative Returns by Optimization Method"
    )

    return chart

def plot_allocations_per_method(allocations, method):
    # allocations: DataFrame with date as index and asset columns for one method.
    df_reset = allocations.reset_index()
    date_column = df_reset.columns[0]
    alloc_df = df_reset.melt(id_vars=date_column, var_name="Asset", value_name="Allocation")
    alloc_df = alloc_df.rename(columns={date_column: "date"})
    chart = alt.Chart(alloc_df).mark_line().encode(
        x="date:T",
        y=alt.Y("Allocation:Q", title="Allocation"),
        color="Asset:N",
        tooltip=["date:T", "Asset:N", alt.Tooltip("Allocation:Q", format=".2%")]
    ).properties(title=f"Asset Allocations Over Time ({method})", width=700, height=400)
    return chart

def pie_chart_allocation(initial_weights, method):
    # initial_weights: a pandas Series with asset weights.
    df = pd.DataFrame({'Asset': initial_weights.index, 'Weight': initial_weights.values})
    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Weight", type="quantitative", stack=True),
        color=alt.Color(field="Asset", type="nominal"),
        tooltip=[alt.Tooltip("Asset:N"), alt.Tooltip("Weight:Q", format=".2%")]
    ).properties(
        title=f"Initial Allocation ({method})",
        width=200,
        height=200
    )
    return chart
'''
def plot_asset_cumulative_returns(price_data: pd.DataFrame):
    """
    Compute and plot cumulative returns for each asset from price data.
    """
    # Step 1: Compute cumulative returns
    returns = price_data.pct_change().fillna(0)
    cumulative = (1 + returns).cumprod()

    # Step 2: Reset index and rename date column safely
    cumulative = cumulative.reset_index()

    # Ensure the first column is named 'date'
    if cumulative.columns[0] != "date":
        cumulative = cumulative.rename(columns={cumulative.columns[0]: "date"})

    # Step 3: Melt for Altair
    cumulative_melted = pd.melt(
        cumulative,
        id_vars="date",
        var_name="Asset",
        value_name="Cumulative Return"
    )

    # Step 4: Build chart
    chart = alt.Chart(cumulative_melted).mark_line().encode(
        x="date:T",
        y=alt.Y("Cumulative Return:Q", title="Cumulative Return"),
        color="Asset:N"
    ).properties(
        width=700,
        height=400,
        title="Cumulative Returns by Asset"
    )

    return chart
'''

def plot_asset_cumulative_returns(price_data: pd.DataFrame,
                                   selected_assets: list,
                                   portfolio_df: pd.DataFrame,
                                   benchmark: str = None,
                                   start=None,
                                   end=None):
    """
    Plot cumulative return of the portfolio (and optionally vs a benchmark) over a date range.
    """
    # --- Filter by date ---
    if start and end:
        price_data = price_data.loc[start:end]

    # --- Compute weights from portfolio_df ---
    latest_prices = price_data.iloc[-1]
    values = portfolio_df.apply(
        lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
    total_value = values.sum()
    weights = {
        row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
        for _, row in portfolio_df.iterrows()
        if row["Asset"] in price_data.columns
    }

    # --- Compute returns ---
    returns = price_data.pct_change().fillna(0)
    portfolio_returns = returns[list(weights.keys())].dot(pd.Series(weights))

    # --- Build cumulative return DataFrame ---
    cumulative_df = pd.DataFrame({
        "date": price_data.index,
        "Portfolio": (1 + portfolio_returns).cumprod()
    })

    if benchmark and benchmark in price_data.columns:
        benchmark_returns = returns[benchmark]
        cumulative_df[benchmark] = (1 + benchmark_returns).cumprod()

    # --- Melt for Altair ---
    melted = pd.melt(
        cumulative_df,
        id_vars="date",
        var_name="Asset",
        value_name="Cumulative Return"
    )

    chart = alt.Chart(melted).mark_line().encode(
        x="date:T",
        y=alt.Y("Cumulative Return:Q", title="Cumulative Return"),
        color="Asset:N"
    ).properties(
        width=700,
        height=400,
        title="Cumulative Return"
    )

    return chart

def plot_rolling_sharpe(results_dict):
    sharpe_df_list = []
    for method, res in results_dict.items():
        temp = res["rolling_sharpe"].reset_index()
        temp.columns = ["date", "rolling_sharpe"]
        temp["Method"] = method
        sharpe_df_list.append(temp)
    sharpe_df = pd.concat(sharpe_df_list)
    chart = alt.Chart(sharpe_df).mark_line().encode(
        x="date:T",
        y=alt.Y("rolling_sharpe:Q", title="Rolling Annualized Sharpe Ratio"),
        color="Method:N"
    ).properties(width=700, height=400, title="Rolling Annualized Sharpe Ratio")
    return chart

def plot_drawdowns(results_dict):
    drawdown_df_list = []
    for method, res in results_dict.items():
        temp = res["drawdowns"].reset_index()
        temp.columns = ["date", "drawdown"]
        temp["Method"] = method
        drawdown_df_list.append(temp)
    drawdown_df = pd.concat(drawdown_df_list)
    chart = alt.Chart(drawdown_df).mark_line().encode(
        x="date:T",
        y=alt.Y("drawdown:Q", title="Rolling Maximum Drawdown"),
        color="Method:N"
    ).properties(width=700, height=400, title="Rolling Maximum Drawdown")
    return chart

def plot_allocations(results_dict):
    alloc_df_list = []
    for method, res in results_dict.items():
        # Reset the index; this will add a column for the dates.
        df_reset = res["allocations"].reset_index()
        # Use the first column name as the date column.
        date_column = df_reset.columns[0]
        # Melt the DataFrame using that column as id_vars.
        alloc_df = df_reset.melt(id_vars=date_column, var_name="Asset", value_name="Allocation")
        # Rename the date column to "date" for consistency.
        alloc_df = alloc_df.rename(columns={date_column: "date"})
        alloc_df["Method"] = method
        alloc_df_list.append(alloc_df)
    alloc_df_all = pd.concat(alloc_df_list)
    alloc_df_all["Method_Asset"] = alloc_df_all["Method"] + " - " + alloc_df_all["Asset"]
    
    chart = alt.Chart(alloc_df_all).mark_line().encode(
        x="date:T",
        y=alt.Y("Allocation:Q", title="Asset Allocation (%)"),
        color=alt.Color("Method_Asset:N", title="Method - Asset")
    ).properties(width=700, height=400, title="Asset Allocation Over Time")
    return chart

def plot_asset_returns(simulation_data, selected_assets):
    # Filter data to selected assets and compute daily returns (%)
    data_filtered = simulation_data[selected_assets].copy()
    returns = data_filtered.pct_change() * 100
    returns = returns.reset_index().melt(id_vars="date", var_name="Asset", value_name="Daily Return (%)")
    chart = alt.Chart(returns).mark_line().encode(
         x="date:T",
         y=alt.Y("Daily Return (%):Q", title="Daily Return (%)"),
         color="Asset:N"
    ).properties(width=700, height=400, title="Daily Returns by Asset")
    return chart

def plot_asset_prices(simulation_data, selected_assets, log_scale=False):
    # Plot absolute prices for the selected assets.
    data_filtered = simulation_data[selected_assets].reset_index().melt(id_vars="date", var_name="Asset", value_name="Price")
    scale_type = "log" if log_scale else "linear"
    chart = alt.Chart(data_filtered).mark_line().encode(
         x="date:T",
         y=alt.Y("Price:Q", title="Price", scale=alt.Scale(type=scale_type)),
         color="Asset:N"
    ).properties(width=700, height=400, title="Asset Prices")
    return chart
