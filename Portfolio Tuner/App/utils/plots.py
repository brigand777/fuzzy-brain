import altair as alt
import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta

# ----- Metric Calculation Function -----
def calculate_portfolio_metrics(price_data: pd.DataFrame) -> dict:
    returns = price_data.pct_change().dropna()
    cumulative_returns = (1 + returns).prod() - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = (price_data / price_data.cummax() - 1).min()
    rolling_max = price_data.cummax()
    drawdown = price_data / rolling_max - 1
    calmar_ratio = returns.mean() * 252 / abs(drawdown.min())
    var_95 = returns.quantile(0.05)

    return {
        "Cumulative Returns": cumulative_returns.mean(),
        "Volatility": volatility.mean(),
        "Sharpe Ratio": sharpe_ratio.mean(),
        "Max Drawdown": max_drawdown.mean(),
        "Calmar Ratio": calmar_ratio.mean(),
        "Value at Risk (95%)": var_95.mean()
    }

import plotly.graph_objects as go

# ---- Single Gauge using Plotly ----
def plot_single_gauge(title: str, value: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,  # Convert to percentage
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "black", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#00ff00'},
                {'range': [33, 66], 'color': '#ffff00'},
                {'range': [66, 100], 'color': '#ff0000'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value * 100
            }
        }
    ))
    return fig

# ---- Layout for Multiple Gauges ----
def plot_gauge_charts(metrics: dict):
    figs = []
    for title, value in metrics.items():
        figs.append(plot_single_gauge(title, value))
    return figs


# ----- Correlation Heatmap Plotting -----
def plot_correlation_heatmap(price_data: pd.DataFrame):
    returns = price_data.pct_change().dropna()
    corr = returns.corr()

    # Sophisticated single-color red scale
    red_blue_scale = [
    [0.0, "rgba(0,0,120,1)"],        # deep blue (strong negative correlation)
    [0.1, "rgba(30,30,160,1)"],      # darkish blue
    [0.2, "rgba(70,70,200,1)"],      # medium blue
    [0.3, "rgba(120,120,240,1)"],    # pale blue
    [0.4, "rgba(200,200,255,1)"],    # very pale blue
    [0.5, "rgba(255,255,255,1)"],    # white (neutral correlation)
    [0.6, "rgba(255,200,200,1)"],    # pale red
    [0.7, "rgba(240,120,120,1)"],    # salmon
    [0.8, "rgba(200,70,70,1)"],      # medium red
    [0.9, "rgba(160,30,30,1)"],      # blood red
    [1.0, "rgba(120,0,0,1)"],        # dark red (strong positive correlation)
]



    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="📈 Correlation Heatmap of Portfolio Holdings",
        color_continuous_scale=red_blue_scale,
        zmin=-1, zmax=1,
        labels=dict(color="Correlation")
    )
    fig.update_layout(margin=dict(t=50, b=30))
    return fig


# ----- Master Dashboard Plotter -----
def plot_portfolio_dashboard(price_data: pd.DataFrame, selected_assets: list, date_range: tuple = None):
    if not selected_assets:
        return None, None

    asset_data = price_data[selected_assets]

    # Default to last 100 days
    if date_range is None:
        end_date = asset_data.index.max()
        start_date = end_date - timedelta(days=100)
    else:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    
    start_date = pd.to_datetime(start_date).tz_localize("UTC") if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date).tz_localize("UTC") if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date)

    filtered_data = asset_data[(asset_data.index >= start_date) & (asset_data.index <= end_date)]

    if filtered_data.empty:
        return None, None

    metrics = calculate_portfolio_metrics(filtered_data)
    needle_fig = plot_gauge_charts(metrics)  # just collects figures
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


import plotly.graph_objects as go
import pandas as pd
import streamlit as st

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
    tooltip_field=None, 
    rule_color="gray", 
    rule_opacity=0.5,
    text_dx=5,
    text_dy=-5
):
    """
    Enhance a base Altair chart with interactive features.
    
    Parameters:
        base_chart (alt.Chart): The base Altair chart.
        x_field (str): The field name used for the x-axis (typically a date field).
        y_field (str): The field name used for the y-axis.
        tooltip_field (str or list, optional): Field(s) to display as tooltip on hover.
            If a list, the fields are concatenated with a separator.
        rule_color (str): Color of the vertical rule.
        rule_opacity (float): Opacity of the vertical rule when active.
        text_dx (int): Horizontal offset for the tooltip text.
        text_dy (int): Vertical offset for the tooltip text.
    
    Returns:
        alt.LayerChart: The interactive chart with a vertical rule and tooltips.
    """
    # Create a nearest selection based on the x_field.
    nearest = alt.selection_single(fields=[x_field], nearest=True, on="mouseover", empty="none")
    
    # Create transparent selectors across the chart to capture the mouse position.
    selectors = base_chart.mark_point(size=0).encode(
        opacity=alt.value(0)
    ).add_selection(nearest)
    
    # Create a vertical rule that highlights the x position.
    rule = base_chart.mark_rule(color=rule_color).encode(
        opacity=alt.condition(nearest, alt.value(rule_opacity), alt.value(0))
    )
    
    # Prepare tooltip text.
    if tooltip_field is not None:
        # If tooltip_field is a list, create a concatenated field.
        if isinstance(tooltip_field, list):
            tooltip_expr = alt.expr.concat(*[f"datum.{field} + ' '" for field in tooltip_field])
        else:
            tooltip_expr = f"datum.{tooltip_field}"
            
        text = base_chart.mark_text(align="left", dx=text_dx, dy=text_dy).encode(
            text=alt.condition(nearest, alt.ExprRef(expr=tooltip_expr), alt.value(''))
        )
        interactive_chart = alt.layer(base_chart, selectors, rule, text).interactive()
    else:
        interactive_chart = alt.layer(base_chart, selectors, rule).interactive()
    
    return interactive_chart

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
