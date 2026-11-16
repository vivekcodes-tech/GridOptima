import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator import GridEnvironment
from optimizers import (
    BatterySpec,
    run_baseline,
    run_greedy,
    run_dynamic_programming,
)

st.set_page_config(
    page_title="GridOptima | Algorithmic Energy Management",
    layout="wide",
)

col_title, col_toggle = st.columns([5, 1])
with col_title:
    st.title("GridOptima: Microgrid Energy Management & Scheduling")
    st.caption(
        "Comparing Greedy Heuristics vs. Bellman Dynamic Programming "
        "under Time-of-Use electricity tariffs."
    )
with col_toggle:
    st.write("")
    dark_mode = st.toggle("Dark Mode", value=False)

if dark_mode:
    chart_theme = "plotly_dark"
    bg_canvas = "#000000"
    grid_color = "#222222"
    axis_color = "#E0E0E0"
    theme_css = """
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
    }
    .stApp {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown span {
        color: #FFFFFF !important;
    }

    div[data-testid="stExpander"],
    details[data-testid="stExpander"],
    details[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] > details,
    div[data-testid="stExpander"] > details > summary {
        background-color: #101010 !important;
        color: #FFFFFF !important;
        border-color: #262626 !important;
    }
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] summary svg {
        fill: #FFFFFF !important;
    }

    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] *,
    label[data-baseweb="label"],
    .stSlider label,
    .stSlider label *,
    .stSelectbox label,
    .stSelectbox label *,
    .stNumberInput label,
    .stNumberInput label *,
    .stCheckbox label,
    .stCheckbox label * {
        color: #E0E0E0 !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="base-input"] input,
    div[data-baseweb="input"] input,
    .stNumberInput input {
        background-color: #181818 !important;
        color: #FFFFFF !important;
        border-color: #333333 !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        background-color: #242424 !important;
        color: #FFFFFF !important;
        border-color: #383838 !important;
    }
    button[data-testid="stNumberInputStepDown"]:hover,
    button[data-testid="stNumberInputStepUp"]:hover {
        background-color: #333333 !important;
    }
    button[data-testid="stNumberInputStepDown"] svg,
    button[data-testid="stNumberInputStepUp"] svg {
        fill: #FFFFFF !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #181818 !important;
        color: #FFFFFF !important;
        border-color: #333333 !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #FFFFFF !important;
    }
    div[data-baseweb="select"] svg {
        fill: #E0E0E0 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #101010 !important;
        border: 1px solid #262626 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: #A0A0A0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    div[data-testid="stAlert"],
    div[data-testid="stAlert"] > div {
        background-color: #0E1626 !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 8px !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] * {
        color: #93C5FD !important;
    }
    div[data-testid="stAlert"] svg {
        fill: #60A5FA !important;
    }

    button[data-baseweb="tab"] {
        color: #A0A0A0 !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #1E88E5 !important;
    }
    div[data-baseweb="tab-border"] {
        background-color: #262626 !important;
    }

    div.stButton > button, div.stDownloadButton > button {
        background-color: #181818 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #282828 !important;
        border-color: #555555 !important;
        color: #FFFFFF !important;
    }

    table.benchmark-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #101010 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid #262626 !important;
        margin: 10px 0 !important;
    }
    table.benchmark-table th {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid #333333 !important;
        text-align: left !important;
        font-weight: 600 !important;
    }
    table.benchmark-table td {
        padding: 10px 16px !important;
        border-bottom: 1px solid #222222 !important;
        color: #E0E0E0 !important;
    }

    div[data-testid="stDataFrame"] {
        filter: invert(0.92) hue-rotate(180deg) !important;
        border-radius: 8px !important;
    }
    </style>
    """
else:
    chart_theme = "plotly_white"
    bg_canvas = "#FFFFFF"
    grid_color = "#E2E8F0"
    axis_color = "#334155"
    theme_css = """
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    div[data-testid="stMetric"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
    }
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 6px !important;
    }
    table.benchmark-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid #E2E8F0 !important;
        margin: 10px 0 !important;
    }
    table.benchmark-table th {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid #E2E8F0 !important;
        text-align: left !important;
        font-weight: 600 !important;
    }
    table.benchmark-table td {
        padding: 10px 16px !important;
        border-bottom: 1px solid #F1F5F9 !important;
        color: #334155 !important;
    }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

with st.expander(
    "System Parameters and Input Controls (Expand / Collapse)",
    expanded=True,
):
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.markdown("**1. Load and Solar PV**")
        base_load = st.number_input(
            "Base Load (kWh/h)",
            min_value=10.0,
            max_value=80.0,
            value=32.0,
            step=1.0,
        )
        pv_capacity = st.number_input(
            "Solar PV Peak (kW)",
            min_value=0.0,
            max_value=150.0,
            value=48.0,
            step=2.0,
        )

    with col_p2:
        st.markdown("**2. Battery Storage System**")
        battery_capacity = st.number_input(
            "Battery Capacity (kWh)",
            min_value=10.0,
            max_value=200.0,
            value=60.0,
            step=5.0,
        )
        max_power = st.number_input(
            "Inverter Power Limit (kW)",
            min_value=5.0,
            max_value=60.0,
            value=20.0,
            step=1.0,
        )
        efficiency = st.slider(
            "Roundtrip Efficiency (η)",
            min_value=0.75,
            max_value=0.98,
            value=0.92,
            step=0.01,
        )

    with col_p3:
        st.markdown("**3. Algorithm and Grid Settings**")
        initial_soc = (
            st.slider(
                "Initial SoC (%)",
                min_value=10,
                max_value=90,
                value=30,
                step=5,
            )
            / 100.0
        )
        num_states = st.selectbox(
            "DP State Discretization (S)",
            options=[21, 31, 51, 81, 101, 151],
            index=4,
        )
        enforce_cyclic = st.checkbox(
            "Enforce Cyclic Boundary (SoC_end >= SoC_start)",
            value=True,
        )

env = GridEnvironment(seed=42)
profile = env.get_24h_profile(
    base_load_kwh=base_load,
    pv_capacity_kw=pv_capacity,
)
hours = profile["hours"]

battery = BatterySpec(
    capacity_kwh=battery_capacity,
    max_power_kw=max_power,
    roundtrip_efficiency=efficiency,
    min_soc=0.10,
    max_soc=0.95,
    initial_soc=initial_soc,
)

res_base = run_baseline(
    profile["demand"],
    profile["solar"],
    profile["tariffs"],
    profile["feed_in"],
)
res_greedy = run_greedy(
    profile["demand"],
    profile["solar"],
    profile["tariffs"],
    profile["feed_in"],
    battery,
)
res_dp = run_dynamic_programming(
    profile["demand"],
    profile["solar"],
    profile["tariffs"],
    profile["feed_in"],
    battery,
    num_states=num_states,
    enforce_cyclic=enforce_cyclic,
)

base_cost = res_base["total_cost"]
greedy_cost = res_greedy["total_cost"]
dp_cost = res_dp["total_cost"]

greedy_pct = (
    ((base_cost - greedy_cost) / base_cost) * 100.0 if base_cost > 0 else 0.0
)
dp_pct = (
    ((base_cost - dp_cost) / base_cost) * 100.0 if base_cost > 0 else 0.0
)
total_evals = res_dp["num_evaluations"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline (No Storage)", f"Rs.{base_cost:,.2f}")
col2.metric(
    "Greedy Heuristic",
    f"Rs.{greedy_cost:,.2f}",
    f"-{greedy_pct:.2f}%",
    delta_color="inverse",
)
col3.metric(
    "Dynamic Programming",
    f"Rs.{dp_cost:,.2f}",
    f"-{dp_pct:.2f}%",
    delta_color="inverse",
)
col4.metric(
    "Subproblems Evaluated",
    f"{total_evals:,d}",
    f"{res_dp['solve_time_ms']:.2f} ms",
)

st.write("")

with st.expander(
    "Problem Formulation & Operating Rules (Expand / Collapse)",
    expanded=False,
):
    st.markdown(r"""
    **Objective:** Minimize 24-hour net electricity bill:
    $$\text{Daily Bill} = \sum_{t=0}^{23} \Big[ (\text{Grid Import}_t \times \text{Tariff}_t) - (\text{Solar Export}_t \times \text{FeedIn}) \Big]$$

    **Time-of-Use Tariffs:**
    * Off-Peak (Night, 00:00–06:00, 23:00–24:00): Rs. 4.50 / unit
    * Standard (Daytime, 06:00–17:00, 22:00–23:00): Rs. 7.50 / unit
    * Peak (Evening, 17:00–22:00): Rs. 13.00 / unit
    * Solar Feed-In Compensation: Rs. 3.80 / unit

    **Physical Constraints:**
    1. Inverter Speed Limit: Battery power $|P_{\text{bat}}| \le P_{\text{max}}$ (e.g. 20 kW).
    2. Battery Safety Bounds: Keep SoC between 10% reserve and 95% maximum.
    3. Day-to-Day Rule (Cyclic): $\text{SoC}(24) \ge \text{SoC}(0)$ ensures tomorrow starts prepared.
    """)

st.subheader("1. 24-Hour Microgrid Load, Solar PV & Time-of-Use Tariff")

fig_dispatch = make_subplots(specs=[[{"secondary_y": True}]])
fig_dispatch.add_trace(
    go.Scatter(
        x=hours,
        y=profile["demand"],
        name="Demand (kWh)",
        line=dict(color="#E74C3C", width=2.5),
    ),
    secondary_y=False,
)
fig_dispatch.add_trace(
    go.Scatter(
        x=hours,
        y=profile["solar"],
        name="Solar Generation (kWh)",
        line=dict(color="#27AE60", width=2.5, dash="dash"),
    ),
    secondary_y=False,
)
fig_dispatch.add_trace(
    go.Scatter(
        x=hours,
        y=profile["tariffs"],
        name="Tariff (INR/kWh)",
        line=dict(color="#F39C12", width=2.2, dash="dot"),
    ),
    secondary_y=True,
)

fig_dispatch.update_layout(
    xaxis=dict(
        title="Hour of Day (00:00 - 23:00)",
        tickmode="linear",
        dtick=1,
        gridcolor=grid_color,
        color=axis_color,
    ),
    yaxis=dict(
        title="Energy (kWh)",
        gridcolor=grid_color,
        color=axis_color,
    ),
    yaxis2=dict(
        title="Tariff Rate (INR / kWh)",
        overlaying="y",
        side="right",
        gridcolor=grid_color,
        color=axis_color,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color=axis_color),
    ),
    margin=dict(l=40, r=40, t=30, b=40),
    template=chart_theme,
    paper_bgcolor=bg_canvas,
    plot_bgcolor=bg_canvas,
    height=360,
)
st.plotly_chart(fig_dispatch, theme=None, width="stretch")

st.subheader("2. Algorithmic Evaluation and Dynamic Programming Table")

tab1, tab2, tab3, tab4 = st.tabs([
    "Battery SoC Trajectory",
    "Hourly Grid Import",
    "Cumulative Cost Curve",
    "DP Tabulation Matrix V[t, s]",
])

with tab1:
    fig_soc = go.Figure()
    fig_soc.add_trace(
        go.Scatter(
            x=hours,
            y=res_greedy["battery_soc"] * 100,
            name="Greedy Heuristic",
            line=dict(color="#1E88E5", width=2.5),
        )
    )
    fig_soc.add_trace(
        go.Scatter(
            x=hours,
            y=res_dp["battery_soc"] * 100,
            name="Dynamic Programming (Optimal)",
            line=dict(color="#27AE60", width=3),
        )
    )
    fig_soc.add_hline(
        y=battery.max_soc * 100,
        line_dash="dash",
        line_color="#C0392B",
        annotation_text=f"Max Limit ({int(battery.max_soc*100)}%)",
        annotation_font_color=axis_color,
    )
    fig_soc.add_hline(
        y=battery.min_soc * 100,
        line_dash="dash",
        line_color="#E67E22",
        annotation_text=f"Min Reserve ({int(battery.min_soc*100)}%)",
        annotation_font_color=axis_color,
    )

    fig_soc.update_layout(
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear",
            dtick=1,
            gridcolor=grid_color,
            color=axis_color,
        ),
        yaxis=dict(
            title="State of Charge (%)",
            range=[0, 105],
            gridcolor=grid_color,
            color=axis_color,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=axis_color),
        ),
        template=chart_theme,
        paper_bgcolor=bg_canvas,
        plot_bgcolor=bg_canvas,
        height=380,
    )
    st.plotly_chart(fig_soc, theme=None, width="stretch")
    st.info(
        "Why Greedy Loses: At 14:00, the tariff is normal (Rs. 7.50). Greedy immediately discharges the battery to save Rs. 7.50, "
        "and runs out of energy. Between 17:00 and 22:00, the tariff spikes to Rs. 13.00. Dynamic Programming plans ahead, "
        "holds the stored solar power, and discharges during the expensive Rs. 13.00 window for maximum savings."
    )

with tab2:
    fig_grid = go.Figure()
    fig_grid.add_trace(
        go.Bar(
            x=hours,
            y=res_base["grid_import"],
            name="Baseline",
            marker_color="#888888" if dark_mode else "#BDC3C7",
        )
    )
    fig_grid.add_trace(
        go.Bar(
            x=hours,
            y=res_greedy["grid_import"],
            name="Greedy",
            marker_color="#1E88E5",
        )
    )
    fig_grid.add_trace(
        go.Bar(
            x=hours,
            y=res_dp["grid_import"],
            name="Dynamic Programming",
            marker_color="#27AE60",
        )
    )

    fig_grid.update_layout(
        barmode="group",
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear",
            dtick=1,
            gridcolor=grid_color,
            color=axis_color,
        ),
        yaxis=dict(
            title="Grid Power Imported (kWh)",
            gridcolor=grid_color,
            color=axis_color,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=axis_color),
        ),
        template=chart_theme,
        paper_bgcolor=bg_canvas,
        plot_bgcolor=bg_canvas,
        height=380,
    )
    st.plotly_chart(fig_grid, theme=None, width="stretch")

with tab3:
    fig_cost = go.Figure()
    fig_cost.add_trace(
        go.Scatter(
            x=hours,
            y=np.cumsum(res_base["hourly_cost"]),
            name=f"Baseline (Rs.{base_cost:,.0f})",
            line=dict(
                color="#888888" if dark_mode else "#7F8C8D",
                width=2,
                dash="dash",
            ),
        )
    )
    fig_cost.add_trace(
        go.Scatter(
            x=hours,
            y=np.cumsum(res_greedy["hourly_cost"]),
            name=f"Greedy (Rs.{greedy_cost:,.0f})",
            line=dict(color="#1E88E5", width=2.5),
        )
    )
    fig_cost.add_trace(
        go.Scatter(
            x=hours,
            y=np.cumsum(res_dp["hourly_cost"]),
            name=f"Optimal DP (Rs.{dp_cost:,.0f})",
            line=dict(color="#27AE60", width=3),
        )
    )

    fig_cost.update_layout(
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear",
            dtick=1,
            gridcolor=grid_color,
            color=axis_color,
        ),
        yaxis=dict(
            title="Cumulative Electricity Bill (INR)",
            gridcolor=grid_color,
            color=axis_color,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=axis_color),
        ),
        template=chart_theme,
        paper_bgcolor=bg_canvas,
        plot_bgcolor=bg_canvas,
        height=380,
    )
    st.plotly_chart(fig_cost, theme=None, width="stretch")

with tab4:
    st.markdown("#### Numerical Dynamic Programming Cost-to-Go Matrix: V[t, s]")
    st.markdown(
        "This table shows the 2D memoization table calculated by Dynamic Programming. "
        "Each cell shows the minimum remaining cost (in INR) to finish the day from hour $t$ given current battery charge."
    )

    V_table = res_dp["dp_table_V"]
    soc_axis = np.round(res_dp["soc_states"] * 100, 1)

    df_dp = pd.DataFrame(
        V_table.T,
        index=[f"SoC {s:.1f}%" for s in soc_axis],
        columns=[f"t={t}" for t in range(25)],
    )

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.dataframe(df_dp.round(2), width="stretch", height=280)
    with col_t2:
        st.markdown("**DP Table Facts:**")
        st.markdown(f"- **Matrix Size:** {df_dp.shape[0]} battery levels × {df_dp.shape[1]} hours")
        st.markdown(f"- **Stored Subproblems:** {df_dp.shape[0] * df_dp.shape[1]:,d}")
        st.markdown(f"- **Ending Rule at t=24:** Penalty if battery ends below starting {initial_soc*100:.1f}%.")

        csv_data = df_dp.to_csv().encode("utf-8")
        st.download_button(
            "Download DP Table (CSV)",
            data=csv_data,
            file_name="dp_cost_to_go_table.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("#### Cost-to-Go Heatmap")
    fig_dp_table = go.Figure(
        data=go.Heatmap(
            z=V_table[:24, :].T,
            x=list(range(24)),
            y=soc_axis,
            colorscale="Viridis",
            colorbar=dict(title="Cost-to-Go (INR)"),
            hovertemplate="Hour: %{x}<br>SoC: %{y:.1f}%<br>Cost-to-Go: Rs.%{z:.2f}<extra></extra>",
        )
    )
    fig_dp_table.update_layout(
        xaxis=dict(
            title="Stage / Hour of Day (t)",
            tickmode="linear",
            dtick=1,
            gridcolor=grid_color,
            color=axis_color,
        ),
        yaxis=dict(
            title="Battery State of Charge (%)",
            gridcolor=grid_color,
            color=axis_color,
        ),
        template=chart_theme,
        paper_bgcolor=bg_canvas,
        plot_bgcolor=bg_canvas,
        height=380,
    )
    st.plotly_chart(fig_dp_table, theme=None, width="stretch")

st.subheader("3. Benchmark Ledger & Evaluation Metrics")

peak_slice = slice(17, 23)
base_peak_imp = np.sum(res_base["grid_import"][peak_slice])
greedy_peak_imp = np.sum(res_greedy["grid_import"][peak_slice])
dp_peak_imp = np.sum(res_dp["grid_import"][peak_slice])

benchmark_df = pd.DataFrame({
    "Evaluation Metric": [
        "Total Daily Operating Cost (INR)",
        "Net Cost Reduction vs. Baseline (%)",
        "Peak Tariff Period Grid Import (17:00–22:00)",
        "Final State of Charge (SoC)",
        "Execution Time (ms)",
        "Theoretical Time Complexity",
        "Theoretical Space Complexity",
    ],
    "Baseline (No Storage)": [
        f"Rs.{base_cost:,.2f}",
        "0.0%",
        f"{base_peak_imp:.1f} kWh",
        "N/A",
        f"{res_base['solve_time_ms']:.3f} ms",
        "O(1)",
        "O(1)",
    ],
    "Greedy Heuristic": [
        f"Rs.{greedy_cost:,.2f}",
        f"{greedy_pct:.2f}%",
        f"{greedy_peak_imp:.1f} kWh",
        f"{res_greedy['battery_soc'][-1]*100:.1f}%",
        f"{res_greedy['solve_time_ms']:.3f} ms",
        "O(T)",
        "O(T)",
    ],
    "Dynamic Programming (Optimal)": [
        f"Rs.{dp_cost:,.2f}",
        f"{dp_pct:.2f}%",
        f"{dp_peak_imp:.1f} kWh",
        f"{res_dp['battery_soc'][-1]*100:.1f}%",
        f"{res_dp['solve_time_ms']:.2f} ms",
        "O(T · S²)",
        "O(T · S)",
    ],
})

table_html = benchmark_df.to_html(index=False, classes="benchmark-table")
st.markdown(table_html, unsafe_allow_html=True)

st.subheader("4. Algorithmic Insights & Search Space Comparison")

col_a1, col_a2 = st.columns(2)

with col_a1:
    st.markdown(f"""
    **Search Space Reduction:**
    * **Stages:** 24 hours
    * **Battery Discretization:** {num_states} levels
    * **Exhaustive Brute-Force Combinations:**
      $$S^T = {num_states}^{{24}} \\approx 10^{{{int(24 * np.log10(num_states))}}}$$
      Evaluating all paths at 1 billion paths/second would take $> 10^{{30}}$ years.
    * **Dynamic Programming Transitions:**
      $$T \\cdot S^2 = 24 \\times {num_states}^2 = {24 * (num_states**2):,d}$$
      Evaluated in **{res_dp['solve_time_ms']:.2f} ms**.
    """)

with col_a2:
    st.markdown("""
    **Why Greedy Fails:**
    * At 14:00, electricity costs Rs. 7.50. Greedy discharges immediately to eliminate grid import.
    * But reserving that energy for 19:00 saves **Rs. 13.00**.
    * Greedy loses Rs. 5.50 per unit of future value because it lacks foresight.
    * Dynamic Programming evaluates backwards from hour 23, identifying and reserving energy for the highest-value intervals.
    """)
