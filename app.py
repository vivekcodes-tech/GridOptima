"""
GridOptima - Interactive Streamlit Dashboard
Real-time microgrid simulation comparing Baseline, Greedy, and Dynamic Programming algorithms.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator import GridEnvironment
from optimizers import BatterySpec, run_baseline, run_greedy, run_dynamic_programming

st.set_page_config(
    page_title="GridOptima | Smart Grid Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.title("⚡ GridOptima Control Panel")
st.sidebar.markdown("Configure grid, generation, and storage parameters.")

st.sidebar.subheader("1. Load & Solar Parameters")
base_load = st.sidebar.slider("Base Commercial Load (kWh/h)", 15.0, 60.0, 32.0, 1.0)
pv_capacity = st.sidebar.slider("Solar PV Capacity (kW Peak)", 10.0, 100.0, 48.0, 2.0)

st.sidebar.subheader("2. Battery Energy Storage System (BESS)")
battery_capacity = st.sidebar.slider("Battery Storage Capacity (kWh)", 20.0, 150.0, 60.0, 5.0)
max_c_rate = st.sidebar.slider("Max Inverter C-Rate Limit (kW)", 10.0, 40.0, 20.0, 2.0)
efficiency = st.sidebar.slider("Round-Trip Efficiency (η)", 0.75, 0.98, 0.92, 0.01)
initial_soc = st.sidebar.slider("Initial SoC (%)", 10, 90, 30, 5) / 100.0

# ----------------- EXECUTE SIMULATION ENGINE -----------------
env = GridEnvironment(seed=42)
profile = env.get_24h_profile(base_load_kwh=base_load, pv_capacity_kw=pv_capacity)

battery = BatterySpec(
    capacity_kwh=battery_capacity,
    max_charge_rate_kw=max_c_rate,
    max_discharge_rate_kw=max_c_rate,
    roundtrip_efficiency=efficiency,
    min_soc=0.10,
    max_soc=0.95,
    initial_soc=initial_soc,
)

res_base = run_baseline(profile["demand"], profile["solar"], profile["tariffs"], profile["feed_in"])
res_greedy = run_greedy(profile["demand"], profile["solar"], profile["tariffs"], profile["feed_in"], battery)
res_dp = run_dynamic_programming(profile["demand"], profile["solar"], profile["tariffs"], profile["feed_in"], battery)

# ----------------- MAIN DASHBOARD HEADER -----------------
st.title("⚡ GridOptima: Smart Grid Energy Optimization Engine")
st.markdown(
    "Algorithmic peak-shaving and arbitrage comparison using **Greedy Heuristics** vs **Bellman Dynamic Programming**."
)

# ----------------- KPI SUMMARY METRICS -----------------
col1, col2, col3, col4 = st.columns(4)

base_cost = res_base["total_cost"]
greedy_cost = res_greedy["total_cost"]
dp_cost = res_dp["total_cost"]

dp_savings = base_cost - dp_cost
dp_pct = (dp_savings / base_cost) * 100 if base_cost > 0 else 0.0
greedy_pct = ((base_cost - greedy_cost) / base_cost) * 100 if base_cost > 0 else 0.0

with col1:
    st.metric("Baseline Daily Cost", f"₹{base_cost:,.2f}", "No BESS")
with col2:
    st.metric("Greedy Heuristic Cost", f"₹{greedy_cost:,.2f}", f"-{greedy_pct:.1f}% Savings", delta_color="inverse")
with col3:
    st.metric("Optimal DP Cost", f"₹{dp_cost:,.2f}", f"-{dp_pct:.1f}% Max Savings", delta_color="inverse")
with col4:
    total_solar_kwh = float(np.sum(profile["solar"]))
    st.metric("Total Solar Generation", f"{total_solar_kwh:.1f} kWh", "Zero Emission")

st.divider()

# ----------------- PLOT 1: 24-HOUR ENERGY DISPATCH & TARIFF -----------------
st.subheader("📊 24-Hour Load, Generation & Time-of-Use Tariff Regime")

fig_dispatch = make_subplots(specs=[[{"secondary_y": True}]])
hours = profile["hours"]

fig_dispatch.add_trace(
    go.Scatter(x=hours, y=profile["demand"], name="Demand (kWh)", line=dict(color="#EF553B", width=3)),
    secondary_y=False
)
fig_dispatch.add_trace(
    go.Scatter(x=hours, y=profile["solar"], name="Solar Generation (kWh)", line=dict(color="#00CC96", width=3, dash="dot")),
    secondary_y=False
)
fig_dispatch.add_trace(
    go.Scatter(x=hours, y=profile["tariffs"], name="Tariff (₹/kWh)", line=dict(color="#FFA15A", width=2, dash="dash")),
    secondary_y=True
)

fig_dispatch.update_layout(
    xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode="linear", dtick=1),
    yaxis=dict(title="Energy (kWh)"),
    yaxis2=dict(title="Tariff (INR / kWh)", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=40, b=40),
    template="plotly_dark",
    height=400,
)
st.plotly_chart(fig_dispatch, use_container_width=True)

# ----------------- PLOT 2: ALGORITHM COMPARISON -----------------
tab1, tab2 = st.tabs(["🔋 Battery State-of-Charge (SoC)", "📈 Cumulative Expenditure Curve"])

with tab1:
    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(x=hours, y=res_greedy["battery_soc"] * 100, name="Greedy SoC (%)", line=dict(color="#636EFA", width=2.5)))
    fig_soc.add_trace(go.Scatter(x=hours, y=res_dp["battery_soc"] * 100, name="Dynamic Programming SoC (%)", line=dict(color="#00CC96", width=3)))
    fig_soc.add_hline(y=battery.max_soc * 100, line_dash="dot", line_color="red", annotation_text="Max Safety Limit (95%)")
    fig_soc.add_hline(y=battery.min_soc * 100, line_dash="dot", line_color="orange", annotation_text="Min Reserve Limit (10%)")
    
    fig_soc.update_layout(
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        yaxis=dict(title="State of Charge (%)", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_soc, use_container_width=True)

with tab2:
    cum_base = np.cumsum(res_base["hourly_cost"])
    cum_greedy = np.cumsum(res_greedy["hourly_cost"])
    cum_dp = np.cumsum(res_dp["hourly_cost"])

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(x=hours, y=cum_base, name="Baseline (No Storage)", line=dict(color="#EF553B", width=2.5)))
    fig_cost.add_trace(go.Scatter(x=hours, y=cum_greedy, name="Greedy Optimization", line=dict(color="#AB63FA", width=2.5)))
    fig_cost.add_trace(go.Scatter(x=hours, y=cum_dp, name="Optimal DP Schedule", line=dict(color="#00CC96", width=3)))

    fig_cost.update_layout(
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        yaxis=dict(title="Cumulative Cost (INR ₹)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_cost, use_container_width=True)

# ----------------- PERFORMANCE ANALYSIS TABLE -----------------
st.subheader("📑 Algorithmic Benchmark & Grid Interaction Ledger")

col_table, col_desc = st.columns([3, 2])

with col_table:
    benchmark_data = {
        "Metric": [
            "Total Daily Cost (INR)",
            "Net Cost Reduction (%)",
            "Peak Period Grid Import (kWh)",
            "Total Solar Utilized (kWh)",
            "Battery Cycles (Equivalent)",
        ],
        "Baseline": [
            f"₹{base_cost:.2f}",
            "0.0%",
            f"{np.sum(res_base['grid_import'][17:23]):.2f} kWh",
            f"{np.sum(np.minimum(profile['demand'], profile['solar'])):.2f} kWh",
            "0.0",
        ],
        "Greedy Heuristic": [
            f"₹{greedy_cost:.2f}",
            f"{greedy_pct:.2f}%",
            f"{np.sum(res_greedy['grid_import'][17:23]):.2f} kWh",
            f"{np.sum(profile['solar'] - res_greedy['grid_export']):.2f} kWh",
            f"{np.sum(np.abs(res_greedy['battery_power'])) / (2 * battery.capacity_kwh):.2f}",
        ],
        "Dynamic Programming": [
            f"₹{dp_cost:.2f}",
            f"{dp_pct:.2f}%",
            f"{np.sum(res_dp['grid_import'][17:23]):.2f} kWh",
            f"{np.sum(profile['solar'] - res_dp['grid_export']):.2f} kWh",
            f"{np.sum(np.abs(res_dp['battery_power'])) / (2 * battery.capacity_kwh):.2f}",
        ],
    }
    st.dataframe(benchmark_data, use_container_width=True)

with col_desc:
    st.markdown("""
    **Algorithm Insights:**
    - **Baseline:** High dependency on grid during evening peak hours (₹13/kWh tariff window) drives up overall operating cost.
    - **Greedy:** Efficiently captures daytime solar excess, but exhausts battery charge early in the peak cycle without looking ahead to peak pricing hours.
    - **Dynamic Programming:** Evaluates all $N$-stage cost trajectories through Bellman backward induction, intentionally reserving battery capacity for the highest tariff intervals to yield the theoretical global minimum cost.
    """)