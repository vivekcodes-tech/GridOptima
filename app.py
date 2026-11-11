import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator import GridEnvironment
from optimizers import BatterySpec, run_baseline, run_greedy, run_dynamic_programming

st.set_page_config(
    page_title="GridOptima | Algorithmic Energy Management",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GridOptima: Microgrid Energy Management & Scheduling")
st.markdown(
    "**Analysis and Design of Algorithms (ADA)**  \n"
    "Finite-horizon sequential decision making comparing **Greedy Heuristics** vs. "
    "**Bellman Dynamic Programming** under Time-of-Use electricity tariffs."
)

with st.expander("System Parameters and Input Controls (Expand / Collapse)", expanded=True):
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.markdown("**1. Load and Solar PV**")
        base_load = st.number_input("Base Load (kWh/h)", min_value=10.0, max_value=80.0, value=32.0, step=1.0)
        pv_capacity = st.number_input("Solar PV Peak Capacity (kW)", min_value=0.0, max_value=150.0, value=48.0, step=2.0)

    with col_p2:
        st.markdown("**2. Battery Storage System**")
        battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=10.0, max_value=200.0, value=60.0, step=5.0)
        max_power = st.number_input("Inverter Power Limit (kW)", min_value=5.0, max_value=60.0, value=20.0, step=1.0)
        efficiency = st.slider("Roundtrip Efficiency (η)", min_value=0.75, max_value=0.98, value=0.92, step=0.01)

    with col_p3:
        st.markdown("**3. Algorithm and Grid Settings**")
        initial_soc = st.slider("Initial State of Charge (%)", min_value=10, max_value=90, value=30, step=5) / 100.0
        num_states = st.selectbox("DP State Discretization (S)", options=[21, 31, 51, 81, 101, 151], index=4)
        enforce_cyclic = st.checkbox("Enforce Cyclic Boundary Condition (SoC_end >= SoC_start)", value=True)

st.sidebar.markdown("### GridOptima Control Panel")
st.sidebar.markdown(f"**Base Load:** {base_load:.1f} kWh/h")
st.sidebar.markdown(f"**Solar PV:** {pv_capacity:.1f} kW Peak")
st.sidebar.markdown(f"**Battery Capacity:** {battery_capacity:.1f} kWh")
st.sidebar.markdown(f"**Max Power:** {max_power:.1f} kW")
st.sidebar.markdown(f"**Efficiency:** {efficiency*100:.1f}%")
st.sidebar.markdown(f"**Initial SoC:** {initial_soc*100:.1f}%")
st.sidebar.markdown(f"**State Levels (S):** {num_states}")
st.sidebar.markdown(f"**Cyclic Boundary:** {'Enabled' if enforce_cyclic else 'Disabled'}")

env = GridEnvironment(seed=42)
profile = env.get_24h_profile(base_load_kwh=base_load, pv_capacity_kw=pv_capacity)
hours = profile["hours"]

battery = BatterySpec(
    capacity_kwh=battery_capacity,
    max_power_kw=max_power,
    roundtrip_efficiency=efficiency,
    min_soc=0.10,
    max_soc=0.95,
    initial_soc=initial_soc,
)

res_base = run_baseline(profile["demand"], profile["solar"], profile["tariffs"], profile["feed_in"])
res_greedy = run_greedy(profile["demand"], profile["solar"], profile["tariffs"], profile["feed_in"], battery)
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

greedy_pct = ((base_cost - greedy_cost) / base_cost) * 100 if base_cost > 0 else 0.0
dp_pct = ((base_cost - dp_cost) / base_cost) * 100 if base_cost > 0 else 0.0
total_evals = res_dp["num_evaluations"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline (No Storage)", f"Rs.{base_cost:,.2f}")
col2.metric("Greedy Heuristic", f"Rs.{greedy_cost:,.2f}", f"-{greedy_pct:.2f}%", delta_color="inverse")
col3.metric("Dynamic Programming", f"Rs.{dp_cost:,.2f}", f"-{dp_pct:.2f}%", delta_color="inverse")
col4.metric("Subproblems Evaluated", f"{total_evals:,d}", f"{res_dp['solve_time_ms']:.2f} ms")

st.write("")

with st.expander("Mathematical Formulation and Theoretical Complexity (Expand / Collapse)", expanded=False):
    st.markdown(r"""
    ### 1. Problem Formulation: Finite-Horizon Optimal Control
    We model the 24-hour battery scheduling problem as a multi-stage discrete-time decision process over $T = 24$ stages:
    $$\min_{P_{\text{bat}}} \sum_{t=0}^{23} \Big[ P_{\text{import}}(t) \cdot \lambda_{\text{buy}}(t) - P_{\text{export}}(t) \cdot \lambda_{\text{sell}}(t) \Big]$$

    **Physical Constraints:**
    * **Power Balance:** $P_{\text{import}}(t) - P_{\text{export}}(t) = \text{Demand}(t) - \text{Solar}(t) - P_{\text{bat}}(t)$
    * **Inverter Bounds:** $-P_{\text{max}} \le P_{\text{bat}}(t) \le P_{\text{max}}$
    * **State of Charge Bounds:** $\text{SoC}_{\min} \le \text{SoC}(t) \le \text{SoC}_{\max}$
    * **Cyclic Boundary Condition:** $\text{SoC}(24) \ge \text{SoC}(0)$

    ### 2. The Bellman Recurrence Relation
    Let $V_t(s)$ denote the optimal minimum cost-to-go from stage $t \in \{0, \dots, 23\}$ starting from state $s$:
    $$V_t(s) = \min_{s' \in \mathcal{S}} \Big[ \text{StageCost}_t(s, s') + V_{t+1}(s') \Big]$$
    * **Base / Terminal Condition:**
      $$V_{24}(s) = \begin{cases} 0 & \text{if } s \ge s_0 \\ \lambda_{\text{peak}} \cdot (s_0 - s) \cdot \text{Capacity} & \text{if } s < s_0 \end{cases}$$

    ### 3. Asymptotic Complexity Comparison
    * **Brute Force (Exhaustive Search):** With $T = 24$ stages and $S$ discretized states, the search space is $\mathcal{O}(S^T)$. For $S=101$, this is $101^{24} \approx 1.27 \times 10^{48}$ combinations.
    * **Dynamic Programming:** Explores only valid stage transitions, memoizing subproblem solutions in a 2D table. Complexity drops to $\mathcal{O}(T \cdot S^2)$ time and $\mathcal{O}(T \cdot S)$ space.
    * **Greedy Heuristic:** Evaluates decisions step-by-step with zero lookahead in $\mathcal{O}(T)$ time.
    """)

st.subheader("1. 24-Hour Microgrid Load, Solar PV & Time-of-Use Tariff")

fig_dispatch = make_subplots(specs=[[{"secondary_y": True}]])
fig_dispatch.add_trace(
    go.Scatter(x=hours, y=profile["demand"], name="Demand (kWh)", line=dict(color="#E74C3C", width=2.5)),
    secondary_y=False,
)
fig_dispatch.add_trace(
    go.Scatter(x=hours, y=profile["solar"], name="Solar Generation (kWh)", line=dict(color="#27AE60", width=2.5, dash="dash")),
    secondary_y=False,
)
fig_dispatch.add_trace(
    go.Scatter(x=hours, y=profile["tariffs"], name="Tariff (INR/kWh)", line=dict(color="#F39C12", width=2.2, dash="dot")),
    secondary_y=True,
)

fig_dispatch.update_layout(
    xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode="linear", dtick=1),
    yaxis=dict(title="Energy (kWh)"),
    yaxis2=dict(title="Tariff Rate (INR / kWh)", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=30, b=40),
    template="plotly_white",
    height=360,
)
st.plotly_chart(fig_dispatch, use_container_width=True)

st.subheader("2. Algorithmic Evaluation and Dynamic Programming Table")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Battery SoC Trajectory",
    "Hourly Grid Import",
    "Cumulative Cost Curve",
    "DP Tabulation Matrix V[t, s]",
    "ADA Complexity Analysis",
])

with tab1:
    fig_soc = go.Figure()
    fig_soc.add_trace(
        go.Scatter(x=hours, y=res_greedy["battery_soc"] * 100, name="Greedy Heuristic", line=dict(color="#1E88E5", width=2.5))
    )
    fig_soc.add_trace(
        go.Scatter(x=hours, y=res_dp["battery_soc"] * 100, name="Dynamic Programming (Optimal)", line=dict(color="#27AE60", width=3))
    )
    fig_soc.add_hline(
        y=battery.max_soc * 100, line_dash="dash", line_color="#C0392B", annotation_text=f"Max Limit ({int(battery.max_soc*100)}%)"
    )
    fig_soc.add_hline(
        y=battery.min_soc * 100, line_dash="dash", line_color="#E67E22", annotation_text=f"Min Reserve ({int(battery.min_soc*100)}%)"
    )

    fig_soc.update_layout(
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        yaxis=dict(title="State of Charge (%)", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_soc, use_container_width=True)
    st.info(
        "Key Algorithmic Finding: Greedy starts discharging early during standard tariff hours (7.50 INR/kWh at 14:00-16:00), "
        "exhausting battery energy prematurely. Dynamic Programming anticipates the steep 13.00 INR/kWh evening peak (17:00–22:00) "
        "and holds charge for maximum marginal savings."
    )

with tab2:
    fig_grid = go.Figure()
    fig_grid.add_trace(go.Bar(x=hours, y=res_base["grid_import"], name="Baseline", marker_color="#BDC3C7"))
    fig_grid.add_trace(go.Bar(x=hours, y=res_greedy["grid_import"], name="Greedy", marker_color="#1E88E5"))
    fig_grid.add_trace(go.Bar(x=hours, y=res_dp["grid_import"], name="Dynamic Programming", marker_color="#27AE60"))

    fig_grid.update_layout(
        barmode="group",
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        yaxis=dict(title="Grid Power Imported (kWh)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_grid, use_container_width=True)

with tab3:
    fig_cost = go.Figure()
    fig_cost.add_trace(
        go.Scatter(x=hours, y=np.cumsum(res_base["hourly_cost"]), name=f"Baseline (Rs.{base_cost:,.0f})", line=dict(color="#7F8C8D", width=2, dash="dash"))
    )
    fig_cost.add_trace(
        go.Scatter(x=hours, y=np.cumsum(res_greedy["hourly_cost"]), name=f"Greedy (Rs.{greedy_cost:,.0f})", line=dict(color="#1E88E5", width=2.5))
    )
    fig_cost.add_trace(
        go.Scatter(x=hours, y=np.cumsum(res_dp["hourly_cost"]), name=f"Optimal DP (Rs.{dp_cost:,.0f})", line=dict(color="#27AE60", width=3))
    )

    fig_cost.update_layout(
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        yaxis=dict(title="Cumulative Electricity Bill (INR)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_cost, use_container_width=True)

with tab4:
    st.markdown("#### Numerical Dynamic Programming Cost-to-Go Matrix: V[t, s]")
    st.markdown(
        "This table displays the actual 2D DP memoization matrix. "
        "Each cell $(s, t)$ stores the minimum cost-to-go (in INR) from stage $t$ to stage 24 given battery state $s$."
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
        st.dataframe(df_dp.round(2), use_container_width=True, height=280)
    with col_t2:
        st.markdown("**DP Matrix Properties:**")
        st.markdown(f"- **Dimensions:** {df_dp.shape[0]} states × {df_dp.shape[1]} stages")
        st.markdown(f"- **Total Stored Cells:** {df_dp.shape[0] * df_dp.shape[1]:,d}")
        st.markdown(f"- **Terminal Condition at t=24:** Cyclic penalty applied if SoC < {initial_soc*100:.1f}%.")

        csv_data = df_dp.to_csv().encode("utf-8")
        st.download_button(
            "Download DP Table (CSV)",
            data=csv_data,
            file_name="dp_cost_to_go_table.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("#### Cost-to-Go Heatmap Visualization")
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
        xaxis=dict(title="Stage / Hour of Day (t)", tickmode="linear", dtick=1),
        yaxis=dict(title="Battery State of Charge (%)"),
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_dp_table, use_container_width=True)

with tab5:
    st.markdown("#### Theoretical vs. Empirical Computational Complexity")
    st.markdown(
        "In Analysis and Design of Algorithms (ADA), we prove how Dynamic Programming eliminates exponential search spaces."
    )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(f"""
        **Search Space Comparison:**
        * **Number of Stages (T):** 24 hours
        * **Number of State Levels (S):** {num_states} states
        * **Exhaustive Brute-Force Combinations:**
          $$S^T = {num_states}^{{24}} \\approx 10^{{{int(24 * np.log10(num_states))}}}$$
          *(At 1 billion evaluations per second, this would take $> 10^{{30}}$ years).*
        * **Dynamic Programming Evaluations:**
          $$T \\cdot S^2 = 24 \\times {num_states}^2 = {24 * (num_states**2):,d}$$
        * **Actual Solve Time:** **{res_dp['solve_time_ms']:.2f} ms**
        """)

    with col_c2:
        st.markdown("""
        **Why Greedy Fails (Counterexample to Greedy Choice Property):**
        * A problem exhibits the **Greedy Choice Property** if locally optimal choices lead to a globally optimal solution.
        * In this system:
          * At $t = 14:00$, tariff is 7.50 INR. Greedy sees a local deficit and discharges to save 7.50 INR/unit.
          * But the stored unit would have saved **13.00 INR/unit** at $t = 19:00$.
          * Discharging locally destroys future global value:
            $$\\text{Local Profit} = 7.50 < \\text{Opportunity Cost} = 13.00$$
          * Therefore, the Greedy Choice Property fails, and Dynamic Programming is strictly required.
        """)

st.subheader("3. Benchmark Ledger & Evaluation Metrics")

peak_slice = slice(17, 23)
base_peak_imp = np.sum(res_base["grid_import"][peak_slice])
greedy_peak_imp = np.sum(res_greedy["grid_import"][peak_slice])
dp_peak_imp = np.sum(res_dp["grid_import"][peak_slice])

benchmark_data = {
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
}

st.dataframe(benchmark_data, use_container_width=True)
