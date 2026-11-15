import os
import argparse
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator import GridEnvironment
from optimizers import BatterySpec, run_baseline, run_greedy, run_dynamic_programming


def run_benchmark(
    base_load: float = 32.0,
    pv_capacity: float = 48.0,
    battery_capacity: float = 60.0,
    max_power: float = 20.0,
    efficiency: float = 0.92,
    initial_soc: float = 0.30,
    seed: int = 42,
    save_plot: bool = False,
    output_path: str = "simulation_results.png",
):
    print("=" * 72)
    print("        GridOptima: Microgrid Storage Optimization Benchmark")
    print("=" * 72)
    print("System Parameters:")
    print(f"  - Base Load:            {base_load:.1f} kWh/h")
    print(f"  - Solar PV Capacity:    {pv_capacity:.1f} kW Peak")
    print(f"  - Battery Capacity:     {battery_capacity:.1f} kWh")
    print(f"  - Inverter Max Power:   {max_power:.1f} kW")
    print(f"  - Roundtrip Efficiency: {efficiency * 100:.1f}%")
    print(f"  - Initial SoC:          {initial_soc * 100:.1f}%\n")

    env = GridEnvironment(seed=seed)
    profile = env.get_24h_profile(base_load_kwh=base_load, pv_capacity_kw=pv_capacity)
    hours = profile["hours"]
    demand = profile["demand"]
    solar = profile["solar"]
    tariffs = profile["tariffs"]
    feed_in = profile["feed_in"]

    battery = BatterySpec(
        capacity_kwh=battery_capacity,
        max_power_kw=max_power,
        roundtrip_efficiency=efficiency,
        min_soc=0.10,
        max_soc=0.95,
        initial_soc=initial_soc,
    )

    res_base = run_baseline(demand, solar, tariffs, feed_in)
    res_greedy = run_greedy(demand, solar, tariffs, feed_in, battery)
    res_dp = run_dynamic_programming(demand, solar, tariffs, feed_in, battery, num_states=101)

    base_cost = res_base["total_cost"]
    greedy_cost = res_greedy["total_cost"]
    dp_cost = res_dp["total_cost"]

    greedy_savings = ((base_cost - greedy_cost) / base_cost) * 100 if base_cost > 0 else 0
    dp_savings = ((base_cost - dp_cost) / base_cost) * 100 if base_cost > 0 else 0

    peak_slice = slice(17, 23)
    base_peak_imp = np.sum(res_base["grid_import"][peak_slice])
    greedy_peak_imp = np.sum(res_greedy["grid_import"][peak_slice])
    dp_peak_imp = np.sum(res_dp["grid_import"][peak_slice])

    greedy_final_soc = f"{res_greedy['battery_soc'][-1]*100:.1f}%"
    dp_final_soc = f"{res_dp['battery_soc'][-1]*100:.1f}%"
    base_time_str = f"{res_base['solve_time_ms']:.3f} ms"
    greedy_time_str = f"{res_greedy['solve_time_ms']:.3f} ms"
    dp_time_str = f"{res_dp['solve_time_ms']:.2f} ms"

    base_peak_str = f"{base_peak_imp:.1f} kWh"
    greedy_peak_str = f"{greedy_peak_imp:.1f} kWh"
    dp_peak_str = f"{dp_peak_imp:.1f} kWh"

    evals_str = f"{res_dp.get('num_evaluations', 244824):,d}"

    print("-" * 72)
    print(f"{'Performance Metric':<32} {'Baseline':<12} {'Greedy':<12} {'Dynamic Prog':<12}")
    print("-" * 72)
    print(f"{'Total Daily Cost (INR)':<32} Rs.{base_cost:<8.2f} Rs.{greedy_cost:<8.2f} Rs.{dp_cost:<8.2f}")
    print(f"{'Cost Reduction vs Baseline':<32} {'0.0%':<12} {f'{greedy_savings:.2f}%':<12} {f'{dp_savings:.2f}%':<12}")
    print(f"{'Peak Import (17-22h)':<32} {base_peak_str:<12} {greedy_peak_str:<12} {dp_peak_str:<12}")
    print(f"{'Final State-of-Charge (SoC)':<32} {'N/A':<12} {greedy_final_soc:<12} {dp_final_soc:<12}")
    print(f"{'Execution Time':<32} {base_time_str:<12} {greedy_time_str:<12} {dp_time_str:<12}")
    print(f"{'Subproblems Evaluated':<32} {'0':<12} {'24':<12} {evals_str:<12}")
    print(f"{'Time Complexity':<32} {'O(1)':<12} {'O(T)':<12} {'O(T * S^2)':<12}")
    print(f"{'Space Complexity':<32} {'O(1)':<12} {'O(T)':<12} {'O(T * S)':<12}")
    print("-" * 72)
    print(f"[*] Brute-Force Exhaustive Paths: 101^24 ≈ 1.27 × 10^48 combinations")
    print(f"[*] Dynamic Programming reduces this to {evals_str} evaluations in {dp_time_str}!\n")

    if save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
        plt.subplots_adjust(hspace=0.32, wspace=0.25)

        ax1 = axes[0, 0]
        ax1.plot(hours, demand, color="#D9381E", linewidth=2.2, label="Demand (kWh)")
        ax1.plot(hours, solar, color="#107C41", linewidth=2.2, linestyle="--", label="Solar Generation (kWh)")
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Energy (kWh)", color="#222222")
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax1_twin = ax1.twinx()
        ax1_twin.step(hours, tariffs, color="#E67E22", linewidth=1.8, where="mid", linestyle=":", label="Tariff (INR/kWh)")
        ax1_twin.set_ylabel("Tariff (INR / kWh)", color="#E67E22")
        ax1.set_title("(a) 24-Hour Energy Profiles & Time-of-Use Tariff", fontweight="bold", fontsize=11)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9, fontsize=9)

        ax2 = axes[0, 1]
        ax2.plot(hours, res_greedy["battery_soc"] * 100, color="#4A90E2", linewidth=2.0, label="Greedy Heuristic")
        ax2.plot(hours, res_dp["battery_soc"] * 100, color="#27AE60", linewidth=2.4, label="Dynamic Programming")
        ax2.axhline(battery.max_soc * 100, color="crimson", linestyle="--", linewidth=1.2, label=f"Max Limit ({int(battery.max_soc*100)}%)")
        ax2.axhline(battery.min_soc * 100, color="darkorange", linestyle="--", linewidth=1.2, label=f"Min Reserve ({int(battery.min_soc*100)}%)")
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("State of Charge (%)")
        ax2.set_ylim(0, 105)
        ax2.set_xticks(range(0, 24, 2))
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.set_title("(b) Battery State-of-Charge Trajectory", fontweight="bold", fontsize=11)
        ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)

        ax3 = axes[1, 0]
        bar_width = 0.28
        ax3.bar(hours - bar_width, res_base["grid_import"], width=bar_width, color="#BDC3C7", label="Baseline (No Storage)")
        ax3.bar(hours, res_greedy["grid_import"], width=bar_width, color="#4A90E2", alpha=0.85, label="Greedy Heuristic")
        ax3.bar(hours + bar_width, res_dp["grid_import"], width=bar_width, color="#27AE60", alpha=0.85, label="Dynamic Programming")
        ax3.set_xlabel("Hour of Day")
        ax3.set_ylabel("Grid Import (kWh)")
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.set_title("(c) Hourly Grid Power Import by Algorithm", fontweight="bold", fontsize=11)
        ax3.legend(loc="upper left", framealpha=0.9, fontsize=9)

        ax4 = axes[1, 1]
        ax4.plot(hours, np.cumsum(res_base["hourly_cost"]), color="#7F8C8D", linewidth=2.0, linestyle="--", label=f"Baseline (Rs.{base_cost:.0f})")
        ax4.plot(hours, np.cumsum(res_greedy["hourly_cost"]), color="#4A90E2", linewidth=2.0, label=f"Greedy (Rs.{greedy_cost:.0f})")
        ax4.plot(hours, np.cumsum(res_dp["hourly_cost"]), color="#27AE60", linewidth=2.4, label=f"Dynamic Programming (Rs.{dp_cost:.0f})")
        ax4.set_xlabel("Hour of Day")
        ax4.set_ylabel("Cumulative Electricity Bill (INR)")
        ax4.set_xticks(range(0, 24, 2))
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.set_title("(d) Cumulative Cost Trajectory Across 24h", fontweight="bold", fontsize=11)
        ax4.legend(loc="upper left", framealpha=0.9, fontsize=9)

        plt.suptitle("GridOptima: Energy Management & Battery Scheduling Performance", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        plt.close()
        print(f"[OK] Benchmark plot saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridOptima Energy Storage Optimization Benchmark")
    parser.add_argument("--load", type=float, default=32.0, help="Base load in kWh/h (default: 32.0)")
    parser.add_argument("--pv", type=float, default=48.0, help="Solar PV capacity in kW (default: 48.0)")
    parser.add_argument("--capacity", type=float, default=60.0, help="Battery capacity in kWh (default: 60.0)")
    parser.add_argument("--power", type=float, default=20.0, help="Inverter max power in kW (default: 20.0)")
    parser.add_argument("--efficiency", type=float, default=0.92, help="Round-trip efficiency (default: 0.92)")
    parser.add_argument("--soc", type=float, default=0.30, help="Initial SoC (default: 0.30)")
    parser.add_argument("--plot", action="store_true", help="Save comparison plot to image")
    parser.add_argument("--output", type=str, default="simulation_results.png", help="Path to save output plot")

    args = parser.parse_args()

    run_benchmark(
        base_load=args.load,
        pv_capacity=args.pv,
        battery_capacity=args.capacity,
        max_power=args.power,
        efficiency=args.efficiency,
        initial_soc=args.soc,
        save_plot=args.plot,
        output_path=args.output,
    )
