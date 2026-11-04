"""
GridOptima Optimization Core
Implements:
1. Baseline Scenario (Uncontrolled)
2. Greedy Heuristic Scheduler
3. Dynamic Programming Backward Induction Optimizer
"""

import numpy as np


class BatterySpec:
    def __init__(
        self,
        capacity_kwh=60.0,
        max_charge_rate_kw=20.0,
        max_discharge_rate_kw=20.0,
        roundtrip_efficiency=0.92,
        min_soc=0.10,
        max_soc=0.95,
        initial_soc=0.30,
    ):
        self.capacity_kwh = capacity_kwh
        self.max_charge_rate_kw = max_charge_rate_kw
        self.max_discharge_rate_kw = max_discharge_rate_kw
        self.efficiency = roundtrip_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.initial_soc = initial_soc


def run_baseline(demand, solar, tariffs, feed_in):
    """
    Baseline strategy: No battery intervention.
    Excess solar is exported to the grid; energy deficits are directly purchased.
    """
    grid_import = np.maximum(0.0, demand - solar)
    grid_export = np.maximum(0.0, solar - demand)
    hourly_cost = (grid_import * tariffs) - (grid_export * feed_in)
    return {
        "grid_import": grid_import,
        "grid_export": grid_export,
        "hourly_cost": hourly_cost,
        "total_cost": float(np.sum(hourly_cost)),
        "battery_soc": np.zeros(24),
        "battery_power": np.zeros(24),
    }


def run_greedy(demand, solar, tariffs, feed_in, battery: BatterySpec):
    """
    Greedy Strategy:
    - Excess solar -> Charge battery first, export remainder.
    - Deficit -> If tariff is Peak or Normal, discharge battery to minimize expensive imports.
    - If tariff is Off-peak, preserve battery or charge if solar is available.
    """
    soc = battery.initial_soc
    battery_power = np.zeros(24)  # > 0 discharging, < 0 charging
    grid_import = np.zeros(24)
    grid_export = np.zeros(24)
    soc_history = np.zeros(24)

    min_kwh = battery.min_soc * battery.capacity_kwh
    max_kwh = battery.max_soc * battery.capacity_kwh
    current_kwh = soc * battery.capacity_kwh

    for t in range(24):
        net_energy = demand[t] - solar[t]

        if net_energy < 0:
            # Solar surplus: charge battery first
            surplus = -net_energy
            room_to_charge = (max_kwh - current_kwh) / battery.efficiency
            charge_amount = min(surplus, battery.max_charge_rate_kw, room_to_charge)
            current_kwh += charge_amount * battery.efficiency
            battery_power[t] = -charge_amount
            excess_export = surplus - charge_amount
            grid_export[t] = excess_export
            grid_import[t] = 0.0
        else:
            # Energy deficit: discharge battery during peak hours (>= 7.5 INR)
            deficit = net_energy
            if tariffs[t] >= 7.50:
                available_discharge = (current_kwh - min_kwh) * battery.efficiency
                discharge_amount = min(deficit, battery.max_discharge_rate_kw, available_discharge)
                current_kwh -= discharge_amount / battery.efficiency
                battery_power[t] = discharge_amount
                remaining_deficit = deficit - discharge_amount
                grid_import[t] = remaining_deficit
            else:
                grid_import[t] = deficit
            grid_export[t] = 0.0

        soc_history[t] = current_kwh / battery.capacity_kwh

    hourly_cost = (grid_import * tariffs) - (grid_export * feed_in)
    return {
        "grid_import": grid_import,
        "grid_export": grid_export,
        "hourly_cost": hourly_cost,
        "total_cost": float(np.sum(hourly_cost)),
        "battery_soc": soc_history,
        "battery_power": battery_power,
    }


def run_dynamic_programming(demand, solar, tariffs, feed_in, battery: BatterySpec, num_states=101):
    """
    Globally Optimal Energy Management via Bellman Dynamic Programming.
    Discretizes State of Charge into 'num_states' grid points.
    Solves backward in time (t=23 down to 0) and reconstructs optimal sequence.
    """
    soc_states = np.linspace(battery.min_soc, battery.max_soc, num_states)
    kwh_states = soc_states * battery.capacity_kwh

    # DP Table: V[t, state_index] stores min cost-to-go from stage t
    V = np.full((25, num_states), np.inf)
    V[24, :] = 0.0  # Terminal boundary condition (no remaining cost)
    policy = np.zeros((24, num_states), dtype=int)

    # Backward Induction
    for t in range(23, -1, -1):
        net_load = demand[t] - solar[t]
        c_tariff = tariffs[t]
        c_feed = feed_in[t]

        for i, s_from in enumerate(kwh_states):
            # Delta battery kWh between consecutive stages
            delta_e = kwh_states - s_from

            # Charging branch (delta_e > 0)
            charge_power = delta_e / battery.efficiency
            valid_charge = (delta_e >= 0) & (charge_power <= battery.max_charge_rate_kw)

            # Discharging branch (delta_e < 0)
            discharge_power = -delta_e * battery.efficiency
            valid_discharge = (delta_e < 0) & (discharge_power <= battery.max_discharge_rate_kw)

            valid_actions = valid_charge | valid_discharge

            # Stage Cost Calculation
            # p_bat: positive = feeding load, negative = taking from system
            p_bat = np.where(delta_e >= 0, -charge_power, discharge_power)
            grid_exchange = net_load - p_bat

            instant_cost = np.where(
                grid_exchange >= 0,
                grid_exchange * c_tariff,
                grid_exchange * c_feed
            )

            total_path_cost = np.where(valid_actions, instant_cost + V[t + 1, :], np.inf)
            best_idx = np.argmin(total_path_cost)

            V[t, i] = total_path_cost[best_idx]
            policy[t, i] = best_idx

    # Forward Reconstruction from initial SoC
    initial_idx = np.argmin(np.abs(soc_states - battery.initial_soc))
    optimal_soc_history = np.zeros(24)
    optimal_battery_power = np.zeros(24)
    grid_import = np.zeros(24)
    grid_export = np.zeros(24)

    current_idx = initial_idx
    for t in range(24):
        next_idx = policy[t, current_idx]
        delta_kwh = kwh_states[next_idx] - kwh_states[current_idx]

        if delta_kwh >= 0:
            p_b = -(delta_kwh / battery.efficiency)
        else:
            p_b = -delta_kwh * battery.efficiency

        optimal_battery_power[t] = p_b
        optimal_soc_history[t] = soc_states[next_idx]

        net_exchange = (demand[t] - solar[t]) - p_b
        if net_exchange >= 0:
            grid_import[t] = net_exchange
            grid_export[t] = 0.0
        else:
            grid_import[t] = 0.0
            grid_export[t] = -net_exchange

        current_idx = next_idx

    hourly_cost = (grid_import * tariffs) - (grid_export * feed_in)
    return {
        "grid_import": grid_import,
        "grid_export": grid_export,
        "hourly_cost": hourly_cost,
        "total_cost": float(np.sum(hourly_cost)),
        "battery_soc": optimal_soc_history,
        "battery_power": optimal_battery_power,
    }