import time
import numpy as np


class BatterySpec:
    def __init__(
        self,
        capacity_kwh: float = 60.0,
        max_power_kw: float = 20.0,
        roundtrip_efficiency: float = 0.92,
        min_soc: float = 0.10,
        max_soc: float = 0.95,
        initial_soc: float = 0.30,
    ):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.roundtrip_efficiency = roundtrip_efficiency
        self.efficiency = np.sqrt(roundtrip_efficiency)
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.initial_soc = initial_soc


def run_baseline(
    demand: np.ndarray,
    solar: np.ndarray,
    tariffs: np.ndarray,
    feed_in: np.ndarray,
) -> dict:
    t0 = time.perf_counter()

    grid_import = np.maximum(0.0, demand - solar)
    grid_export = np.maximum(0.0, solar - demand)
    hourly_cost = (grid_import * tariffs) - (grid_export * feed_in)

    return {
        "grid_import": grid_import,
        "grid_export": grid_export,
        "battery_soc": np.zeros(24),
        "battery_power": np.zeros(24),
        "hourly_cost": hourly_cost,
        "total_cost": float(np.sum(hourly_cost)),
        "solve_time_ms": (time.perf_counter() - t0) * 1000.0,
    }


def run_greedy(
    demand: np.ndarray,
    solar: np.ndarray,
    tariffs: np.ndarray,
    feed_in: np.ndarray,
    battery: BatterySpec,
) -> dict:
    t0 = time.perf_counter()

    soc_history = np.zeros(24)
    battery_power = np.zeros(24)
    grid_import = np.zeros(24)
    grid_export = np.zeros(24)

    min_energy = battery.min_soc * battery.capacity_kwh
    max_energy = battery.max_soc * battery.capacity_kwh
    current_energy = battery.initial_soc * battery.capacity_kwh
    eta = battery.efficiency

    for t in range(24):
        net_load = demand[t] - solar[t]

        if net_load < 0:
            surplus = -net_load
            headroom = (max_energy - current_energy) / eta
            charge_power = min(surplus, battery.max_power_kw, headroom)

            current_energy += charge_power * eta
            battery_power[t] = -charge_power
            grid_export[t] = surplus - charge_power
            grid_import[t] = 0.0
        else:
            deficit = net_load
            if tariffs[t] >= 7.50:
                available = (current_energy - min_energy) * eta
                discharge_power = min(deficit, battery.max_power_kw, available)

                current_energy -= discharge_power / eta
                battery_power[t] = discharge_power
                grid_import[t] = deficit - discharge_power
            else:
                grid_import[t] = deficit

            grid_export[t] = 0.0

        soc_history[t] = current_energy / battery.capacity_kwh

    hourly_cost = (grid_import * tariffs) - (grid_export * feed_in)

    return {
        "grid_import": grid_import,
        "grid_export": grid_export,
        "battery_soc": soc_history,
        "battery_power": battery_power,
        "hourly_cost": hourly_cost,
        "total_cost": float(np.sum(hourly_cost)),
        "solve_time_ms": (time.perf_counter() - t0) * 1000.0,
    }


def run_dynamic_programming(
    demand: np.ndarray,
    solar: np.ndarray,
    tariffs: np.ndarray,
    feed_in: np.ndarray,
    battery: BatterySpec,
    num_states: int = 101,
    enforce_cyclic: bool = True,
) -> dict:
    t0 = time.perf_counter()

    soc_states = np.linspace(battery.min_soc, battery.max_soc, num_states)
    energy_states = soc_states * battery.capacity_kwh
    eta = battery.efficiency

    V = np.full((25, num_states), np.inf)

    if enforce_cyclic:
        peak_rate = float(np.max(tariffs))
        V[24, :] = np.where(
            soc_states >= battery.initial_soc,
            0.0,
            (battery.initial_soc - soc_states) * battery.capacity_kwh * peak_rate,
        )
    else:
        V[24, :] = 0.0

    policy = np.zeros((24, num_states), dtype=int)

    for t in range(23, -1, -1):
        net_load = demand[t] - solar[t]
        c_buy = tariffs[t]
        c_sell = feed_in[t]

        for i, e_from in enumerate(energy_states):
            delta_e = energy_states - e_from

            charge_power = delta_e / eta
            valid_charge = (delta_e >= 0) & (charge_power <= battery.max_power_kw)

            discharge_power = -delta_e * eta
            valid_discharge = (delta_e < 0) & (discharge_power <= battery.max_power_kw)

            valid_actions = valid_charge | valid_discharge

            p_bat = np.where(delta_e >= 0, -charge_power, discharge_power)
            grid_exchange = net_load - p_bat

            stage_cost = np.where(grid_exchange >= 0, grid_exchange * c_buy, grid_exchange * c_sell)
            path_cost = np.where(valid_actions, stage_cost + V[t + 1, :], np.inf)

            best_idx = np.argmin(path_cost)
            V[t, i] = path_cost[best_idx]
            policy[t, i] = best_idx

    initial_idx = int(np.argmin(np.abs(soc_states - battery.initial_soc)))

    optimal_soc = np.zeros(24)
    optimal_battery_power = np.zeros(24)
    grid_import = np.zeros(24)
    grid_export = np.zeros(24)

    current_idx = initial_idx
    for t in range(24):
        next_idx = policy[t, current_idx]
        delta_e = energy_states[next_idx] - energy_states[current_idx]

        if delta_e >= 0:
            p_b = -(delta_e / eta)
        else:
            p_b = -delta_e * eta

        optimal_battery_power[t] = p_b
        optimal_soc[t] = soc_states[next_idx]

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
        "battery_soc": optimal_soc,
        "battery_power": optimal_battery_power,
        "hourly_cost": hourly_cost,
        "total_cost": float(np.sum(hourly_cost)),
        "solve_time_ms": (time.perf_counter() - t0) * 1000.0,
        "dp_table_V": V,
        "policy_table": policy,
        "soc_states": soc_states,
        "num_evaluations": 24 * (num_states ** 2),
        "num_states": num_states,
    }
