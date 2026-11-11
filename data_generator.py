import numpy as np


class GridEnvironment:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def get_24h_profile(self, base_load_kwh: float = 32.0, pv_capacity_kw: float = 48.0) -> dict:
        hours = np.arange(24)

        morning_peak = 14.0 * np.exp(-0.5 * ((hours - 9.5) / 1.8) ** 2)
        evening_peak = 28.0 * np.exp(-0.5 * ((hours - 20.0) / 2.2) ** 2)
        noise = self.rng.normal(0, 1.2, 24)
        demand = np.clip(base_load_kwh + morning_peak + evening_peak + noise, 10.0, None)

        solar = pv_capacity_kw * np.exp(-0.5 * ((hours - 12.5) / 2.8) ** 2)
        solar[hours < 6] = 0.0
        solar[hours > 18] = 0.0
        solar = np.clip(solar, 0.0, None)

        tariffs = np.full(24, 7.50)
        tariffs[(hours >= 0) & (hours < 6)] = 4.50
        tariffs[hours >= 23] = 4.50
        tariffs[(hours >= 17) & (hours <= 22)] = 13.00

        feed_in = np.full(24, 3.80)

        return {
            "hours": hours,
            "demand": demand,
            "solar": solar,
            "tariffs": tariffs,
            "feed_in": feed_in,
        }
