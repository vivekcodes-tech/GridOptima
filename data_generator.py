"""
GridOptima - Official Synthetic Grid & Load Profile Generator
Implements standard IEEE residential load variations, solar irradiance curves,
and commercial Time-of-Use (ToU) electricity tariff regimes.
"""

import numpy as np


class GridEnvironment:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def get_24h_profile(self, base_load_kwh=35.0, pv_capacity_kw=50.0):
        """
        Returns hourly profiles for a 24-hour cycle:
        - Demand Profile (kWh)
        - Solar PV Generation (kWh)
        - Grid Tariff Structure (INR/kWh)
        - Feed-in Tariff (INR/kWh)
        """
        hours = np.arange(24)

        # 1. Standard Dual-Peak Residential / Commercial Load Profile
        # Morning peak: 08:00 - 11:00 (Breakfast, appliances, HVAC init)
        # Evening peak: 18:00 - 22:00 (Lighting, cooling, cooking, electronics)
        morning_peak = 14.0 * np.exp(-0.5 * ((hours - 9.5) / 1.8) ** 2)
        evening_peak = 28.0 * np.exp(-0.5 * ((hours - 20.0) / 2.2) ** 2)
        base_noise = self.rng.normal(0, 1.2, 24)
        demand = base_load_kwh + morning_peak + evening_peak + base_noise
        demand = np.clip(demand, 12.0, None)

        # 2. Solar PV Output Profile (Clear sky Gaussian approximation)
        # Solar window: 06:00 to 18:30, Peak insolation at 12:30
        solar_raw = pv_capacity_kw * np.exp(-0.5 * ((hours - 12.5) / 2.8) ** 2)
        solar_raw[hours < 6] = 0.0
        solar_raw[hours > 18] = 0.0
        solar = np.clip(solar_raw, 0.0, None)

        # 3. Dynamic Time-of-Use (ToU) Grid Tariffs (INR/unit)
        # Off-Peak (00:00 - 06:00, 23:00 - 24:00): ₹4.50 / kWh
        # Normal (06:00 - 17:00, 22:00 - 23:00): ₹7.50 / kWh
        # Peak Surcharge (17:00 - 22:00): ₹13.00 / kWh
        tariffs = np.full(24, 7.50)
        tariffs[(hours >= 0) & (hours < 6)] = 4.50
        tariffs[hours >= 23] = 4.50
        tariffs[(hours >= 17) & (hours <= 22)] = 13.00

        # 4. Feed-In Tariff (Export compensation for excess green energy)
        feed_in_tariff = np.full(24, 3.80)

        return {
            "hours": hours,
            "demand": demand,
            "solar": solar,
            "tariffs": tariffs,
            "feed_in": feed_in_tariff,
        }