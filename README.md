# GridOptima
### Microgrid Energy Management & Battery Storage Optimization

An optimization framework comparing **Baseline (No Storage)**, **Greedy Heuristics**, and **Bellman Dynamic Programming** for 24-hour battery scheduling under Time-of-Use (ToU) electricity tariffs.

---

## Project Overview

In a solar microgrid with battery storage, deciding *when* to charge or discharge directly controls the electricity bill. Utilities charge **Time-of-Use tariffs**, with high prices during evening peak hours.

**Core Question:**
> *When should a battery charge, hold, or discharge to minimize a building's 24-hour electricity bill while ensuring the battery is ready for tomorrow?*

---

## Problem Formulation (Simple & Direct)

Every hour of the day, the system decides whether to **charge the battery from solar**, **hold energy**, or **discharge power** to run the building.

### Goal: Minimize the 24-Hour Electricity Bill
$$\text{Daily Bill} = \sum_{t=0}^{23} \Big[ (\text{Grid Import} \times \text{Tariff Rate}) - (\text{Solar Export} \times \text{Export Rate}) \Big]$$

Where:
* **Grid Import:** Power purchased when building demand exceeds solar and battery discharge.
* **Solar Export:** Excess solar power sold back to the utility grid.
* **Tariff Rates:**
  - Off-Peak (Night, 00:00–06:00, 23:00–24:00): Rs. 4.50 / unit
  - Standard (Daytime, 06:00–17:00, 22:00–23:00): Rs. 7.50 / unit
  - Peak (Evening, 17:00–22:00): Rs. 13.00 / unit
* **Export Rate:** Rs. 3.80 / unit

### The 3 Operating Rules (Constraints)
1. **Inverter Speed Limit:** Battery cannot charge or discharge faster than inverter capacity (e.g., 20 kW max).
2. **Safe Battery Buffer:** State of charge stays between 10% reserve and 95% maximum.
3. **Day-to-Day Balance (Cyclic):** Battery must end at hour 24 with at least its initial charge (30%) so the next day is ready.

---

## Algorithms Compared

| Feature | Baseline | Greedy Heuristic | Dynamic Programming (DP) |
| :--- | :--- | :--- | :--- |
| **Battery Storage** | None (Grid-tied PV only) | Battery Energy Storage (BESS) | Battery Energy Storage (BESS) |
| **Decision Rule** | Passive import/export | Quick local rules (myopic) | Global optimization (foresighted) |
| **Lookahead** | None | Current hour only ($t$) | Full 24-hour horizon ($t \dots 23$) |
| **Time Complexity** | $\mathcal{O}(1)$ | $\mathcal{O}(T)$ | $\mathcal{O}(T \cdot S^2)$ |
| **Solve Time** | $< 0.01\text{ ms}$ | $< 0.05\text{ ms}$ | $\approx 16\text{ ms}$ |
| **Optimality** | Reference baseline | Sub-optimal | **Guaranteed Global Optimum** |

### 1. Baseline Strategy
Excess solar is sold to the grid; all deficits are imported from the grid at current tariffs.

### 2. Greedy Heuristic
Makes obvious decisions hour-by-hour without looking ahead:
* When solar exceeds demand: charge the battery first, export remainder.
* When demand exceeds solar: discharge the battery during standard and peak tariff hours.

### 3. Dynamic Programming (Bellman Equation)
Solves backwards from hour 23 down to hour 0:
$$\text{Best Cost from Hour } t = \min \Big[ \text{This Hour's Cost} + \text{Best Cost for the Rest of the Day} \Big]$$

Stores the answers in a 2D memoization table, ensuring it saves stored solar power specifically for the expensive evening peak.

---

## Benchmark Results

Running default specifications (32 kWh/h base load, 48 kW solar, 60 kWh battery, 20 kW inverter, 92% efficiency, initial SoC = 30%):

```text
------------------------------------------------------------------------
Performance Metric               Baseline     Greedy       Dynamic Prog
------------------------------------------------------------------------
Total Daily Cost (INR)           Rs.5904.69   Rs.5673.46   Rs.5418.18  
Cost Reduction vs Baseline       0.0%         3.92%        8.24%       
Peak Import (17-22h)             298.7 kWh    288.8 kWh    249.8 kWh   
Final State-of-Charge (SoC)      N/A          10.0%        30.4%       
Execution Time                   0.003 ms     0.018 ms     16.44 ms    
Subproblems Evaluated            0            24           244,824     
Time Complexity                  O(1)         O(T)         O(T * S^2)  
Space Complexity                 O(1)         O(T)         O(T * S)    
------------------------------------------------------------------------
[*] Brute-Force Exhaustive Paths: 101^24 ≈ 1.27 × 10^48 combinations
[*] Dynamic Programming reduces this to 244,824 evaluations in 16.44 ms!
```

---

## Repository Structure

```text
GridOptima/
├── app.py                 # Interactive dashboard with numerical DP table & theme toggle
├── run_simulation.py      # Standalone CLI runner (table output; --plot flag for figure)
├── benchmark_complexity.py# Big-O complexity scaling verification (--plot flag for figure)
├── optimizers.py          # Baseline, Greedy, and Dynamic Programming algorithms
├── data_generator.py      # Load curve, solar profile & tariff generator
├── requirements.txt       # Dependencies (numpy, matplotlib, streamlit, plotly, pandas)
├── PROJECT_GUIDE.md       # 1-day presentation prep guide & viva Q&A
└── README.md              # Project documentation
```

---

## How to Run

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Standalone CLI Benchmark (Terminal Table)
```bash
python run_simulation.py
```
*(Add `--plot` if you want to export an image: `python run_simulation.py --plot`)*

### 3. Complexity Verification
```bash
python benchmark_complexity.py
```

### 4. Interactive Web Dashboard
```bash
streamlit run app.py
```

---

## Academic Defense & Viva Q&A
See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for quick presentation scripts and expected teacher questions.
