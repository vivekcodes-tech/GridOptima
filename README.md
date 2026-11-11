# GridOptima
### Microgrid Energy Management & Battery Storage Optimization

A simulation and optimization framework comparing **Baseline (No Storage)**, **Greedy Heuristics**, and **Bellman Dynamic Programming** for 24-hour battery scheduling under dynamic Time-of-Use (ToU) electricity tariffs.

---

## Project Overview

In modern smart microgrids with rooftop solar photovoltaics (PV) and Battery Energy Storage Systems (BESS), the timing of charging and discharging directly affects operational economics. Utilities enforce **Time-of-Use (ToU) tariffs**, charging significantly higher rates during evening peak hours.

**GridOptima answers a central engineering question:**
> *When should a battery charge, hold, or discharge to minimize a building's 24-hour electricity bill while preserving battery health and cyclic charge for subsequent days?*

---

## Mathematical Formulation

The optimization is framed as a discrete-time finite-horizon optimal control problem over a 24-hour cycle ($T = 24$ stages, $\Delta t = 1\text{ h}$):

### Objective Function
Minimize the net daily expenditure on electricity:
$$\min_{P_{\text{bat}}} \sum_{t=0}^{23} \Big[ P_{\text{import}}(t) \cdot \lambda_{\text{buy}}(t) - P_{\text{export}}(t) \cdot \lambda_{\text{sell}}(t) \Big]$$

Where:
* $P_{\text{import}}(t) = \max\Big(0, \text{Demand}(t) - \text{Solar}(t) - P_{\text{bat}}(t)\Big)$
* $P_{\text{export}}(t) = \max\Big(0, \text{Solar}(t) + P_{\text{bat}}(t) - \text{Demand}(t)\Big)$
* $\lambda_{\text{buy}}(t)$: Grid import tariff (4.50 off-peak, 7.50 normal, 13.00 peak in INR/kWh)
* $\lambda_{\text{sell}}(t)$: Feed-in compensation tariff (3.80 INR/kWh)
* $P_{\text{bat}}(t)$: Battery power delivered to load ($> 0$ discharge, $< 0$ charge)

### Operational Constraints
1. **Power Limit:**
   $$-P_{\text{max}} \le P_{\text{bat}}(t) \le P_{\text{max}}$$
2. **State-of-Charge (SoC) Bounds:**
   $$\text{SoC}_{\min} \le \text{SoC}(t) \le \text{SoC}_{\max} \quad (10\% \le \text{SoC} \le 95\%)$$
3. **Battery Conversion Efficiency:**
   * Charging: $E(t+1) = E(t) + P_{\text{ch}}(t) \cdot \Delta t \cdot \eta$
   * Discharging: $E(t+1) = E(t) - \frac{P_{\text{dis}}(t) \cdot \Delta t}{\eta}$
   * Where $\eta = \sqrt{\eta_{\text{roundtrip}}}$ is the one-way efficiency.
4. **Cyclic Terminal Boundary Condition:**
   $$\text{SoC}(24) \ge \text{SoC}(0)$$
   *Ensures the optimizer does not achieve artificial savings by draining the battery and leaving it depleted for the following day.*

---

## Optimization Algorithms

| Feature | Baseline | Greedy Heuristic | Dynamic Programming (DP) |
| :--- | :--- | :--- | :--- |
| **Storage Element** | None (Grid-tied PV only) | Battery Energy Storage (BESS) | Battery Energy Storage (BESS) |
| **Decision Rule** | Passive import/export | Instantaneous rules (myopic) | Global optimization (foresighted) |
| **Lookahead Horizon**| None | None ($t$) | Full 24-hour horizon ($t \dots 23$) |
| **Algorithmic Complexity** | $\mathcal{O}(1)$ | $\mathcal{O}(T)$ | $\mathcal{O}(T \cdot S^2)$ |
| **Typical Runtime** | $< 0.01\text{ ms}$ | $< 0.05\text{ ms}$ | $\approx 16\text{ ms}$ |
| **Optimality Guarantee** | Reference baseline | Sub-optimal | **Mathematically Guaranteed Global Optimum** |

### 1. Baseline Strategy
Excess solar is exported to the grid at feed-in tariff $\lambda_{\text{sell}}$; all deficits are imported from the grid at tariff $\lambda_{\text{buy}}(t)$.

### 2. Greedy Heuristic
Evaluates actions one hour at a time without future price knowledge:
* If $\text{Solar} > \text{Demand}$: Charge battery first up to $P_{\text{max}}$ and $\text{SoC}_{\max}$; export remainder.
* If $\text{Demand} > \text{Solar}$: Discharge battery to shave peak if tariff is $\ge 7.50\text{ INR/kWh}$.
* If tariff is off-peak ($< 7.50\text{ INR/kWh}$): Import cheap grid power, saving battery charge.

### 3. Dynamic Programming (Bellman Backward Induction)
Discretizes the continuous state-of-charge $\text{SoC} \in [0.10, 0.95]$ into $S = 101$ discrete energy levels.
* Solves the Bellman equation backward from $t = 23$ to $t = 0$:
  $$V_t(s) = \min_{s' \in \mathcal{S}} \Big[ \text{StageCost}_t(s, s') + V_{t+1}(s') \Big]$$
* Enforces the cyclic boundary penalty at $t = 24$:
  $$V_{24}(s) = \lambda_{\text{peak}} \cdot \max(0, s_0 - s) \cdot \text{Capacity}$$
* Reconstructs the exact optimal path forward from $s_0$.

---

## Benchmark Results

Running the benchmark with default specifications (32 kWh/h base load, 48 kW solar, 60 kWh battery, 20 kW inverter, 92% efficiency, initial SoC = 30%):

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

### Why does Dynamic Programming beat Greedy?
* **Premature Discharging in Greedy:** Greedy starts discharging battery energy during normal afternoon hours (7.50 INR/kWh) and depletes its charge before the evening peak finishes.
* **Price Arbitrage in DP:** Dynamic Programming "knows" that electricity costs 13.00 INR/kWh between 17:00 and 22:00. It strategically hoards solar charge until that exact window, yielding maximum financial savings while restoring its 30% charge level for the next day.

---

## Repository Structure

```text
GridOptima/
├── app.py                 # Academic Streamlit dashboard with numerical DP table inspection
├── run_simulation.py      # Standalone CLI runner & publication figure generator
├── benchmark_complexity.py# Empirical Big-O complexity scaling & convergence experiment
├── optimizers.py          # Baseline, Greedy, and Dynamic Programming algorithms
├── data_generator.py      # Dual-peak load curve, solar profile & ToU tariff engine
├── requirements.txt       # Dependencies (numpy, matplotlib, streamlit, plotly, pandas)
├── simulation_results.png # 4-panel benchmark figure exported by run_simulation.py
├── complexity_analysis.png# Empirical Big-O vs theoretical fit exported by benchmark_complexity.py
├── PROJECT_GUIDE.md       # Viva / Defense preparation guide & model answers
└── README.md              # Project documentation
```

---

## How to Run

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/GridOptima.git
cd GridOptima

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Standalone CLI Benchmark (Terminal Table + Plot)
Run the simulation in the terminal and export `simulation_results.png`:
```bash
python run_simulation.py
```
Custom parameters can be passed via command line flags:
```bash
python run_simulation.py --load 35 --pv 55 --capacity 80 --power 25
```

### 3. Empirical Complexity & Big-O Scaling Verification
Run the empirical asymptotic scaling benchmark and export `complexity_analysis.png`:
```bash
python benchmark_complexity.py
```

### 4. Interactive Web Dashboard
Launch the clean academic dashboard:
```bash
streamlit run app.py
```

---

## Academic Defense & Viva Q&A
See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for detailed viva preparation, expected professor questions, and step-by-step mathematical explanations.
