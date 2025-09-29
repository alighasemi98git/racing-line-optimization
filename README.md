# Racing Line Optimization – Model Predictive Control

This project is part of the **EL2700 – Model Predictive Control** course at KTH. The focus is on designing an optimal racing line using convex optimization techniques, with the goal of minimizing curvature and thereby reducing lap time. The case study is the **Circuit de Monaco (Monte Carlo GP track)**.

## Project Overview

* **Problem Type:** Finite-time optimal control, convex optimization
* **Context:** Vehicle dynamics and racing line optimization
* **Key Features:**

  * Linearized vehicle model (bicycle model)
  * Convex optimization formulation for racing line smoothing
  * Constraints on track boundaries
  * Lap time estimation using curvature and friction model

## Tasks

1. **Racing Line Optimization**

   * Minimize total acceleration using convex optimization
   * Ensure vehicle stays within track width constraints
   * Implemented in `compute_racing_line()` (Python file `racing_line.py`)

2. **Lap Time Computation**

   * Compute curvature at each discretized track point
   * Apply tire friction limits to determine maximum velocity per segment
   * Compute lap time as the sum of travel times across all segments
   * Implemented in `compute_lap_time()`

3. **Monte Carlo GP Simulation**

   * Apply racing line optimization to the **Circuit de Monaco**
   * Compare optimized racing line vs. centerline trajectory
   * Evaluate improvement in lap time and trajectory smoothness

## Results

* **Racing Line vs. Centerline:**
  The optimized racing line results in a smoother path with larger effective radii in corners, leading to reduced curvature and higher feasible speeds.

* **Lap Time:**
  The optimized racing line achieves a shorter lap time compared to following the geometric centerline, validating the “slow is smooth, smooth is fast” principle.

## Repository Structure

```
.
├── src/
│   ├── task2.py                # Entry point script
│   ├── racing_line.py          # Core implementation (Q1–Q2)
│   ├── racetrack.py            # Provided class for track geometry
│
├── report/
│   └── Assignment2_GroupX.pdf  # Written report with answers & analysis
│
├── results/
│   ├── racing_line_plot.png    # Plot comparing optimized vs centerline trajectory
│   ├── lap_time_results.txt    # Lap time comparison output
│
└── README.md
```

## Usage Instructions

### Requirements

* Python 3.10+
* [CVXPY](https://www.cvxpy.org/)
* [MOSEK Solver](https://www.mosek.com/license/request/?i=acp) (free academic license)

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure that your **mosek.lic** license file is placed in:

* Linux/Mac: `$HOME/mosek/mosek.lic`
* Windows: `%USERPROFILE%\mosek\mosek.lic`

### Running the Project

1. Run the main script:

   ```bash
   python task2.py
   ```

2. The script will:

   * Compute the optimized racing line
   * Compare it against the centerline
   * Plot the trajectories
   * Output lap time comparisons

3. Results will be saved in the `results/` folder.

## Technologies

* **Python 3**
* **CVXPY + MOSEK** for convex optimization
* **Numerical simulation** for curvature and lap time

## Authors

* Group X (EL2700 – Model Predictive Control, 2025)

---

📄 This repository contains the Python implementation, results, and report for Assignment 2 of the MPC course, focusing on racing line optimization on the Monte Carlo circuit.
