# Forecast-Driven Periodic Review Inventory Management System

## Overview
A Python implementation of a periodic review inventory control system that uses demand forecasts to optimize order quantities and safety stock levels. The system simulates inventory dynamics over time, supports rolling safety stock updates based on forecast accuracy, and can project future inventory levels using forecast data beyond historical demand periods.

## Problem Statement
- Problem being solved: Managing inventory in a periodic review system where orders can only be placed at fixed intervals (e.g., weekly, monthly) and arrive after a lead time, while maintaining adequate service levels despite demand uncertainty
- Key objectives:
  - Minimize stockouts while avoiding excessive inventory holding costs
  - Dynamically adjust safety stock based on forecast accuracy
  - Project future inventory positions to enable proactive decision-making
  - Track performance metrics including stockouts and periods below safety stock targets
- Constraints:
  - Orders can only be placed at review periods (not continuously)
  - Orders arrive after a fixed lead time
  - Demand uncertainty requires safety stock buffer
  - Initial inventory level is fixed

## Methods
- Algorithms/methods used:
  - Periodic review (R,S) inventory policy with order-up-to level
  - Safety stock calculation using forecast error standard deviation
  - Rolling window forecast error calculation for adaptive safety stock
  - Order quantity = max(0, Order-up-to level - Inventory position)
  - Inventory position = On-hand inventory + In-transit orders
- Key parameters and hyperparameters:
  - `lead_time`: Time between order placement and receipt (periods)
  - `review_period`: Interval between order opportunities (periods)
  - `safety_factor`: Z-score for desired service level (e.g., 1.645 for 95%)
  - `initial_inventory`: Starting inventory level
  - `use_rolling_ss`: Enable/disable adaptive safety stock updates
  - `rolling_window`: Window size for calculating rolling forecast error statistics
  - `include_review_period_in_ss`: Whether to include review period in safety stock time factor
  - Safety stock formula: safety_factor × std_error × sqrt(time_factor)
    - Where `time_factor` is either `lead_time` or `lead_time + review_period`

## Requirements
```bash
numpy
pandas
matplotlib
```
## Project Structure

```basic
├── data/              # Dataset files
│   ├── processed/     # Results of simulations
│   │   ├── simulation_results_future_forecasts.csv   # Results of simulation with future projections
│   │   └── simulation_results.csv   # Results of simulation without future projections
│   └── raw/           # Original data
│       ├── sample_future_forecasts.csv   # Sample raw data with future forecasts
│       └── sample.csv   # Sample raw data with no future forecasts; only historical demand and forecasts
├── images/            # Resulting plots from running experiment.py
│   ├── simulation_results_future_forecasts.png     # Resulting plot with future projections
│   └── simulation_results.png       # Resulting plot with only historical data (no future projections)
├── notebooks/         # Jupyter notebooks, currently empty
├── references/        # Safety Stock Formula used
│   └── safety-stock-formula.jpg      # Image of safety stock formula taken from Nicolas Vandeput LinkedIn
├── src/              # Source code
│   ├── experiment.py # Code to experiment with different inputs and settings
│   └── utils.py      # Contains main function for implementing the periodic inventory system
├── .gitignore        # Git ignore file
├── README.md         # Project documentation
└── requirements.txt  # Dependencies
```

## Results
Results are found in `data/processed` directory as CSV files. Below are only the plots of simulating inventory dynamics. The first plot shows results from `simulation_results.csv` where there is only historical demand and forecasts data (no future projections). The second plot shows results from `simulation_results_future_forecasts.csv` where there are forecasts beyond actual demand data.

![Simulation Results with No Future Projections](https://github.com/oalotaik/forecast-driven-inventory-control/blob/main/images/simulation_results.png)

![Simulation Results with Future Projections](https://github.com/oalotaik/forecast-driven-inventory-control/blob/main/images/simulation_results_future_forecasts.png)


## Setup and Installation
```bash
git clone https://github.com/oalotaik/forecast-driven-inventory-control.git
cd forecast-driven-inventory-control
pip install -r requirements.txt
```


## Citation
If this project was useful for your research, please cite:
```bash
@article{Alotaik2025project,
  title={Forecast-Driven Periodic Review Inventory Management System},
  author={Alotaik, O.},
  year={2025}
}
```


