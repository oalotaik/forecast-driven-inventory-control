import numpy as np
import matplotlib.pyplot as plt


def forecast_periodic_review_inventory(
    df,
    lead_time,
    review_period,
    safety_factor,
    initial_inventory,
    use_rolling_ss=False,
    rolling_window=None,
    include_review_period_in_ss=True,
    plot=False,
):
    """
    Simulates a forecast-driven periodic review inventory control system with explicit safety stock tracking.
    Can project future inventory levels and orders based on forecasts beyond actual demand data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with columns:
        - 'period': time period index
        - 'demand': actual demand (can be NaN for future periods)
        - 'forecast': forecasted demand (can extend beyond actual demand)

    lead_time : int
        Lead time in periods between order placement and receipt.

    review_period : int
        Length of the review period (e.g., 3 means ordering every 3 periods).

    safety_factor : float
        Z-score for the desired service level (e.g., 1.645 for 95%).

    initial_inventory : float
        Initial total inventory level (including safety stock).

    use_rolling_ss : bool, default False
        If True, updates calculated safety stock every rolling_window based on rolling forecast error.

    rolling_window : int, optional
        Size of rolling window for safety stock error std calculation.
        Defaults to 2 * review_period if not provided.

    include_review_period_in_ss : bool, default True
        If True, uses sqrt(lead_time + review_period) in safety stock formula.
        If False, uses sqrt(lead_time) only.

    plot : bool, default False
        If True, shows two plots: inventory levels and demand vs forecast.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns:
        - 'order_quantity': order placed at each review point
        - 'on_hand_inventory': total inventory on hand before demand
        - 'safety_stock_target': calculated safety stock requirement
        - 'below_safety_stock': 1 if inventory falls below safety stock target
        - 'stockout': 1 if demand exceeded available inventory
        - 'ending_inventory': total inventory after demand
        - 'error': forecast error (positive = over-forecast, negative = under-forecast)
        - 'is_projection': 1 for periods without actual demand data
    """

    df = df.copy()
    n_periods = len(df)

    # Initialize columns
    df["order_quantity"] = 0.0
    df["on_hand_inventory"] = 0.0
    df["safety_stock_target"] = 0.0
    df["below_safety_stock"] = 0
    df["stockout"] = 0
    df["ending_inventory"] = 0.0
    df["error"] = np.nan
    df["is_projection"] = 0

    # Identify where actual demand ends and projections begin
    last_actual_period = df["demand"].last_valid_index()
    if last_actual_period is None:
        last_actual_period = -1  # All periods are projections

    # Calculate error only where we have actual demand
    df.loc[:last_actual_period, "error"] = (
        df.loc[:last_actual_period, "forecast"] - df.loc[:last_actual_period, "demand"]
    )

    # Mark projection periods
    df.loc[last_actual_period + 1 :, "is_projection"] = 1

    # Calculate initial safety stock based on historical errors (if any)
    historical_errors = df["error"].dropna()
    if len(historical_errors) > 0:
        std_error_all = np.std(historical_errors)
    else:
        # If no historical data, use a fraction of average forecast as proxy
        avg_forecast = df["forecast"].mean()
        std_error_all = 0.2 * avg_forecast if avg_forecast > 0 else 0

    # Calculate safety stock with optional review period inclusion
    if include_review_period_in_ss:
        time_factor = np.sqrt(lead_time + review_period)
    else:
        time_factor = np.sqrt(lead_time)

    base_safety_stock = safety_factor * std_error_all * time_factor

    if rolling_window is None:
        rolling_window = 2 * review_period

    # Initialize tracking variables
    df["safety_stock_target"] = base_safety_stock
    current_ss_target = base_safety_stock
    on_hand_inventory = initial_inventory
    in_transit_orders = [0.0] * lead_time

    for t in range(n_periods):
        # Update safety stock target if rolling option is enabled (only for historical periods)
        if (
            use_rolling_ss
            and t >= rolling_window
            and t % rolling_window == 0
            and t <= last_actual_period
        ):
            recent_errors = df["error"].iloc[max(0, t - rolling_window) : t]
            recent_errors = recent_errors.dropna()
            if len(recent_errors) > 0:
                std_error_rolling = np.std(recent_errors)
                current_ss_target = safety_factor * std_error_rolling * time_factor
                df.loc[t:, "safety_stock_target"] = current_ss_target

        # Receive any arriving order
        arriving_order = in_transit_orders.pop(0)
        on_hand_inventory += arriving_order

        # Record inventory before demand
        df.loc[t, "on_hand_inventory"] = on_hand_inventory

        # Place order if at review period
        if t % review_period == 0:
            # Calculate inventory position (on-hand + in-transit)
            inventory_position = on_hand_inventory + sum(in_transit_orders)

            # Calculate expected demand during lead time + review period
            future_periods = min(lead_time + review_period, n_periods - t)
            future_demand = df["forecast"].iloc[t : t + future_periods].sum()

            # Calculate order-up-to level (expected demand + safety stock)
            order_up_to_level = future_demand + current_ss_target

            # Place order to reach target level
            order_qty = max(0, order_up_to_level - inventory_position)
            df.loc[t, "order_quantity"] = order_qty
            in_transit_orders.append(order_qty)
        else:
            in_transit_orders.append(0.0)

        # Use actual demand if available, otherwise use forecast for projections
        if t <= last_actual_period:
            demand = df.loc[t, "demand"]
        else:
            demand = df.loc[t, "forecast"]

        # Fulfill demand
        if demand <= on_hand_inventory:
            on_hand_inventory -= demand
            stockout = 0
        else:
            # Stockout occurs
            on_hand_inventory = 0
            stockout = 1

        # Check if we're below safety stock target
        below_ss = 1 if on_hand_inventory < current_ss_target else 0

        # Record results
        df.loc[t, "ending_inventory"] = on_hand_inventory
        df.loc[t, "stockout"] = stockout
        df.loc[t, "below_safety_stock"] = below_ss

    # Plotting
    if plot:
        periods = df["period"]
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Inventory levels
        # Split historical and projected data
        hist_mask = df["is_projection"] == 0
        proj_mask = df["is_projection"] == 1

        # Historical inventory
        axs[0].plot(
            periods[hist_mask],
            df.loc[hist_mask, "ending_inventory"],
            label="Ending Inventory (Actual)",
            marker="o",
            color="blue",
        )

        # Projected inventory
        if proj_mask.any():
            # Connect the last historical point to first projection
            if hist_mask.any():
                bridge_periods = [
                    periods[hist_mask].iloc[-1],
                    periods[proj_mask].iloc[0],
                ]
                bridge_values = [
                    df.loc[hist_mask, "ending_inventory"].iloc[-1],
                    df.loc[proj_mask, "ending_inventory"].iloc[0],
                ]
                axs[0].plot(
                    bridge_periods,
                    bridge_values,
                    color="lightblue",
                    linestyle="--",
                    alpha=0.5,
                )

            axs[0].plot(
                periods[proj_mask],
                df.loc[proj_mask, "ending_inventory"],
                label="Ending Inventory (Projected)",
                marker="o",
                linestyle="--",
                color="lightblue",
            )

        # Safety stock target
        axs[0].plot(
            periods,
            df["safety_stock_target"],
            label="Safety Stock Target",
            linestyle="--",
            color="red",
            alpha=0.7,
        )
        axs[0].fill_between(
            periods,
            0,
            df["safety_stock_target"],
            alpha=0.2,
            color="red",
            label="Safety Stock Zone",
        )

        # Mark stockouts
        stockout_mask = df["stockout"] == 1
        if stockout_mask.any():
            axs[0].scatter(
                periods[stockout_mask & hist_mask],
                df.loc[stockout_mask & hist_mask, "ending_inventory"],
                color="black",
                s=100,
                marker="x",
                label="Stockout (Actual)",
                zorder=5,
            )
            axs[0].scatter(
                periods[stockout_mask & proj_mask],
                df.loc[stockout_mask & proj_mask, "ending_inventory"],
                color="gray",
                s=100,
                marker="x",
                label="Stockout (Projected)",
                zorder=5,
            )

        # Add vertical line at projection start
        if last_actual_period >= 0 and last_actual_period < len(periods) - 1:
            axs[0].axvline(
                x=periods[last_actual_period + 1],
                color="gray",
                linestyle=":",
                alpha=0.7,
                label="Projection Start",
            )

        axs[0].set_ylabel("Inventory Levels")
        axs[0].set_title("Inventory Levels Over Time")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # Plot 2: Demand vs Forecast
        axs[1].plot(
            periods[hist_mask],
            df.loc[hist_mask, "demand"],
            label="Actual Demand",
            marker="x",
            color="green",
        )
        axs[1].plot(
            periods,
            df["forecast"],
            label="Forecast",
            linestyle="--",
            marker="s",
            color="orange",
            markersize=4,
        )

        # Add vertical line at projection start
        if last_actual_period >= 0 and last_actual_period < len(periods) - 1:
            axs[1].axvline(
                x=periods[last_actual_period + 1],
                color="gray",
                linestyle=":",
                alpha=0.7,
            )

        axs[1].set_xlabel("Period")
        axs[1].set_ylabel("Demand")
        axs[1].set_title("Actual Demand vs Forecast")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return df
