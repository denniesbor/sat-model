import os
import time
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from capacity_analysis.io_model import InputOutputModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import jit
from tqdm import tqdm
from config import FIGURE_DIR, get_logger, ECONOMIC_DIR, NETWORK_DIR, LOG_DIR
from satellite_fleet import SatelliteAltitudes
from joblib import Parallel, delayed

logger = get_logger(__name__, log_file="simulation.log")
io = InputOutputModel(data_path=ECONOMIC_DIR)


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def select_failed_satellites(n_failures, weights, visible_sats):
    weights = np.array(weights) / np.sum(weights)
    if n_failures == 0:
        return []  # No satellites failed
    if len(weights) != visible_sats:
        logger.error("Mismatch between visible_sats and weight length.")
        raise ValueError("Mismatch between visible_sats and weight length.")
    return np.random.choice(
        range(visible_sats), size=n_failures, p=weights, replace=False
    )


@jit(nopython=True)
def calculate_capacity_metrics(
    failed_sats_capacities,  # Capacities of failed sats within coverage
    initial_total_capacity,  # Initial capacity for coverage area
    failed_sats_total,  # Total failed satellites in constellation
    total_constellation_sats,  # Total constellation size
    min_coverage_sats,  # Minimum sats needed for coverage
    degradation_factor,  # Steepness of sigmoid degradation
):
    # Direct capacity loss from failed satellites in coverage
    failed_capacity = np.sum(failed_sats_capacities)
    remaining_capacity = initial_total_capacity - failed_capacity

    # Calculate coverage loss proportionally
    coverage_loss_ratio = failed_sats_total / total_constellation_sats
    lost_in_coverage = int(coverage_loss_ratio * min_coverage_sats)

    # Sigmoid degradation model for capacity
    midpoint = 0.5  # Sigmoid midpoint (50% constellation failures)
    sigmoid_degradation = 1 / (
        1 + np.exp(-degradation_factor * (coverage_loss_ratio - midpoint))
    )

    # Apply degradation to remaining capacity
    remaining_capacity *= 1 - sigmoid_degradation
    return remaining_capacity


@jit(nopython=True)
def calculate_capacity_metrics(
    failed_sats_capacities,  # Array of capacities for failed satellites (e.g., in Gbps)
    initial_total_capacity,  # Initial capacity for the coverage area (Gbps)
    failed_sats_total,  # Total number of failed satellites in the constellation
    total_constellation_sats,  # Total constellation size
    min_coverage_sats,  # Minimum satellites required for effective coverage (unused here)
    degradation_factor,  # Additional degradation fraction (0 to 1)
):
    # Direct capacity loss from the failed satellites
    failed_capacity = np.sum(failed_sats_capacities)
    remaining_capacity = initial_total_capacity - failed_capacity

    # Compute coverage loss ratio: fraction of satellites lost
    coverage_loss_ratio = failed_sats_total / total_constellation_sats

    # Use a fixed sigmoid to model additional degradation from coverage loss.
    # Here k is a steepness parameter and midpoint is where degradation accelerates.
    k = 10.0  # Fixed steepness parameter (adjust as needed)
    midpoint = 0.5  # Midpoint at 50% failure
    sigmoid_value = 1 / (1 + np.exp(-k * (coverage_loss_ratio - midpoint)))

    # Scale the sigmoid by the degradation_factor (which ranges from 0 to 1)
    additional_degradation = degradation_factor * sigmoid_value

    # Apply the additional degradation to the remaining capacity.
    remaining_capacity *= 1 - additional_degradation
    return remaining_capacity


def simulate_failure(
    satellites,
    visible_sats,
    n_failures,
    capacities,
    weights,
    stats_df,
    total_cells,
    initial_total_capacity,
    assigned_cells,
    degradation_factor=15,
    capacity_only=False,
):
    """
    Simulate satellite failures and network impacts for a batch.
    """
    num_total_sats = len(satellites)

    # Calculate probability of visibility (ensuring nonzero)
    prob_visible = min((visible_sats / num_total_sats) + 0.001, 1.0)
    adjusted_failures = np.int32(n_failures * prob_visible)

    # Determine which satellites fail using precomputed weights
    failed_sats = select_failed_satellites(adjusted_failures, weights, visible_sats)
    if len(failed_sats) == 0:
        # No satellites failed, return initial capacity

        return {
            "capacity_gbps": initial_total_capacity,
            "capacity_per_cell_mbps": 0,
            "capacity_reduction_pct": 0,
            "cells_affected": 0,
            "cells_affected_pct": 0,
            "satellite_users_affected": 0,
            "total_economic_impact": 0,
            "direct_impact": 0,
            "indirect_impact": 0,
        }

    # Compute the number of network cells affected by the failed satellites
    n_affected_cells = np.sum(assigned_cells[failed_sats])
    # Calculate remaining capacity based on failed satellites and degradation
    remaining_capacity = calculate_capacity_metrics(
        capacities[failed_sats],
        initial_total_capacity,
        n_failures,
        len(satellites),
        len(capacities),
        degradation_factor,
    )

    # Determine capacity per remaining cell (converted to Mbps)
    remaining_cells = total_cells - n_affected_cells
    capacity_per_cell = (
        (remaining_capacity * 1000 / remaining_cells) if remaining_cells > 0 else 0
    )
    capacity_reduction_pct = (
        (initial_total_capacity - remaining_capacity) / initial_total_capacity
    ) * 100

    if capacity_only:
        result = pd.DataFrame(
            {
                "capacity_gbps": remaining_capacity,
                "capacity_per_cell_mbps": capacity_per_cell,
                "capacity_reduction_pct": capacity_reduction_pct,
                "cells_affected": 0,
                "cells_affected_pct": 0,
                "satellite_users_affected": 0,
                "total_economic_impact": 0,
                "direct_impact": 0,
                "indirect_impact": 0,
            }
        )
    else:
        impact = io.analyze_impact(stats_df, n_affected_cells)
        economic_shocks = io.prepare_shocks(impact)
        va_results = io.get_value_added(economic_shocks)
        d_impact = va_results["Direct Impact"]
        i_impact = va_results["Indirect Impact"]
        t_impact = va_results["Total Impact"]

        result = {
            "capacity_gbps": [remaining_capacity] * len(t_impact),
            "capacity_per_cell_mbps": [capacity_per_cell] * len(t_impact),
            "capacity_reduction_pct": [capacity_reduction_pct] * len(t_impact),
            "cells_affected": [n_affected_cells] * len(t_impact),
            "cells_affected_pct": [n_affected_cells / total_cells * 100]
            * len(t_impact),
            "satellite_users_affected": [float(impact["satellite_users_affected"])]
            * len(t_impact),
            "total_economic_impact": t_impact,
            "direct_impact": d_impact,
            "indirect_impact": i_impact,
        }

    df = pd.DataFrame(result)

    return df


def run_parallel_simulations(
    n_runs,
    sat_alts,
    cap_data,
    stats_df,
    failure_counts,
    constellation,
    degradation_factor=15,
    capacity_only=False,
    n_jobs=12,
):
    """
    Run simulate_failure in parallel and return a DataFrame of results.
    """
    capacities = cap_data["precomputed_capacities"]
    weights = cap_data["precomputed_weights"]
    initial_total_capacity = cap_data["initial_total_capacity"]
    total_cells = cap_data["total_cells"]
    assigned_cells = cap_data["assigned_cells"]
    visible_sats = int(cap_data["visible_sats"])
    sats = sat_alts[constellation]
    n_failures = failure_counts[constellation]

    logger.info(
        "Running %d parallel simulations with %d failures each...", n_runs, n_failures
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_failure)(
            sats,
            visible_sats,
            n_failures,
            capacities,
            weights,
            stats_df,
            total_cells,
            initial_total_capacity,
            assigned_cells,
            degradation_factor,
            capacity_only,
        )
        for _ in tqdm(range(n_runs), desc="Economic impact simulations")
    )

    logger.info("Parallel simulations complete.")

    return results
