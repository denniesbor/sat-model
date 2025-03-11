import os
from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from config import get_logger, SATELLITE_DIR, FIGURE_DIR, CONSTELLATIONS, ECONOMIC_DIR
from capacity_analysis.capacity_model import (
    continental_us_states,
    continental_hex_gdf,
    stats_df,
)
from capacity_analysis.monte_carlo_run import run_parallel_simulations
from capacity_analysis.coverage_estimator import precompute_satellite_data

logger = get_logger(__name__, "cum_failire.log")


def load_satellite_data():
    path = SATELLITE_DIR / "satellite_alts.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sensitivity_results(filename):

    with open(filename, "rb") as f:
        data = pickle.load(f)
    return dict(list(data.values())[0])


def calc_cum_failure(failure_rates, dt):
    cumulative_lambda = sum(rate * dt for rate in failure_rates)
    return 1 - np.exp(-cumulative_lambda)


def aggregate_failures(up_set_rates, dt):
    results = {}
    for constellation, sats in up_set_rates.items():
        results[constellation] = {}
        for sat, metrics in sats.items():
            fr = metrics["upset"]["Failure Rate"]
            results[constellation][sat] = calc_cum_failure(fr, dt)
    return results


def count_failures(cum_failures, threshold=0.08):
    counts = {}
    for constellation, sats in cum_failures.items():
        counts[constellation] = sum(1 for prob in sats.values() if prob >= threshold)
    return counts


def get_constellation_results(
    constellation,
    filename,
    degradation_factor=15,
    n_runs=1000,
    n_jobs=12,
    threshold=0.05,
):
    sat_alts = load_satellite_data()
    up_set_rates = load_sensitivity_results(filename)
    dt = 10 * 60  # seconds

    cum_failures = aggregate_failures(up_set_rates, dt)
    failure_counts = count_failures(cum_failures, threshold=threshold)
    logger.info("Failure Counts: %s threshold: %d", failure_counts, threshold)

    cap_data = precompute_satellite_data(
        sat_alts, stats_df, constellation, plot_fig=False
    )

    result = run_parallel_simulations(
        n_runs,
        sat_alts,
        cap_data,
        stats_df,
        failure_counts,
        constellation,
        degradation_factor=degradation_factor,
        capacity_only=False,
        n_jobs=n_jobs,
    )
    return result


if __name__ == "__main__":

    # Create directory for economic simulation results
    economic_sim_dir = ECONOMIC_DIR / "economic_sim_dir"
    economic_sim_dir.mkdir(parents=True, exist_ok=True)

    simulation_dir = SATELLITE_DIR / "simulation_data"
    files = list(simulation_dir.glob("sensitivity_results_*yr_*_*_*.pkl"))
    constellation = "STARLINK"
    n_jobs = 6
    n_runs = 30  # Will adjust later

    logger.info("Processing economic impact for %s", constellation)
    for file in tqdm(files, desc="Processing simulation files"):

        # save file
        econ_filename = economic_sim_dir / file.name.replace(
            "sensitivity_results", "economic_impact"
        )
        if econ_filename.exists():
            logger(f"Skipping {file.name} as economic impact results already exist")
            continue

        # Process the simulation file for the given constellation
        results = get_constellation_results(
            constellation, filename=file, n_runs=n_runs, n_jobs=n_jobs
        )

        with open(econ_filename, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger(f"Saved economic impact results to {econ_filename}")
