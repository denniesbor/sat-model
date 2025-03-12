# %%
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import get_logger, SATELLITE_DIR, ECONOMIC_DIR
from capacity_analysis.capacity_model import stats_df
from capacity_analysis.monte_carlo_run import run_parallel_simulations
from capacity_analysis.coverage_estimator import precompute_satellite_data

logger = get_logger(__name__, "cum_failure.log")

DT = 10 * 60  # seconds


def load_satellite_data() -> dict:
    """Load satellite data from a pickle file."""
    path = SATELLITE_DIR / "satellite_alts.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sensitivity_results(filename: Path) -> dict:
    """Load sensitivity results from a pickle file."""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return dict(list(data.values())[0])


def calc_cum_failure(failure_rates: list, dt: int) -> float:
    """Calculate cumulative failure probability."""
    cumulative_lambda = sum(rate * dt for rate in failure_rates)
    return 1 - np.exp(-cumulative_lambda)


def aggregate_failures(up_set_rates: dict, dt: int) -> dict:
    """Aggregate failure rates for each satellite."""
    results = {}
    for constellation, sats in up_set_rates.items():
        results[constellation] = {}
        for sat, metrics in sats.items():
            fr = metrics["upset"]["Failure Rate"]
            results[constellation][sat] = calc_cum_failure(fr, dt)
    return results


def count_failures(cum_failures: dict, threshold: float = 0.08) -> dict:
    """Count the number of failures above a threshold for each constellation."""
    counts = {}
    for constellation, sats in cum_failures.items():
        counts[constellation] = sum(1 for prob in sats.values() if prob >= threshold)
    return counts


def get_constellation_results(
    constellation: str,
    filename: Path,
    degradation_factor: int = 15,
    n_runs: int = 1000,
    n_jobs: int = 12,
    threshold: float = 0.05,
) -> tuple:
    """Get results for a given constellation."""
    sat_alts = load_satellite_data()
    up_set_rates = load_sensitivity_results(filename)
    dt = DT

    cum_failures = aggregate_failures(up_set_rates, dt)
    failure_counts = count_failures(cum_failures, threshold=threshold)
    logger.info("Failure Counts: %s threshold: %f", failure_counts, threshold)

    if failure_counts[constellation] < 10:
        logger.info(
            "Skipping %s as less than 10 satellites failed: %d",
            constellation,
            failure_counts[constellation],
        )
        return None

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
    return result, failure_counts[constellation]


# %%
if __name__ == "__main__":
    economic_sim_dir = ECONOMIC_DIR / "economic_sim_dir"
    economic_sim_dir.mkdir(parents=True, exist_ok=True)

    simulation_dir = SATELLITE_DIR / "simulation_data"
    files = list(simulation_dir.glob("sensitivity_results_*yr_*_*_*.pkl"))
    constellation = "STARLINK"
    n_jobs = 4
    n_runs = 5

    logger.info("Processing economic impact for %s", constellation)
    for file in tqdm(files, desc="Processing simulation files"):
        econ_filename = economic_sim_dir / file.name.replace(
            "sensitivity_results", "economic_impact"
        ).replace(".pkl", ".csv")

        if econ_filename.exists():
            logger.info(
                "Skipping %s as economic impact results already exist", file.name
            )
            continue

        results = get_constellation_results(
            constellation, filename=file, n_runs=n_runs, n_jobs=n_jobs
        )

        if results is None:
            logger.info("No results for %s", file)
            continue

        try:
            results, failure_counts = results

            pcts = [2.5, 50, 97.5]

            econ_impacts = np.array([res["total_economic_impact"] for res in results])
            direct_impacts = np.array([res["direct_impact"] for res in results])
            indirect_impacts = np.array([res["indirect_impact"] for res in results])

            NAICS = results[0].index

            econ_pct = np.percentile(econ_impacts, pcts, axis=0)
            direct_pct = np.percentile(direct_impacts, pcts, axis=0)
            indirect_pct = np.percentile(indirect_impacts, pcts, axis=0)

            capacity_gbps = np.array([res["capacity_gbps"][0] for res in results])
            capacity_per_cell_mbps = np.array(
                [res["capacity_per_cell_mbps"][0] for res in results]
            )
            capacity_reduction_pct = np.array(
                [res["capacity_reduction_pct"][0] for res in results]
            )

            cap_gbps_pct = np.percentile(capacity_gbps, pcts)
            cap_cell_pct = np.percentile(capacity_per_cell_mbps, pcts)
            cap_reduction_pct = np.percentile(capacity_reduction_pct, pcts)

            summary_df = pd.DataFrame(
                {
                    "NAICS": NAICS,
                    "total_economic_impact_2.5": econ_pct[0],
                    "total_economic_impact_50": econ_pct[1],
                    "total_economic_impact_97.5": econ_pct[2],
                    "direct_impact_2.5": direct_pct[0],
                    "direct_impact_50": direct_pct[1],
                    "direct_impact_97.5": direct_pct[2],
                    "indirect_impact_2.5": indirect_pct[0],
                    "indirect_impact_50": indirect_pct[1],
                    "indirect_impact_97.5": indirect_pct[2],
                    "capacity_gbps_2.5": cap_gbps_pct[0],
                    "capacity_gbps_50": cap_gbps_pct[1],
                    "capacity_gbps_97.5": cap_gbps_pct[2],
                    "capacity_per_cell_mbps_2.5": cap_cell_pct[0],
                    "capacity_per_cell_mbps_50": cap_cell_pct[1],
                    "capacity_per_cell_mbps_97.5": cap_cell_pct[2],
                    "capacity_reduction_pct_2.5": cap_reduction_pct[0],
                    "capacity_reduction_pct_50": cap_reduction_pct[1],
                    "capacity_reduction_pct_97.5": cap_reduction_pct[2],
                }
            )

            summary_df["constellation"] = constellation
            summary_df["failure_counts"] = failure_counts
            summary_df.to_csv(econ_filename, index=False)

            logger.info("Saved economic impact results to %s", econ_filename)

        except Exception as e:
            logger.error("Error processing %s: %s", file, e)
            continue

# %%
