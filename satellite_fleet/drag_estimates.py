# %%
import pandas as pd
import numpy as np
import os
import sys
import pymsis
from pymsis import msis
from datetime import datetime
from pathlib import Path
from satellite_fleet.download_tles import process_and_save_satellite_data

from config import (
    get_logger,
    space_track_login as login,
    SATELLITE_DIR,
    FIGURE_DIR,
    CONSTELLATIONS,
)

from dataclasses import dataclass, field
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Custom imports
from viz import plot_propellant_by_constellation


@dataclass
class SatelliteAltitudes:
    storm_start: datetime = datetime(2024, 5, 9)
    storm_end: datetime = datetime(2024, 5, 13)
    constellations: list = field(default_factory=lambda: CONSTELLATIONS)
    resolution: int = 10
    satellite_alts_path: Path = field(
        default_factory=lambda: SATELLITE_DIR / "satellite_alts.pkl"
    )
    satellite_propellant_path: Path = field(
        default_factory=lambda: SATELLITE_DIR / "satellite_propellant.pkl"
    )
    satellite_comparison_path: Path = field(
        default_factory=lambda: SATELLITE_DIR / "satellite_drag_comparison.csv"
    )

    logger: object = field(
        default_factory=lambda: get_logger(__name__, log_file="drag_estimates.log")
    )
    satellite_alts: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.satellite_alts = self.load_satellite_alts()

    def load_satellite_alts(self):
        username, password = login()
        if self.satellite_alts_path.exists():
            self.logger.info("Loading satellite data from disk...")
            with open(self.satellite_alts_path, "rb") as f:
                return pickle.load(f)
        else:
            self.logger.info("Downloading satellite data...")
            satellite_alts = process_and_save_satellite_data(
                username,
                password,
                self.storm_start,
                self.storm_end,
                constellations=self.constellations,
                resolution=self.resolution,
            )
            with open(self.satellite_alts_path, "wb") as f:
                pickle.dump(satellite_alts, f)
            self.logger.info("Satellite data saved to disk.")
            return satellite_alts


def classify_satellite_altitudes(satellite_data, num_bins=8):
    all_heights = [
        sat_info["median_altitude"]
        for constellation in satellite_data.satellite_alts.values()
        for sat_info in constellation.values()
    ]

    min_altitude = np.nanmin(all_heights)
    max_altitude = np.nanmax(all_heights)
    bins = np.linspace(min_altitude, max_altitude, num_bins + 1)

    bin_labels = [
        f"Bin {i+1} ({int(bins[i])} km - {int(bins[i+1])} km)" for i in range(num_bins)
    ]

    satellite_classifications = []
    for constellation, satellites in satellite_data.satellite_alts.items():
        for sat_id, sat_data in satellites.items():
            try:
                median_altitude = sat_data["median_altitude"]
                bin_index = np.digitize(median_altitude - 0.01, bins) - 1
                bin_index = min(bin_index, num_bins - 1)

                satellite_classifications.append(
                    {
                        "Constellation": constellation,
                        "Satellite ID": sat_id,
                        "Satellite Name": sat_data["name"],
                        "Median Altitude (km)": median_altitude,
                        "Altitude Bin": bin_labels[bin_index],
                    }
                )
            except IndexError:
                continue

    return pd.DataFrame(satellite_classifications), bin_labels


# %%
def calculate_propellant_usage(results):
    """Calculate propellant usage from results dictionary."""
    try:
        total_by_sat = {}
        for sat_id in results:
            dt = np.diff([t.timestamp() for t in results[sat_id]["times"]])
            m_dots = np.array(results[sat_id]["mass_flow_rates"][:-1])
            propellant = np.sum(m_dots * dt)
            results[sat_id]["propellant_used"] = propellant
            total_by_sat[sat_id] = propellant

    except Exception as e:
        satellite_data.logger.error(f"Error calculating propellant usage: {e}")
    return results


def calculate_drag_propellant(
    satellite_alts, condition="actual", Isp=4000, A=5, Cd=2.2, G=6.67430e-11, M=5.972e24
):
    """
    Calculate drag forces and corresponding propellant usage for a constellation.

    Parameters:
      satellite_alts : dict
          Dictionary of satellite information (times, latitudes, longitudes, altitudes,
          velocities, and positions).
      condition : str, optional
          'actual', 'quiet', or 'storm' to choose the geophysical conditions.
      Isp : float, optional
          Specific impulse [s].
      A : float, optional
          Cross-sectional area [m²].
      Cd : float, optional
          Drag coefficient.
      G : float, optional
          Gravitational constant.
      M : float, optional
          Mass of the Earth [kg].

    Returns:
      results : dict
          Dictionary keyed by satellite IDs with computed properties.
    """

    results = {}
    density_stats = {"min": float("inf"), "max": float("-inf"), "sum": 0, "count": 0}

    conditions = {
        "quiet": {
            "f107": 65.0,  # Solar minimum
            "f107a": 65.0,  # Sustained minimum
            "aps": [2.0] * 7,  # Very quiet geomagnetic
        },
        "storm": {
            "f107": 230.0,  # Very high solar activity
            "f107a": 200.0,  # Sustained high activity
            "aps": [50.0] * 7,  # Strong geomagnetic storm
        },
    }

    for sat_id, sat_info in satellite_alts.items():
        try:
            times = sat_info["times"]
            lats = sat_info["latitudes"]
            lons = sat_info["longitudes"]
            alts = sat_info["altitudes"]

            sgp4_velocities = sat_info["velocities"]
            sgp_pos = sat_info["cartesian_positions"]
            vx = np.array(sgp4_velocities["vx"])
            vy = np.array(sgp4_velocities["vy"])
            vz = np.array(sgp4_velocities["vz"])
            x = np.array(sgp_pos["x"])
            y = np.array(sgp_pos["y"])
            z = np.array(sgp_pos["z"])

            v_vec = np.column_stack((vx, vy, vz))
            p_vec = np.column_stack((x, y, z))

            if condition == "actual":
                # Use pymsis to extract geophysical indices
                f107, f107a, aps = zip(*[pymsis.utils.get_f107_ap(t) for t in times])
                f107, f107a, aps = list(f107), list(f107a), list(aps)
            else:
                f107 = [conditions[condition]["f107"]] * len(times)
                f107a = [conditions[condition]["f107a"]] * len(times)
                aps = [conditions[condition]["aps"]] * len(times)

            # Run MSIS model (vectorized call)
            msis_op = msis.run(
                np.array(times),
                lons,
                lats,
                alts,
                f107s=f107,
                f107as=f107a,
                aps=aps,
                geomagnetic_activity=-1,
            )
            densities = msis_op[:, 0]

            # Use p_vec
            r_norm = np.linalg.norm(p_vec, axis=1)
            r_hat = p_vec / r_norm[:, None]

            # Compute radial and tangential velocity components
            v_radial = np.sum(v_vec * r_hat, axis=1)[:, None] * r_hat
            v_tangential = v_vec - v_radial
            v_t_mag = np.linalg.norm(v_tangential, axis=1)

            # Compute drag force: F_drag = 0.5 * Cd * density * (v_t)^2 * A
            drag_forces = 0.5 * Cd * densities * (v_t_mag**2) * A
            # Compute mass flow rate from drag force: mass_flow = F_drag / (Isp * g)
            mass_flow_rates = drag_forces / (Isp * 9.81)

            results[sat_id] = {
                "times": list(times),
                "densities": densities.tolist(),
                "tangential_velocities": v_t_mag.tolist(),
                "drag_forces": drag_forces.tolist(),
                "mass_flow_rates": mass_flow_rates.tolist(),
                "f107": f107,  # already lists
                "f107a": f107a,
                "ap": aps,
            }

            density_stats["min"] = min(density_stats["min"], np.min(densities))
            density_stats["max"] = max(density_stats["max"], np.max(densities))
            density_stats["sum"] += np.sum(densities)
            density_stats["count"] += len(densities)

        except Exception as e:
            satellite_data.logger.warning(f"Error processing satellite {sat_id}: {e}")
            continue

    return results


# %%
def load_or_calculate_results(satellite_data):
    # Try to load existing results
    if os.path.exists(satellite_data.satellite_propellant_path):
        print("Loading existing results...")
        with open(satellite_data.satellite_propellant_path, "rb") as f:
            results = pickle.load(f)
            actual_results = results["actual"]
            quiet_results = results["quiet"]
            storm_results = results["storm"]
    else:
        print("Calculating new results...")
        # Calculate all three conditions for all constellations
        actual_results = {
            constellation: calculate_propellant_usage(
                calculate_drag_propellant(
                    satellite_data.satellite_alts[constellation], condition="actual"
                )
            )
            for constellation in satellite_data.constellations
        }

        quiet_results = {
            constellation: calculate_propellant_usage(
                calculate_drag_propellant(
                    satellite_data.satellite_alts[constellation], condition="quiet"
                )
            )
            for constellation in satellite_data.constellations
        }

        storm_results = {
            constellation: calculate_propellant_usage(
                calculate_drag_propellant(
                    satellite_data.satellite_alts[constellation], condition="storm"
                )
            )
            for constellation in satellite_data.constellations
        }

        # Save results
        with open(satellite_data.satellite_propellant_path, "wb") as f:
            pickle.dump(
                {
                    "actual": actual_results,
                    "quiet": quiet_results,
                    "storm": storm_results,
                },
                f,
            )

    # Create tabulated comparison
    comparison_data = []
    for constellation in satellite_data.constellations:
        for alt in sorted(actual_results[constellation].keys()):
            row = {
                "Constellation": constellation,
                "Satellite ID": alt,
                "Quiet Propellant (kg)": quiet_results[constellation][alt][
                    "propellant_used"
                ],
                "Actual Propellant (kg)": actual_results[constellation][alt][
                    "propellant_used"
                ],
                "Storm Propellant (kg)": storm_results[constellation][alt][
                    "propellant_used"
                ],
                "Actual/Quiet Ratio": actual_results[constellation][alt][
                    "propellant_used"
                ]
                / quiet_results[constellation][alt]["propellant_used"],
                "Storm/Quiet Ratio": storm_results[constellation][alt][
                    "propellant_used"
                ]
                / quiet_results[constellation][alt]["propellant_used"],
            }
            comparison_data.append(row)

    # Convert to DataFrame for nice tabular output
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(2)

    return actual_results, quiet_results, storm_results, comparison_df


def compute_grouped_density_and_propellant(
    actual_results, quiet_results, storm_results, comparison_df, classification_df
):
    """
    Computes average densities for each satellite within nested constellations, merges with the classification DataFrame,
    and generates a grouped DataFrame with propellant usage and density statistics.

    Returns:
        pandas.DataFrame: Grouped DataFrame with summed propellant usage, mean densities, and satellite counts.
    """
    # Create copies to avoid modifying original DataFrames
    comparison_df = comparison_df.copy()
    classification_df = classification_df.copy()

    # Compute average densities for each satellite in each condition
    for condition, results in [
        ("actual", actual_results),
        ("quiet", quiet_results),
        ("storm", storm_results),
    ]:
        avg_densities = {
            sat_id: np.mean(data["densities"])
            for constellation in results.values()  # Loop through constellations
            for sat_id, data in constellation.items()  # Loop through satellites
        }
        comparison_df[f"{condition}_avg_density"] = comparison_df["Satellite ID"].map(
            avg_densities
        )

    # Merge classification and comparison DataFrames
    classification_df["Satellite ID"] = classification_df["Satellite ID"].astype(str)
    comparison_df["Satellite ID"] = comparison_df["Satellite ID"].astype(str)

    classified_df = classification_df.merge(
        comparison_df,
        on=["Constellation", "Satellite ID"],
        how="inner",
    )

    # Aggregate by constellation and altitude bin
    grouped_df = (
        classified_df.groupby(["Constellation", "Altitude Bin"])
        .agg(
            {
                "Quiet Propellant (kg)": "sum",
                "Actual Propellant (kg)": "sum",
                "Storm Propellant (kg)": "sum",
                "Actual/Quiet Ratio": sum,  # Ensure total ratios are meaningful
                "Storm/Quiet Ratio": sum,
                "actual_avg_density": np.nanmean,  # Ignores NaNs in averaging
                "quiet_avg_density": np.nanmean,
                "storm_avg_density": np.nanmean,
                "Satellite ID": "count",
            }
        )
        .rename(columns={"Satellite ID": "Satellite Count"})
    ).reset_index()

    # Dynamically extract all unique altitude bins from classification_df
    all_bins = classification_df["Altitude Bin"].unique()

    # Get unique constellations from grouped_df
    unique_constellations = grouped_df["Constellation"].unique()

    # Create a full index of (Constellation, Altitude Bin) pairs
    full_index = pd.MultiIndex.from_product(
        [unique_constellations, all_bins], names=["Constellation", "Altitude Bin"]
    )

    # Reindex with all bins and fill missing values with a small placeholder
    grouped_df = (
        grouped_df.set_index(["Constellation", "Altitude Bin"])
        .reindex(full_index, fill_value=0.001)
        .reset_index()
    )

    return grouped_df


# %%

if __name__ == "__main__":
    satellite_data = SatelliteAltitudes()

    if os.path.exists(satellite_data.satellite_propellant_path):
        satellite_data.logger.info("Loading existing results...")
        with open(satellite_data.satellite_propellant_path, "rb") as f:
            results = pickle.load(f)
            actual_results = results["actual"]
            quiet_results = results["quiet"]
            storm_results = results["storm"]

        comparison_df = pd.read_csv(satellite_data.satellite_comparison_path)

    else:
        actual_results, quiet_results, storm_results, comparison_df = (
            load_or_calculate_results(satellite_data)
        )

        comparison_df.to_csv(satellite_data.satellite_comparison_path, index=False)
        print("Results saved to starlink_drag_comparison.csv")

    # Read classification df
    classification_df, bin_labels = classify_satellite_altitudes(satellite_data)
    grouped_df = compute_grouped_density_and_propellant(
        actual_results, quiet_results, storm_results, comparison_df, classification_df
    )

    file_name = FIGURE_DIR / "propellant_by_constellation.png"
    plot_propellant_by_constellation(grouped_df, file_name)

# %%
