# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from satellite_fleet import SatelliteAltitudes
from config import ECONOMIC_DIR, get_logger, FIGURE_DIR
from viz import plot_satellite_visibility

logger = get_logger(__name__, log_file="coverage_estimator.log")

EARTH_RADIUS_KM = 6371

def haversine_distance_and_components(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate great-circle distance between two points using the Haversine formula.

    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)

    Returns:
        float: Distance between points in kilometers.
    """
    R = EARTH_RADIUS_KM
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def get_conus_centroid(continental_us_states) -> tuple:
    """
    Calculate the centroid of the continental United States.

    Returns:
        tuple: (latitude, longitude) of the CONUS centroid.
    """
    conus_centroid = continental_us_states.geometry.unary_union.centroid
    center_lat, center_lon = conus_centroid.y, conus_centroid.x
    return center_lat, center_lon


def calculate_elevation(
    ground_distance: float, altitude: float, R: float = EARTH_RADIUS_KM
) -> float:
    """
    Calculate satellite elevation angle from a ground point using spherical geometry.

    Args:
        ground_distance: Great circle distance in km between ground point and satellite nadir.
        altitude: Satellite altitude in km.
        R: Earth radius in km (default: 6371).

    Returns:
        float: Elevation angle in degrees, or None if the satellite is beyond maximum visibility distance.
    """
    Rs = R + altitude
    max_ground_distance = R * np.arccos(R / Rs)
    if ground_distance > max_ground_distance:
        return None

    central_angle = ground_distance / R
    slant_range = np.sqrt(R**2 + Rs**2 - 2 * R * Rs * np.cos(central_angle))
    cos_theta_prime = (R**2 + slant_range**2 - Rs**2) / (2 * R * slant_range)
    theta_prime = np.degrees(np.arccos(cos_theta_prime))
    elevation = theta_prime - 90
    return elevation


def check_visibility_across_us(
    t: int, satellite_sats: dict, bins: np.ndarray, min_elevation: float = 25
) -> dict:
    """
    Check satellite visibility across continental US grid points.

    Args:
        t: Time index for satellite positions.
        satellite_sats: Dictionary of satellite data with positions and altitudes.
        bins: Altitude bin edges for grouping satellites.
        min_elevation: Minimum elevation angle in degrees (default: 25).

    Returns:
        dict: Satellites visible from US grouped by altitude plane,
              e.g., {plane_index: [{'sat_id': ..., 'lat': ..., 'lon': ...}, ...]}.
    """
    lat_points = np.linspace(25, 49, 20)
    lon_points = np.linspace(-125, -67, 30)
    visible_by_plane = {i: [] for i in range(len(bins) - 1)}

    for sat_id, sat_data in satellite_sats.items():
        median_alt = sat_data["median_altitude"]
        plane_idx = np.digitize(median_alt, bins) - 1
        plane_idx = min(max(plane_idx, 0), len(bins) - 2)

        sat_lat = sat_data["latitudes"][t]
        sat_lon = sat_data["longitudes"][t]
        visible = False

        for lat in lat_points:
            for lon in lon_points:
                ground_distance = haversine_distance_and_components(
                    lat, lon, sat_lat, sat_lon
                )
                elevation = calculate_elevation(ground_distance, median_alt)
                if elevation is not None and elevation >= min_elevation:
                    visible = True
                    break
            if visible:
                break

        if visible:
            visible_by_plane[plane_idx].append(
                {"sat_id": sat_id, "lat": sat_lat, "lon": sat_lon}
            )

    return visible_by_plane


def generate_satellite_distribution(n_sats: int) -> tuple:
    """
    Generate satellite distribution into groups with a total count matching n_sats.
    """
    distribution = {
        "64QAM_high": 0.6,
        "16APSK_mid": 0.3,
        "QPSK_low": 0.1,
    }
    modulation_efficiencies = {
        "64QAM_high": 5.5547,
        "16APSK_mid": 2.967,
        "QPSK_low": 0.989,
    }

    group_counts = {}
    total_assigned = 0

    for i, (key, percentage) in enumerate(distribution.items()):
        if i == len(distribution) - 1:
            count = n_sats - total_assigned
        else:
            count = round(n_sats * percentage)
            total_assigned += count
        group_counts[key] = count

    return group_counts, modulation_efficiencies


def generate_satellite_capacity(
    n_sats: int, bandwidth_GHz: float = 2.5
) -> tuple:
    """
    Generate satellite capacities using a bandwidth parameter.
    Bandwidth_GHz represents the available bandwidth (in GHz) per satellite.
    """
    groups, efficiencies = generate_satellite_distribution(n_sats)
    capacities = []
    satellites = []

    for group, count in groups.items():
        base_capacity = bandwidth_GHz * efficiencies[group]
        for _ in range(int(count)):
            variation_factor = np.random.beta(5, 2)
            degradation_factor = np.random.uniform(0.7, 1.0)
            adjusted_capacity = (
                base_capacity * degradation_factor * (0.7 + variation_factor * 0.6)
            )
            adjusted_capacity = min(adjusted_capacity, base_capacity)
            satellites.append({"group": group, "capacity": adjusted_capacity})
            capacities.append(adjusted_capacity)

    weights = np.array(capacities) / np.nanmax(capacities)
    return np.array(capacities), weights


def analyze_all_timestamps(
    satellite_sats: dict, bins: np.ndarray, n: int, constellation: str
) -> list:
    """
    Analyze satellite visibility across continental US for multiple timestamps.

    Args:
        satellite_sats: Dictionary of satellite data with positions and altitudes.
        bins: Altitude bin edges for grouping satellites.
        n: Number of timestamps to analyze.

    Returns:
        list: Analysis results for each timestamp containing:
              {'timestamp': t, 'planes': {plane_idx: {'altitude_range': (min_alt, max_alt),
                                                      'num_satellites': count,
                                                      'coverage_radius': value,
                                                      'satellites': [{sat_id, lat, lon}, ...]}}}.
    """
    save_path = ECONOMIC_DIR / f"{constellation.lower()}_visibility_analysis.pkl"

    if os.path.exists(save_path):
        logger.info(f"Loading existing analysis from {save_path}")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    logger.info("Generating new visibility analysis...")
    visibility_data = []
    timestamps = np.linspace(0, n, 20, dtype=int)

    for t in tqdm(timestamps, desc="Analyzing timestamps"):
        try:
            time_data = {"timestamp": t, "planes": {}}
            valid_timestamp = any(
                t < len(sat_data["times"])
                for sat_id, sat_data in satellite_sats.items()
            )
            if not valid_timestamp:
                logger.info(f"Skipping timestamp {t} - beyond data range")
                continue

            try:
                visibility = check_visibility_across_us(t, satellite_sats, bins)
            except IndexError:
                logger.warning(f"Index error for timestamp {t}, skipping...")
                continue

            for plane_idx, sats in visibility.items():
                bin_min, bin_max = bins[plane_idx], bins[plane_idx + 1]
                plane_data = {
                    "altitude_range": (bin_min, bin_max),
                    "num_satellites": len(sats),
                    "coverage_radius": (
                        0.5 * (bin_min + bin_max) if len(sats) > 0 else 0
                    ),
                    "satellites": sats,
                }
                time_data["planes"][plane_idx] = plane_data

            visibility_data.append(time_data)

        except Exception as e:
            logger.error(f"Error processing timestamp {t}: {str(e)}")
            continue

    if visibility_data:
        logger.info(f"Saving analysis to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(visibility_data, f)
        logger.info("Analysis complete!")
    else:
        logger.info("No valid data to save!")

    return visibility_data


def precompute_satellite_data(
    satellite_sats: dict, stats_df: pd.DataFrame, constellation: str, plot_fig: bool = False
) -> dict:
    """
    Precompute satellite distribution and capacity data, save to a pickle file,
    and generate a satellite visibility plot.

    Args:
        satellite_sats (dict): Dictionary containing satellite data.
        stats_df (pd.DataFrame): Statistics DataFrame.
        constellation (str): Name of the satellite constellation.
        plot_fig (bool): Whether to plot the satellite visibility figure.

    Returns:
        dict: Dictionary with precomputed data.
    """
    f_name = ECONOMIC_DIR / f"{constellation.lower()}_precomputed_data.pkl"
    if os.path.exists(f_name):
        logger.info(f"Loading precomputed data from {f_name}")
        with open(f_name, "rb") as f:
            return pickle.load(f)

    logger.info("Precomputed data file not found; computing data...")

    const_sats = satellite_sats[constellation]
    median_heights = np.asarray([sat["median_altitude"] for sat in const_sats.values()])
    min_altitude = np.nanmin(median_heights)
    max_altitude = np.nanmax(median_heights)
    bins = np.linspace(min_altitude, max_altitude, 9)

    n = len(list(const_sats.items())[1][1]["times"])
    time_indices = np.linspace(0, n - 1, 20, dtype=int)
    satellite_times = [list(const_sats.items())[1][1]["times"][i] for i in time_indices]
    visibility_data = analyze_all_timestamps(const_sats, bins, n, constellation)
    logger.info(f"Visibility analysis of {constellation} constellation is complete.")

    satellite_counts = []
    for time_data, timestamp in zip(visibility_data, satellite_times):
        total_sats = sum(
            plane_data["num_satellites"] for plane_data in time_data["planes"].values()
        )
        satellite_counts.append(
            {"timestamp": timestamp, "total_satellites": total_sats}
        )
    visibility_df = pd.DataFrame(satellite_counts)
    visible_sats = np.ceil(visibility_df["total_satellites"].mean())

    capacities, weights = generate_satellite_capacity(visible_sats)

    initial_total_capacity = capacities.sum()
    num_total_sats = len(const_sats)
    normalized_weights = weights / np.sum(weights)
    total_cells = len(stats_df)
    assigned_cells = (normalized_weights * total_cells).astype(int)

    precomputed_data = {
        "precomputed_capacities": capacities,
        "precomputed_weights": weights,
        "initial_total_capacity": initial_total_capacity,
        "num_total_sats": num_total_sats,
        "normalized_weights": normalized_weights,
        "total_cells": total_cells,
        "assigned_cells": assigned_cells,
        "visible_sats": visible_sats,
        "visibility_df": visibility_df,
    }

    with open(f_name, "wb") as f:
        pickle.dump(precomputed_data, f)
    logger.info(f"Precomputed data saved to {f_name}")

    if plot_fig:
        plot_satellite_visibility(
            visibility_df,
            file_name=FIGURE_DIR / f"{constellation.lower()}_satellite_visibility.png",
        )

    return precomputed_data
