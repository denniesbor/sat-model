# %%
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from skyfield.api import load, EarthSatellite, Topos
from skyfield.positionlib import Geocentric
from tqdm import tqdm
import pytz
import requests
import logging
import httpx

import spacetrack.operators as op
from spacetrack import SpaceTrackClient
from skyfield.api import load, EarthSatellite
from skyfield.api import wgs84
from skyfield.framelib import itrs

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import pickle

# Configs
from config import get_logger, space_track_login as login, SATELLITE_DIR

logger = get_logger(__name__, log_file="download_tles.log")


# %%
def fetch_satellite_tles(category="starlink"):
    """
    Fetch satellite IDs and TLEs from CelesTrak for a given category.

    Args:
        category (str): Satellite category (default: "starlink")
    Returns:
        list: List of dictionaries with satellite IDs and TLE data
    """
    base_url = "https://celestrak.org/NORAD/elements/gp.php"
    params = {"GROUP": category, "FORMAT": "json"}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def fetch_tle_data(norad_id, username, password, start_date=None, end_date=None):
    """
    Fetch TLE data from Space-Track API for specific NORAD ID.

    Args:
        norad_id: Satellite NORAD catalog ID
        username: Space-Track login credentials
        password: Space-Track login credentials
        start_date: Optional date range start for historical data
        end_date: Optional date range end for historical data

    Returns:
        str: Raw TLE data in standard format from Space-Track
    """
    base_url = "https://www.space-track.org"
    login_url = f"{base_url}/ajaxauth/login"

    # If no TLEs found in time range, fetch latest TLE
    with requests.Session() as session:
        session.post(login_url, data={"identity": username, "password": password})
        if start_date and end_date and start_date.date() == end_date.date():
            date = start_date.date().isoformat()
            query_url = (
                f"{base_url}/basicspacedata/query/class/tle/"
                f"norad_cat_id/{norad_id}/EPOCH/{date}/"
                "orderby/EPOCH asc/format/tle"
            )
        elif start_date and end_date:
            query_url = (
                f"{base_url}/basicspacedata/query/class/tle/"
                f"norad_cat_id/{norad_id}/EPOCH/{start_date.isoformat()}--{end_date.isoformat()}/"
                "orderby/EPOCH asc/format/tle"
            )
        else:
            query_url = (
                f"{base_url}/basicspacedata/query/class/tle_latest/"
                f"norad_cat_id/{norad_id}/orderby/EPOCH desc/limit/1/format/tle"
            )
        response = session.get(query_url).text

        # If no TLEs found, get latest
        if not response.strip():
            query_url = (
                f"{base_url}/basicspacedata/query/class/tle_latest/"
                f"norad_cat_id/{norad_id}/orderby/EPOCH desc/limit/1/format/tle"
            )
            response = session.get(query_url).text
        return response


def get_bulk_satellite_tles(
    satellite_type,
    username,
    password,
    start_date=None,
    end_date=None,
    retries=5,
    delay=5,
):
    """
    Fetch bulk TLE data using the Space-Track API.

    Uses gp_history for historical data (with an epoch range) and gp for current data.
    Implements simple retry logic to handle read timeouts.

    Args:
        satellite_type (str): Target satellite pattern (e.g., "STARLINK").
        username (str): Space-Track API username.
        password (str): Space-Track API password.
        start_date (datetime, optional): Start date for historical TLE data.
        end_date (datetime, optional): End date for historical TLE data.
        retries (int, optional): Number of retry attempts on timeout.
        delay (int, optional): Seconds to wait between retries.

    Returns:
        list: List of TLE lines as strings.
    """
    for attempt in range(retries):
        try:
            with SpaceTrackClient(identity=username, password=password) as st:
                if start_date and end_date:
                    epoch_range = f"{start_date.strftime('%Y-%m-%d')}--{end_date.strftime('%Y-%m-%d')}"
                    # Use gp_history for historical data.
                    tle = st.gp_history(
                        object_name=op.like(satellite_type),
                        epoch=epoch_range,
                        orderby="epoch asc",
                        format="tle",
                    )
                else:
                    # Use gp for current active satellites.
                    tle = st.gp(
                        object_name=op.like(satellite_type),
                        decay_date="null-val",
                        orderby="epoch desc",
                        format="tle",
                    )
            return tle
        except httpx.ReadTimeout as e:
            if attempt < retries - 1:
                logger.warning(
                    f"Read timeout on attempt {attempt + 1}. Retrying in {delay} seconds."
                )
                time.sleep(delay)
            else:
                logger.error(f"Failed to fetch TLE data after {retries} attempts.")
                raise e
    return ""


def organize_tle_data(tle_string):
    """
    Organize raw TLE string data into a structured dictionary format.

    Args:
        tle_string: Multi-line string containing TLE data sets

    Returns:
        dict: Satellite data organized as:
            {
                sat_id: {
                    'name': satellite name,
                    'tles': [
                        {
                            'epoch': datetime of TLE,
                            'line1': first line of TLE,
                            'line2': second line of TLE
                        }
                    ]
                }
            }
    """
    # Split into lines and remove empty lines
    lines = [line.strip() for line in tle_string.split("\n") if line.strip()]

    # Dictionary to store organized data
    satellites = {}

    # Process lines in pairs
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):  # Skip incomplete pairs
            break

        line1 = lines[i]
        line2 = lines[i + 1]

        # Extract satellite ID and name from Line 1
        # Line 1 format: 1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
        sat_id = line1[2:7].strip()  # NORAD Catalog Number

        # Extract epoch from Line 1
        year = int("20" + line1[18:20])
        day_of_year = float(line1[20:32])
        epoch = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

        # If this is a new satellite, initialize its entry
        if sat_id not in satellites:
            satellites[sat_id] = {"name": f"STARLINK-{sat_id}", "tles": []}

        # Add this TLE set to the satellite's data
        tle_data = {"epoch": epoch, "line1": line1, "line2": line2}
        satellites[sat_id]["tles"].append(tle_data)

    # Sort TLEs by epoch for each satellite
    for sat_id in satellites:
        satellites[sat_id]["tles"].sort(key=lambda x: x["epoch"])

    return satellites


def process_and_save_satellite_data(
    username,
    password,
    storm_start,
    storm_end,
    constellations=["STARLINK"],
    resolution=30,
):
    """
    Fetch, process, and save satellite TLE data to a pickle file.

    Args:
        username: Space-Track login credentials
        password: Space-Track login credentials
        storm_start: Start date for historical data
        storm_end: End date for historical data
        constellations: List of satellite constellations to process

    Returns:
        dict: Contains altitude data for each constellation.
    """
    all_satellite_altitudes = {}

    for constellation in constellations:
        logger.info(f"Fetching historical TLE data for {constellation}...")

        # Fetch historical constellation data
        historical_tles = get_bulk_satellite_tles(
            constellation,
            username,
            password,
            start_date=storm_start,
            end_date=storm_end,
        )
        logger.info(f"Fetching historical data for {constellation}... Done")

        # Organize TLE data
        tle_data = organize_tle_data(historical_tles)

        # Check if altitude data exists
        pickle_filename = SATELLITE_DIR / f"{constellation.lower()}_altitudes.pkl"

        if os.path.exists(pickle_filename):
            with open(pickle_filename, "rb") as f:
                all_satellite_altitudes[constellation] = pickle.load(f)
        else:
            logger.info(f"Calculating altitudes for {constellation}...")

            # Compute satellite altitudes
            constellation_altitudes = calculate_satellite_altitudes_parallel(
                tle_data, storm_start, storm_end, step_minutes=resolution
            )

            # Store in dictionary
            all_satellite_altitudes[constellation] = constellation_altitudes

            # Save results to a file
            logger.info(f"Saving {constellation} altitudes to {pickle_filename}...")
            with open(pickle_filename, "wb") as f:
                pickle.dump(constellation_altitudes, f)

        # Sleep to prevent overwhelming API (adjust sleep duration if needed)
        logger.info(f"Sleeping for 2 seconds before processing next constellation...")
        time.sleep(2)

    # print("Fetching current data...")
    # # Fetch current Starlink data
    # current_tles = get_bulk_satellite_tles("STARLINK", username, password)

    # print("Fetching ISS data...")
    # # Fetch ISS (Zarya) data
    # tle_data_zarya = fetch_tle_data(25544, username, password, storm_start, storm_end)

    return all_satellite_altitudes


def preprocess_tle_data(sat_data, ts, window_start, window_end):
    """Preprocess TLE data for a satellite"""
    tle_with_epochs = []

    # Process each TLE once
    for tle in sat_data["tles"]:
        sat = EarthSatellite(tle["line1"], tle["line2"], sat_data["name"], ts)
        epoch = sat.epoch.utc_datetime()
        if not epoch.tzinfo:
            epoch = pytz.UTC.localize(epoch)
        tle_with_epochs.append({"epoch": epoch, "sat_obj": sat})

    # Sort and create validity intervals
    sorted_tles = sorted(tle_with_epochs, key=lambda x: x["epoch"])
    tle_intervals = []

    for i, tle in enumerate(sorted_tles[:-1]):
        tle_intervals.append((tle["epoch"], sorted_tles[i + 1]["epoch"], tle))
    if sorted_tles:
        tle_intervals.append((sorted_tles[-1]["epoch"], window_end, sorted_tles[-1]))

    return sorted_tles, tle_intervals


def find_valid_tle(time, tle_intervals, sorted_tles):
    """Binary search for valid TLE"""
    left, right = 0, len(tle_intervals)
    while left < right:
        mid = (left + right) // 2
        interval = tle_intervals[mid]
        if interval[0] <= time < interval[1]:
            return interval[2]
        elif time < interval[0]:
            right = mid
        else:
            left = mid + 1
    return sorted_tles[0] if sorted_tles else None


def process_satellite(sat_id, sat_data, time_steps, ts, window_start, window_end):
    """Process single satellite data with added velocities and Cartesian coordinates"""
    # Pre-compute time objects
    ts_times = [
        ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in time_steps
    ]

    # Initialize arrays for better performance
    n_steps = len(time_steps)
    latitudes = np.zeros(n_steps)
    longitudes = np.zeros(n_steps)
    altitudes = np.zeros(n_steps)
    velocities = {
        "vx": np.zeros(n_steps),
        "vy": np.zeros(n_steps),
        "vz": np.zeros(n_steps),
    }
    cartesian_positions = {
        "x": np.zeros(n_steps),
        "y": np.zeros(n_steps),
        "z": np.zeros(n_steps),
    }
    valid_indices = []

    # Preprocess TLE data
    sorted_tles, tle_intervals = preprocess_tle_data(
        sat_data, ts, window_start, window_end
    )

    # Process each timestep
    for i, (t, ts_time) in enumerate(zip(time_steps, ts_times)):
        valid_tle = find_valid_tle(t, tle_intervals, sorted_tles)

        if valid_tle:
            sat_obj = valid_tle["sat_obj"]
            position = sat_obj.at(ts_time)
            subpoint = position.subpoint()

            # Geodetic (lat, lon, alt)
            latitudes[i] = subpoint.latitude.degrees
            longitudes[i] = subpoint.longitude.degrees
            altitudes[i] = subpoint.elevation.km

            # Convert to Earth-fixed (ECEF) frame for drag calculations
            # Convert to Earth-fixed (ECEF) frame for drag calculations
            itrs_position = position.frame_xyz(itrs)
            x_ecef, y_ecef, z_ecef = itrs_position.km
            x, y, z = (
                x_ecef * 1000.0,
                y_ecef * 1000.0,
                z_ecef * 1000.0,
            )  # Convert to meters

            # Get velocity in ECEF frame
            itrs_velocity = position.frame_xyz_and_velocity(itrs)[
                1
            ]  # Extract velocity part
            vx_ecef, vy_ecef, vz_ecef = itrs_velocity.km_per_s
            vx, vy, vz = (
                vx_ecef * 1000.0,
                vy_ecef * 1000.0,
                vz_ecef * 1000.0,
            )  # Convert to m/s

            # Store in dictionary
            cartesian_positions["x"][i] = x
            cartesian_positions["y"][i] = y
            cartesian_positions["z"][i] = z

            velocities["vx"][i] = vx
            velocities["vy"][i] = vy
            velocities["vz"][i] = vz

            valid_indices.append(i)

    # Trim arrays to valid indices
    valid_indices = np.array(valid_indices)
    if len(valid_indices) > 0:
        latitudes = latitudes[valid_indices]
        longitudes = longitudes[valid_indices]
        altitudes = altitudes[valid_indices]
        times = [time_steps[i] for i in valid_indices]

        for key in velocities:
            velocities[key] = velocities[key][valid_indices].tolist()
        for key in cartesian_positions:
            cartesian_positions[key] = cartesian_positions[key][valid_indices].tolist()
    else:
        latitudes = longitudes = altitudes = np.array([])
        times = []
        velocities = {k: [] for k in velocities}
        cartesian_positions = {k: [] for k in cartesian_positions}

    return sat_id, {
        "name": sat_data["name"],
        "times": times,
        "latitudes": latitudes.tolist(),
        "longitudes": longitudes.tolist(),
        "altitudes": altitudes.tolist(),
        "velocities": velocities,
        "cartesian_positions": cartesian_positions,
        "mean_altitude": np.nanmean(altitudes) if len(altitudes) > 0 else None,
        "median_altitude": np.nanmedian(altitudes) if len(altitudes) > 0 else None,
        "altitude_change": np.ptp(altitudes) if len(altitudes) > 0 else None,
    }


def calculate_satellite_altitudes_parallel(
    satellites_dict, window_start, window_end, step_minutes=30, num_workers=12
):
    """Calculate satellite positions in parallel"""
    # Load timescale once
    ts = load.timescale()

    # Ensure proper timezone
    try:
        window_start = pytz.UTC.localize(window_start)
    except ValueError:
        pass
    try:
        window_end = pytz.UTC.localize(window_end)
    except ValueError:
        pass

    # Generate time steps
    time_steps = []
    current_time = window_start
    while current_time <= window_end:
        time_steps.append(current_time)
        current_time += timedelta(minutes=step_minutes)

    # Process satellites in parallel
    altitude_data = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_satellite,
                sat_id,
                sat_data,
                time_steps,
                ts,
                window_start,
                window_end,
            ): sat_id
            for sat_id, sat_data in satellites_dict.items()
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing satellites"
        ):
            try:
                sat_id, result = future.result()
                altitude_data[sat_id] = result
            except Exception as e:
                logger.error(f"Error processing satellite {sat_id}: {e}")
                continue

    return altitude_data


if __name__ == "__main__":
    username, password = login()

    # For Gannon storm / historical data:
    storm_start = datetime(2024, 5, 9)
    storm_end = datetime(2024, 5, 13)

    # Download oneweb, kuiper, and starlink data
    constellations = ["ONEWEB", "KUIPER", "STARLINK"]
    resolution = 30

    # Fetch and process satellite data
    satellite_alts = process_and_save_satellite_data(
        username,
        password,
        storm_start,
        storm_end,
        constellations=constellations,
        resolution=resolution,
    )
    logger.info("Done!")
