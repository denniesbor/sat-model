import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from skyfield.api import load, EarthSatellite
from skyfield.framelib import itrs
from tqdm import tqdm
import pytz
import requests
import logging
import os
from pathlib import Path
from typing import List, Dict
import pickle
import httpx

import spacetrack.operators as op
from spacetrack import SpaceTrackClient

from config import get_logger, space_track_login as login, SATELLITE_DIR, CONSTELLATIONS

logger = get_logger(__name__, log_file="download_tles.log")

eph = load(str(SATELLITE_DIR / "de421.bsp"))


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
                    tle = st.gp_history(
                        object_name=op.like(satellite_type),
                        epoch=epoch_range,
                        orderby="epoch asc",
                        format="tle",
                    )
                else:
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
    lines = [line.strip() for line in tle_string.split("\n") if line.strip()]
    satellites = {}

    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break

        line1 = lines[i]
        line2 = lines[i + 1]

        sat_id = line1[2:7].strip()
        year = int("20" + line1[18:20])
        day_of_year = float(line1[20:32])
        epoch = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

        if sat_id not in satellites:
            satellites[sat_id] = {"name": f"STARLINK-{sat_id}", "tles": []}

        tle_data = {"epoch": epoch, "line1": line1, "line2": line2}
        satellites[sat_id]["tles"].append(tle_data)

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

        historical_tles = get_bulk_satellite_tles(
            constellation,
            username,
            password,
            start_date=storm_start,
            end_date=storm_end,
        )
        logger.info(f"Fetching historical data for {constellation}... Done")

        tle_data = organize_tle_data(historical_tles)

        pickle_filename = SATELLITE_DIR / f"{constellation.lower()}_altitudes.pkl"

        if os.path.exists(pickle_filename):
            with open(pickle_filename, "rb") as f:
                all_satellite_altitudes[constellation] = pickle.load(f)
        else:
            logger.info(f"Calculating altitudes for {constellation}...")

            constellation_altitudes = calculate_satellite_altitudes_parallel(
                tle_data, storm_start, storm_end, step_minutes=resolution
            )

            all_satellite_altitudes[constellation] = constellation_altitudes

            logger.info(f"Saving {constellation} altitudes to {pickle_filename}...")
            with open(pickle_filename, "wb") as f:
                pickle.dump(constellation_altitudes, f)

        logger.info(f"Sleeping for 2 seconds before processing next constellation...")
        time.sleep(2)

    return all_satellite_altitudes


def preprocess_tle_data(sat_data, ts, window_start, window_end):
    """Preprocess TLE data for a satellite"""
    tle_with_epochs = []

    for tle in sat_data["tles"]:
        sat = EarthSatellite(tle["line1"], tle["line2"], sat_data["name"], ts)
        epoch = sat.epoch.utc_datetime()
        if not epoch.tzinfo:
            epoch = pytz.UTC.localize(epoch)
        tle_with_epochs.append({"epoch": epoch, "sat_obj": sat})

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


def is_in_earth_shadow(satellite, t, eph):
    """
    Determine if a satellite is in Earth's shadow using Skyfield.
    The satellite is in shadow if it is not sunlit.

    Args:
        satellite: A Skyfield satellite object (e.g., from TLE propagation).
        t: A Skyfield Time object.
        eph: An ephemeris (e.g., loaded from 'de421.bsp').

    Returns:
        bool: True if the satellite is in Earth's shadow, False otherwise.
    """
    return not satellite.at(t).is_sunlit(eph)


def process_satellite(sat_id, sat_data, time_steps, ts, window_start, window_end):
    """Process single satellite data with added velocities and Cartesian coordinates"""
    logger.info(f"Processing satellite {sat_id}...")
    ts_times = [
        ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in time_steps
    ]

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
    is_in_shadow = np.zeros(n_steps)

    sorted_tles, tle_intervals = preprocess_tle_data(
        sat_data, ts, window_start, window_end
    )

    for i, (t, ts_time) in enumerate(zip(time_steps, ts_times)):
        valid_tle = find_valid_tle(t, tle_intervals, sorted_tles)

        if valid_tle:
            sat_obj = valid_tle["sat_obj"]
            position = sat_obj.at(ts_time)
            subpoint = position.subpoint()

            latitudes[i] = subpoint.latitude.degrees
            longitudes[i] = subpoint.longitude.degrees
            altitudes[i] = subpoint.elevation.km

            itrs_position = position.frame_xyz(itrs)
            x_ecef, y_ecef, z_ecef = itrs_position.km
            x, y, z = (
                x_ecef * 1000.0,
                y_ecef * 1000.0,
                z_ecef * 1000.0,
            )

            itrs_velocity = position.frame_xyz_and_velocity(itrs)[1]
            vx_ecef, vy_ecef, vz_ecef = itrs_velocity.km_per_s
            vx, vy, vz = (
                vx_ecef * 1000.0,
                vy_ecef * 1000.0,
                vz_ecef * 1000.0,
            )

            cartesian_positions["x"][i] = x
            cartesian_positions["y"][i] = y
            cartesian_positions["z"][i] = z

            velocities["vx"][i] = vx
            velocities["vy"][i] = vy
            velocities["vz"][i] = vz

            in_shadow = is_in_earth_shadow(sat_obj, ts_time, eph)
            is_in_shadow[i] = in_shadow

            valid_indices.append(i)

    valid_indices = np.array(valid_indices)
    if len(valid_indices) > 0:
        latitudes = latitudes[valid_indices]
        longitudes = longitudes[valid_indices]
        altitudes = altitudes[valid_indices]
        times = [time_steps[i] for i in valid_indices]
        is_in_shadow = is_in_shadow[valid_indices]

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
        "is_in_shadow": is_in_shadow.tolist(),
    }


def calculate_satellite_altitudes_parallel(
    satellites_dict, window_start, window_end, step_minutes=30, num_workers=12
):
    """Calculate satellite positions in parallel"""
    ts = load.timescale()

    try:
        window_start = pytz.UTC.localize(window_start)
    except ValueError:
        pass
    try:
        window_end = pytz.UTC.localize(window_end)
    except ValueError:
        pass

    time_steps = []
    current_time = window_start
    while current_time <= window_end:
        time_steps.append(current_time)
        current_time += timedelta(minutes=step_minutes)

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
                failed_sat_id = futures[future]
                logger.error(f"Error processing satellite {failed_sat_id}: {e}")
                continue

    return altitude_data


if __name__ == "__main__":
    username, password = login()

    storm_start = datetime(2024, 5, 9)
    storm_end = datetime(2024, 5, 13)

    resolution = 10

    satellite_alts = process_and_save_satellite_data(
        username,
        password,
        storm_start,
        storm_end,
        constellations=CONSTELLATIONS,
        resolution=resolution,
    )
    logger.info("Done!")
