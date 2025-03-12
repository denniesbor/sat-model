# %%
import os
from pathlib import Path
from io import StringIO, BytesIO

import requests
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from spacepy import pycdf
import h5py

from config import get_logger, FLUX_DIR

logger = get_logger(__name__, log_file="download_tles.log")


def read_epam(input_file=FLUX_DIR / "exported_data.txt", output_file=FLUX_DIR / "epam.csv"):
    """Reads and processes EPAM data, filtering for May 10-15."""
    logger.info(f"Reading EPAM data from {input_file}")

    cols = [
        "year", "day", "hr", "min", "sec", "fp_year", "fp_doy", "ACEepoch", "P1", "P2",
        "P3", "P4", "P5", "P6", "P7", "P8", "unc_P1", "unc_P2", "unc_P3", "unc_P4",
        "unc_P5", "unc_P6", "unc_P7", "unc_P8", "DE1", "DE2", "DE3", "DE4", "unc_DE1",
        "unc_DE2", "unc_DE3", "unc_DE4", "W3", "W4", "W5", "W6", "W7", "W8", "unc_W3",
        "unc_W4", "unc_W5", "unc_W6", "unc_W7", "unc_W8", "E1p", "E2p", "E3p", "E4p",
        "FP5p", "FP6p", "FP7p", "unc_E1p", "unc_E2p", "unc_E3p", "unc_E4p", "unc_FP5p",
        "unc_FP6p", "unc_FP7p", "Z2", "Z2A", "Z3", "Z4", "unc_Z2", "unc_Z2A", "unc_Z3",
        "unc_Z4", "P1p", "P2p", "P3p", "P4p", "P5p", "P6p", "P7p", "P8p", "unc_P1p",
        "unc_P2p", "unc_P3p", "unc_P4p", "unc_P5p", "unc_P6p", "unc_P7p", "unc_P8p",
        "E1", "E2", "E3", "E4", "FP5", "FP6", "FP7", "unc_E1", "unc_E2", "unc_E3",
        "unc_E4", "unc_FP5", "unc_FP6", "unc_FP7", "lifetime",
    ]

    df = pd.read_csv(input_file, delimiter="\\s+", names=cols)

    seconds = df["sec"].apply(lambda x: f"{float(x):05.2f}")

    datetime_strings = (
        df["year"].astype(str) + " " + df["day"].astype(str) + " " +
        df["hr"].astype(str) + ":" + df["min"].astype(str) + ":" + seconds
    )

    df["datetime"] = pd.to_datetime(datetime_strings, format="%Y %j %H:%M:%S.%f")
    df = df[
        (df["datetime"].dt.month == 5) &
        (df["datetime"].dt.day >= 10) &
        (df["datetime"].dt.day <= 15)
    ]

    epam_cols = [
        col for col in df.columns if (
            col.startswith(("P", "DE", "W", "E", "FP", "Z")) and not col.startswith("unc_")
        )
    ]

    df = process_timeseries(
        df, instrument="ACE EPAM", cols_to_use=epam_cols, invalid_markers=-999.9,
        sampling_time=5, apply_savgol=False,
    )

    df["datetime"] = df.index

    df.to_csv(output_file, index=False)

    logger.info(f"EPAM data saved to {output_file}")
    return df


def get_stereo_het_data(url, output_file):
    """Fetches and processes STEREO/HET data."""
    response = requests.get(url)
    txt = response.text
    sections = txt.split("#End")
    part_after = sections[1].strip()

    cols = ["flag", "year", "month", "day", "time"]

    for energy in ["0.7-1.4", "1.4-2.8", "2.8-4.0"]:
        cols.extend([f"e_{energy}_flux", f"e_{energy}_uncertainty"])

    proton_ranges = [
        "13.6-15.1", "14.9-17.1", "17.0-19.3", "20.8-23.8", "23.8-26.4", "26.3-29.7",
        "29.5-33.4", "33.4-35.8", "35.5-40.5", "40.0-60.0", "60.0-100.0",
    ]
    for energy in proton_ranges:
        cols.extend([f"H_{energy}_flux", f"H_{energy}_uncertainty"])

    df = pd.read_csv(StringIO(part_after), delim_whitespace=True, names=cols)
    df["month"] = df["month"].replace("May", "5")

    df["time"] = df["time"].astype(str).str.zfill(4)
    df["hour"] = df["time"].str[:2]
    df["minute"] = df["time"].str[2:]

    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]])

    mask = (df.datetime.dt.day >= 10) & (df.datetime.dt.day <= 15)
    df = df[mask]

    flux_columns = [col for col in df.columns if "flux" in col]
    df = process_timeseries(
        df, instrument="STEREO", cols_to_use=flux_columns, invalid_markers=-999,
        sampling_time=5,
    )

    df["datetime"] = df.index

    df.to_csv(output_file, index=False)
    logger.info(f"STEREO HET data saved to {output_file}")
    return df


def read_soho_erne(date):
    """Reads SOHO ERNE proton data for a given date and returns a DataFrame."""
    filename = FLUX_DIR / "ERNE_MAY" / f"soho_erne-hed_l2-1min_{date}_v01.cdf"

    try:
        with pycdf.CDF(str(filename)) as cdf:
            df = pd.DataFrame(index=cdf["Epoch"][...])
            proton_energies, proton_delta = cdf["P_energy"][...], cdf["P_energy_delta"][...]

            for i, (energy, delta) in enumerate(zip(proton_energies, proton_delta)):
                df[f"H_{energy - delta/2:.1f}-{energy + delta/2:.1f}MeV"] = cdf["PH"][..., i]

        return df
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return None


def read_soho(start_date="20240510", end_date="20240515", save_csv=True):
    """Reads and combines SOHO ERNE data for a date range."""
    dfs = [read_soho_erne(f"202405{day:02d}") for day in range(10, 16)]
    df = pd.concat([df for df in dfs if df is not None])

    df = process_timeseries(
        df, instrument="SOHO ERNE", invalid_markers=[np.inf, -np.inf], sampling_time=5
    )

    df["datetime"] = df.index

    if save_csv:
        output_path = FLUX_DIR / "soho_erne.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"SOHO ERNE data saved to {output_path}")

    return df


def process_timeseries(
    df, instrument="EPAM", time_col="datetime", invalid_markers=None, cols_to_use=None,
    start_date="2024-05-10", end_date="2024-05-15", window_size=None, poly_order=2,
    sampling_time=5, apply_savgol=True,
):
    """
    Process time series data by applying Savitzky-Golay filtering at 1-minute resolution,
    then resampling to a final desired interval (e.g., 5 minutes).

    Parameters:
    df: DataFrame
    time_col: str, column name containing datetime
    invalid_markers: list/value to be replaced with NaN
    cols_to_use: list of columns to process (if None, uses all columns)
    start_date: str, start date for filtering
    end_date: str, end date for filtering
    window_size: int or None, window size for Savitzky-Golay filter (auto if None, must be odd)
    poly_order: int, polynomial order for Savitzky-Golay filter
    final_sampling_time: int, final resampling frequency in minutes (default 5)

    Returns:
    DataFrame with smoothed and resampled sensor data.
    """
    logger.info(f"Formating and cleaning {instrument} time series data... ")

    df_subset = df[cols_to_use] if cols_to_use else df.copy()

    df_clean = df_subset.replace(invalid_markers, np.nan) if invalid_markers else df_subset

    if time_col in df.columns:
        df_clean = df_clean.set_index(pd.to_datetime(df[time_col]))

    if apply_savgol:
        df_clean = df_clean.loc[start_date:end_date]

        df_1min = df_clean.resample("1T").mean()

        if window_size is None:
            window_size = max(5, 11)
        if window_size % 2 == 0:
            window_size += 1

        if poly_order >= window_size:
            poly_order = window_size - 1

        df_savgol = df_1min.apply(
            lambda x: (
                savgol_filter(
                    x.fillna(method="ffill").fillna(method="bfill"),
                    window_size,
                    poly_order,
                )
                if x.notna().sum() >= window_size
                else x
            ),
            axis=0,
        )

        df_savgol = pd.DataFrame(df_savgol, index=df_1min.index, columns=df_1min.columns)

        df_resampled = df_savgol.resample(f"{sampling_time}T").mean()

    else:
        if instrument.upper() == "ACE EPAM":
            df_clean.index = df_clean.index.floor("5T")
        df_resampled = df_clean.copy()

    return df_resampled.fillna(method="ffill").fillna(method="bfill")


if __name__ == "__main__":
    logger.info("Starting data processing...")

    epam_df = read_epam()

    url = "https://izw1.caltech.edu/STEREO/DATA/HET/Ahead/1minute/AeH24May.1m"
    stereo_df = get_stereo_het_data(url, output_file=FLUX_DIR / "stereoa_het.csv")

    soho_df = read_soho()

    logger.info("All data processing completed successfully.")
# %%
