import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Custom imports
from viz import custom_line_plots

# Configs
from config import get_logger, FLUX_DIR, FIGURE_DIR

logger = get_logger(__name__, log_file="download_tles.log")


@dataclass
class SOHO_STEREO_Band:
    """Represents energy bands for SOHO and STEREO instruments."""

    soho: List[str]
    stereo: List[str]


@dataclass
class EPAM_Band:
    """Represents energy ranges for EPAM channels."""

    range: str


@dataclass
class EnergyData:
    """Unified energy bands and EPAM energies database."""

    soho_stereo_bands: Dict[str, SOHO_STEREO_Band] = field(
        default_factory=lambda: {
            "13-15": SOHO_STEREO_Band(
                soho=["H_13.8-15.2MeV"], stereo=["H_13.6-15.1_flux"]
            ),
            "17-19": SOHO_STEREO_Band(
                soho=["H_17.0-19.0MeV"], stereo=["H_17.0-19.3_flux"]
            ),
            "21-24": SOHO_STEREO_Band(
                soho=["H_21.2-23.8MeV"], stereo=["H_20.8-23.8_flux"]
            ),
            "26-30": SOHO_STEREO_Band(
                soho=["H_26.8-30.2MeV"], stereo=["H_26.3-29.7_flux"]
            ),
            "33-38": SOHO_STEREO_Band(
                soho=["H_34.0-38.0MeV"], stereo=["H_33.4-35.8_flux", "H_35.5-40.5_flux"]
            ),
            "40-60": SOHO_STEREO_Band(
                soho=["H_42.5-47.5MeV", "H_53.5-60.5MeV"], stereo=["H_40.0-60.0_flux"]
            ),
            "60-100": SOHO_STEREO_Band(
                soho=["H_68.0-76.0MeV", "H_85.0-95.0MeV"], stereo=["H_60.0-100.0_flux"]
            ),
        }
    )

    epam_energies: Dict[str, EPAM_Band] = field(
        default_factory=lambda: {
            "P1p": EPAM_Band(range="0.047-0.068"),
            "P2p": EPAM_Band(range="0.068-0.115"),
            "P3p": EPAM_Band(range="0.115-0.195"),
            "P4p": EPAM_Band(range="0.195-0.321"),
            "P5p": EPAM_Band(range="0.321-0.580"),
            "P6p": EPAM_Band(range="0.587-1.06"),
            "P7p": EPAM_Band(range="1.06-1.90"),
            "P8p": EPAM_Band(range="1.90-4.80"),
        }
    )

    def get_soho_stereo_band(self, energy_range: str) -> Optional[SOHO_STEREO_Band]:
        """Fetch SOHO-STEREO energy band safely."""
        return self.soho_stereo_bands.get(energy_range)

    def get_epam_energy_range(self, channel: str) -> Optional[str]:
        """Fetch energy range for a given EPAM channel."""
        return (
            self.epam_energies.get(channel).range
            if channel in self.epam_energies
            else None
        )


def load_and_process_data() -> (
    Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]
):
    """
    Load, reconcile SOHO & STEREO energy bands, rename EPAM columns using EnergyData, and return processed DataFrames.

    Returns:
        tuple: (df_soho_reconciled, df_stereo_reconciled, df_epam)
    """
    try:
        # Load all datasets
        df_soho = pd.read_csv(FLUX_DIR / "soho_erne.csv", parse_dates=True)
        df_stereo = pd.read_csv(FLUX_DIR / "stereoa_het.csv", parse_dates=True)
        df_epam = pd.read_csv(FLUX_DIR / "epam.csv", parse_dates=True)

        energy_data = EnergyData()

        epam_cols = list(energy_data.epam_energies.keys())

        # Get only the renamed columns and datetime column
        df_epam = df_epam[["datetime"] + epam_cols]

        # Reconcile SOHO & STEREO bands
        df_soho_bands = pd.DataFrame(index=df_soho.index)
        df_stereo_bands = pd.DataFrame(index=df_stereo.index)

        for band_name, band in energy_data.soho_stereo_bands.items():
            if band.soho and set(band.soho).issubset(df_soho.columns):
                df_soho_bands[f"SOHO_{band_name}"] = df_soho[band.soho].sum(axis=1)
            if band.stereo and set(band.stereo).issubset(df_stereo.columns):
                df_stereo_bands[f"STEREO_{band_name}"] = df_stereo[band.stereo].sum(
                    axis=1
                )

        # Add back datetime columns
        df_soho_bands["datetime"] = df_soho["datetime"]
        df_stereo_bands["datetime"] = df_stereo["datetime"]

        return df_soho_bands, df_stereo_bands, df_epam

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None, None, None


if __name__ == "__main__":
    # Load the energy bands and EPAM energies
    df_soho, df_stereo, df_epam = load_and_process_data()
    if df_soho is None or df_stereo is None or df_epam is None:
        logger.error("One or more data files could not be loaded.")
    else:
        energy_data = EnergyData()  # Instantiate the unified energy mappings.
        # custom_line_plots(
        #     df_epam,
        #     df_soho,
        #     df_stereo,
        #     energy_data,
        #     logger,
        #     filename=FIGURE_DIR / "epam_soho_stereo.png",
        # )
