import os
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Ignore warnings and load environment variables
warnings.simplefilter("ignore")
load_dotenv()

# Path to the root of the project
ROOT_DIR = Path(__file__).resolve().parents[1]

# Path to the data directory
DATA_DIR = ROOT_DIR / "data"

# Satellite data directory
SATELLITE_DIR = DATA_DIR / "Satellites"
SATELLITE_DIR.mkdir(parents=True, exist_ok=True)

# Flux data directory
FLUX_DIR = DATA_DIR / "ace_soho_stereo"

# Figures directory
FIGURE_DIR = ROOT_DIR / "viz" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Economic data directory
ECONOMIC_DIR = DATA_DIR / "economics"
ECONOMIC_DIR.mkdir(parents=True, exist_ok=True)

# Network data directory
NETWORK_DIR = DATA_DIR / "network"
NETWORK_DIR.mkdir(parents=True, exist_ok=True)

# Logs directory
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# CONSTELLATIONS (TO add more)
CONSTELLATIONS = ["ONEWEB", "KUIPER", "STARLINK", "TELESAT"]


def get_logger(name=__name__, log_file=None, level=logging.INFO):
    """
    Creates and returns a logger with the specified name and log level.

    Args:
        name (str): Logger name (usually `__name__`).
        log_file (str, optional): If provided, logs are saved to this file.
        level (int): Logging level (default: INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():  # Avoid adding multiple handlers
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        file_handler = logging.FileHandler(LOG_DIR / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def space_track_login():
    """
    Retrieves SpaceTrack credentials from .env file.

    Returns:
        tuple: (username, password)

    Raises:
        ValueError: If .env file is missing or credentials are not set.
    """
    if os.path.exists(".env"):
        username = os.getenv("SPACETRACK_USERNAME")
        password = os.getenv("SPACETRACK_PASSWORD")
        if not username or not password:
            raise ValueError("SPACETRACK credentials are missing in .env file.")
    else:
        raise ValueError(
            "No .env file found! Expected SPACETRACK_USERNAME and SPACETRACK_PASSWORD."
        )

    return username, password
