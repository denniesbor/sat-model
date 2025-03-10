import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
import geopandas as gpd
import h3
from shapely.geometry import Polygon
from config import ECONOMIC_DIR, NETWORK_DIR, get_logger


@dataclass
class US_BOUNDARY:
    states_gdf_path: Path = ECONOMIC_DIR / "tl_2022_us_state.zip"
    h3_resolution: int = 5
    bbox: List[float] = field(
        default_factory=lambda: [-125.0, 24.0, -66.0, 50.0]
    )  # Continental US bounding box
    h3_grid_path: Path = NETWORK_DIR / f"h3_grid_res{5}.gpkg"
    conus_grid_path: Path = NETWORK_DIR / f"h3_grid_conus_res{5}.gpkg"

    continental_states: gpd.GeoDataFrame = field(init=False)
    logger: object = field(
        default_factory=lambda: get_logger(__name__, log_file="h3_grid.log")
    )

    def __post_init__(self):
        self.logger.info("Loading state boundaries from: %s", self.states_gdf_path)
        states_gdf = gpd.read_file(self.states_gdf_path)

        # Filter for continental US (excluding Alaska, Hawaii, Puerto Rico)
        self.continental_states = states_gdf[
            ~states_gdf["STATEFP"].isin(["02", "15", "72"])
        ]

        # Reproject to EPSG 4326
        self.continental_states.to_crs(epsg=4326, inplace=True)


def process_h3_grid(bbox, force_regenerate=False):
    """
    Generate or load H3 grid for continental US with specified resolution.

    Args:
        bbox (tuple): (min_lon, min_lat, max_lon, max_lat) bounding box
        force_regenerate (bool): Force regeneration of H3 grid (default: False)

    Returns:
        tuple: (GeoDataFrame of H3 hexagons intersecting continental US, US state boundaries)
    """
    logger = get_logger(__name__, log_file="h3_grid.log")
    us_boundary = US_BOUNDARY()

    h3_grid_path = us_boundary.h3_grid_path
    conus_grid_path = us_boundary.conus_grid_path

    def generate_hexagons_in_bbox(bbox, resolution):
        """Generate H3 hexagons within bounding box."""
        hexagons = set()
        min_lon, min_lat, max_lon, max_lat = bbox
        for lat in range(int(min_lat * 1e6), int(max_lat * 1e6), 10000):
            for lon in range(int(min_lon * 1e6), int(max_lon * 1e6), 10000):
                hex_id = h3.latlng_to_cell(lat / 1e6, lon / 1e6, resolution)
                hexagons.add(hex_id)
        return hexagons

    def generate_initial_grid():
        """Generate initial H3 grid."""
        logger.info("Generating initial H3 grid...")
        hex_ids = generate_hexagons_in_bbox(bbox, us_boundary.h3_resolution)
        hex_polygons = [
            Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h)])
            for h in hex_ids
        ]
        return gpd.GeoDataFrame({"geometry": hex_polygons}, crs="EPSG:4326")

    def get_conus_states():
        """Load and return continental US states."""
        logger.info("Loading CONUS states...")
        return us_boundary.continental_states

    # Load or regenerate CONUS grid
    if not force_regenerate and conus_grid_path.exists():
        try:
            logger.info("Loading existing CONUS H3 grid from %s", conus_grid_path)
            return gpd.read_file(conus_grid_path), get_conus_states()
        except Exception as e:
            logger.warning("Error loading existing CONUS grid: %s", e)
            logger.info("Proceeding to regenerate...")

    # Load or regenerate initial H3 grid
    if not force_regenerate and h3_grid_path.exists():
        try:
            logger.info("Loading existing H3 grid from %s", h3_grid_path)
            hex_gdf = gpd.read_file(h3_grid_path)
        except Exception as e:
            logger.warning("Error loading existing H3 grid: %s", e)
            hex_gdf = generate_initial_grid()
    else:
        hex_gdf = generate_initial_grid()

    # Save initial H3 grid if it doesn't exist
    if not h3_grid_path.exists():
        try:
            logger.info("Saving H3 grid to %s", h3_grid_path)
            hex_gdf.to_file(h3_grid_path, driver="GPKG")
        except Exception as e:
            logger.error("Error saving H3 grid: %s", e)

    # Get continental US states
    continental_us_states = get_conus_states()

    # Get the boundary of continental US
    us_boundary_geom = continental_us_states.geometry.unary_union

    # Perform spatial join to get H3 cells that intersect with continental US
    logger.info("Performing spatial join with CONUS boundary...")
    continental_hex_gdf = gpd.sjoin(
        hex_gdf,
        gpd.GeoDataFrame(geometry=[us_boundary_geom], crs="EPSG:4326"),
        predicate="intersects",
    )

    # Save CONUS grid
    try:
        logger.info("Saving CONUS H3 grid to %s", conus_grid_path)
        continental_hex_gdf.to_file(conus_grid_path, driver="GPKG")
    except Exception as e:
        logger.error("Error saving CONUS grid: %s", e)

    return continental_hex_gdf, continental_us_states


# **Use the function**
continental_hex_gdf, continental_us_states = process_h3_grid(
    bbox=US_BOUNDARY().bbox,
    force_regenerate=False,  # Set to True to force regeneration
)
