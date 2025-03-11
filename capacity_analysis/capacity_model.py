# %%
from config import ECONOMIC_DIR, NETWORK_DIR, get_logger
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
import dask_geopandas
import os
import numpy as np
import pickle
from tqdm import tqdm

from network_analysis.h3_grid import continental_hex_gdf, continental_us_states


@dataclass
class TECHNO_ECONOMICS:
    data_zcta_path: Path = ECONOMIC_DIR / "NAICS_EST_GDP2022_ZCTA.csv"
    pop_data_path: Path = ECONOMIC_DIR / "2020_decennial_census_at_ZCTA_level.csv"
    h3_stats_file: Path = ECONOMIC_DIR / "h3_stats.pkl"
    stats_df_path: Path = ECONOMIC_DIR / "stats_df.csv"
    logger: object = field(
        default_factory=lambda: get_logger(__name__, log_file="economic_model.log")
    )

    def read_data(self):
        """Loads and processes ZCTA and population data."""
        self.logger.info("Loading ZCTA data from: %s", self.data_zcta_path)
        data_zcta = pd.read_csv(self.data_zcta_path)
        data_zcta["ZCTA"] = data_zcta["ZCTA"].astype(int)

        self.logger.info("Loading population data from: %s", self.pop_data_path)
        population_df = pd.read_csv(self.pop_data_path)
        population_df["ZCTA"] = population_df["ZCTA"].astype(int)

        # Merge with unique ZCTA regions
        unique_zcta_regions = data_zcta[["ZCTA", "REGIONS", "STABBR"]].drop_duplicates()
        regions_pop_df = unique_zcta_regions.merge(population_df, on=["ZCTA", "STABBR"])

        return data_zcta, regions_pop_df

    def process_zcta_business_data(self, data_zcta):
        """Processes ZCTA business data and computes daily GDP values."""
        self.logger.info("Processing ZCTA business data...")
        zcta_business_gdf = self.optimized_zcta_processing(data_zcta)
        zcta_business_gdf["GDP2022"] = (zcta_business_gdf["GDP2022"] * 1000) / 365
        return gpd.GeoDataFrame(zcta_business_gdf, geometry="geometry")

    @staticmethod
    def optimized_zcta_processing(data_zcta):
        """
        Process ZCTA (ZIP Code Tabulation Areas) data using Dask for large datasets.

        Args:
            data_zcta: DataFrame containing ZCTA business and economic data

        Returns:
            GeoDataFrame: Processed ZCTA data with geometries and business info
        """
        zcta_gdf = dask_geopandas.read_file(
            ECONOMIC_DIR / "tl_2020_us_zcta520.zip", npartitions=10
        )

        # Ensure CRS is WGS84
        if zcta_gdf.crs != "EPSG:4326":
            zcta_gdf = zcta_gdf.to_crs("EPSG:4326")

        # Standardize column names and data types
        zcta_gdf = zcta_gdf.rename(columns={"ZCTA5CE20": "ZCTA"})
        zcta_gdf["ZCTA"] = zcta_gdf["ZCTA"].astype(int)

        # Convert input to dask DataFrame if needed
        if not isinstance(data_zcta, dd.DataFrame):
            data_zcta = dd.from_pandas(data_zcta, npartitions=10)
            data_zcta["ZCTA"] = data_zcta["ZCTA"].astype(int)

        # Merge business data with geometries
        zcta_business_gdf = data_zcta.merge(
            zcta_gdf[["ZCTA", "geometry"]], on="ZCTA", how="inner"
        )

        return zcta_business_gdf.compute()

    def process_full_dataset(
        self, regions_pop_df, zcta_business_gdf, continental_hex_gdf
    ):
        """
        Intersect socio-economic data with H3 cells and calculate population and business statistics.
        Assign populations and businesses to each cell based on area proportion.
        """
        if self.h3_stats_file.exists():
            self.logger.info("Loading existing H3 stats...")
            with open(self.h3_stats_file, "rb") as f:
                return pickle.load(f)

        h3_stats = {
            hex_id: {"pop": 0, "naics_stats": {}}
            for hex_id in continental_hex_gdf.index
        }

        for zcta_id in tqdm(regions_pop_df.ZCTA.unique(), desc="Processing ZCTAs"):
            zcta_businesses = zcta_business_gdf[zcta_business_gdf.ZCTA == zcta_id]
            if zcta_businesses.empty:
                continue

            zcta_geom = zcta_businesses.iloc[0].geometry
            zcta_area = zcta_geom.area

            intersecting_cells = continental_hex_gdf[
                continental_hex_gdf.geometry.intersects(zcta_geom)
            ]
            if intersecting_cells.empty:
                continue

            intersection_areas = {
                cell_id: cell.geometry.intersection(zcta_geom).area / zcta_area
                for cell_id, cell in intersecting_cells.iterrows()
            }

            zcta_pop = regions_pop_df.loc[regions_pop_df.ZCTA == zcta_id, "POP20"].iloc[
                0
            ]

            for cell_id, area_proportion in intersection_areas.items():
                h3_stats[cell_id]["pop"] += zcta_pop * area_proportion

                for _, business in zcta_businesses.iterrows():
                    naics = business.NAICS
                    if naics not in h3_stats[cell_id]["naics_stats"]:
                        h3_stats[cell_id]["naics_stats"][naics] = {"est": 0, "gdp": 0}

                    h3_stats[cell_id]["naics_stats"][naics]["est"] += (
                        business.EST * area_proportion
                    )
                    h3_stats[cell_id]["naics_stats"][naics]["gdp"] += (
                        business.GDP2022 * area_proportion
                    )

        self.logger.info("Saving H3 statistics to %s", self.h3_stats_file)
        with open(self.h3_stats_file, "wb") as f:
            pickle.dump(h3_stats, f)

        return h3_stats

    @staticmethod
    def distribute_satellite_users(h3_stats, total_users=2_500_000, beta=1e-5):
        weights = {
            cell: np.exp(-beta * stats["pop"]) if stats["pop"] > 0 else 0
            for cell, stats in h3_stats.items()
        }
        total_weight = sum(weights.values())
        normalized_weights = {
            cell: weight / total_weight for cell, weight in weights.items()
        }
        users_per_cell = {
            cell: int(normalized_weights[cell] * total_users) for cell in h3_stats
        }
        total_assigned = sum(users_per_cell.values())
        remainder = total_users - total_assigned

        fractional_parts = {
            cell: (normalized_weights[cell] * total_users) % 1 for cell in h3_stats
        }
        sorted_cells = sorted(fractional_parts, key=fractional_parts.get, reverse=True)
        for cell in sorted_cells[:remainder]:
            users_per_cell[cell] += 1

        assert sum(users_per_cell.values()) == total_users
        return users_per_cell

    def generate_stats_dataframe(self, h3_stats, users_per_cell):
        """
        Convert H3 statistics to a DataFrame and merge LEO sat user distribution.
        """
        if self.stats_df_path.exists():
            self.logger.info(
                "Loading existing stats DataFrame from %s", self.stats_df_path
            )
            return pd.read_csv(self.stats_df_path)

        self.logger.info("Generating new stats DataFrame...")

        cell_stats = []
        for cell_id, stats in h3_stats.items():
            cell_data = {
                "cell_id": cell_id,
                "population": stats["pop"],
                "total_est": sum(
                    naics_stat["est"] for naics_stat in stats["naics_stats"].values()
                ),
                "total_gdp": sum(
                    naics_stat["gdp"] for naics_stat in stats["naics_stats"].values()
                ),
            }

            for naics, naics_stats in stats["naics_stats"].items():
                cell_data[f"est_{naics}"] = naics_stats["est"]
                cell_data[f"gdp_{naics}"] = naics_stats["gdp"]

            cell_stats.append(cell_data)

        stats_df = pd.DataFrame(cell_stats)
        stats_df["satellite_users"] = stats_df["cell_id"].map(users_per_cell)

        stats_df.to_csv(self.stats_df_path, index=False)
        self.logger.info("Stats DataFrame saved to %s", self.stats_df_path)

        return stats_df

    def process_economic_model(
        self, regions_pop_df, zcta_business_gdf, continental_hex_gdf
    ):
        """
        End-to-end function to process full economic model: H3 stats, satellite distribution, and DataFrame.
        """
        self.logger.info("Processing full economic model...")

        h3_stats = self.process_full_dataset(
            regions_pop_df, zcta_business_gdf, continental_hex_gdf
        )
        users_per_cell = self.distribute_satellite_users(h3_stats)
        stats_df = self.generate_stats_dataframe(h3_stats, users_per_cell)

        stats_df["satellite_users"] = stats_df["cell_id"].map(users_per_cell)

        # Add stats back to H3 GeoDataFrame
        continental_hex_gdf = continental_hex_gdf.merge(
            stats_df, left_index=True, right_on="cell_id"
        )

        return stats_df, continental_hex_gdf


# %%
economic_model = TECHNO_ECONOMICS()
data_zcta, regions_pop_df = economic_model.read_data()
zcta_business_gdf = economic_model.process_zcta_business_data(data_zcta)
stats_df, continental_hex_gdf = economic_model.process_economic_model(
    regions_pop_df, zcta_business_gdf, continental_hex_gdf
)
# %%
