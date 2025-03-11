import os
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass, field
from typing import Dict
from config import get_logger


# -------------------------------------------------------------------
# NAICS Industries Mapping as a Dataclass
# -------------------------------------------------------------------
@dataclass
class NAICSMapping:
    industries: Dict[str, str] = field(
        default_factory=lambda: {
            "11": "Agriculture, forestry, and fishing",
            "21": "Mining, quarrying, and gas extraction",
            "22": "Utilities",
            "23": "Construction",
            "31": "Manufacturing",
            "42": "Wholesale Trade",
            "44": "Retail Trade",
            "48": "Transportation and Warehousing",
            "51": "Information",
            "FIRE": "Finance, insurance, and real estate",
            "PROF": "Professional and business services",
            "6": "Educational, health, and social assistance",
            "7": "Arts, entertainment and food",
            "81": "Other (except public admin)",
            "G": "Government",
        }
    )


# Create a sorting key based on NAICSMapping industries
sort_key = {code: i for i, code in enumerate(NAICSMapping().industries.keys())}


# -------------------------------------------------------------------
# InputOutputModel Class with Economic Shock Methods
# -------------------------------------------------------------------
class InputOutputModel:
    def __init__(self, data_path, value_added=None, final_demand=None):
        self.final_demand = final_demand
        self.value_added = value_added
        self.data_path = data_path
        self.logger = get_logger(__name__, log_file="economic_model.log")
        self.load_input_data()
        self.calculate_B_matrix()

    def load_input_data(self):
        """
        Load the input data.
        """
        # Technical coefficients matrix
        self.technology_matrix = pd.read_csv(
            os.path.join(self.data_path, "direct_requirements.csv")
        )

        # A matrix (trim the value added in the I-O matrix)
        self.A = self.technology_matrix.iloc[:15].to_numpy()

        # Identity matrix
        self.I = np.eye(self.A.shape[1])

        # Real 2022 US gross output by industries (in billion dollars)
        self.industry_output = (
            pd.read_csv(
                os.path.join(self.data_path, "us_gross_output.csv"), usecols=["2022Q3"]
            )
            .to_numpy()
            .reshape(-1)
        )

    def calculate_B_matrix(self):
        """
        Calculate the B matrix:
          B = x_hat^-1 * A * x_hat, where x_hat is the diagonalized industry output.
        """
        gross_diag = np.diag(self.industry_output)
        gross_diag_inv = np.linalg.inv(gross_diag)
        self.B = np.dot(np.dot(gross_diag_inv, self.A), gross_diag)

    def leontief(self):
        """
        Calculate the Leontief model output:
            L = (I - A)^-1;  X = L * final_demand.
        Returns:
            numpy.ndarray: Total output vector.
        """
        if self.final_demand.any():
            return np.linalg.inv(self.I - self.A).dot(self.final_demand)

    def ghosh(self):
        """
        Calculate the Ghosh model impacts:
            Ghosh = (I - B)^-1;  delta_va = value_added^T @ Ghosh.
        Returns:
            numpy.ndarray: Downstream impacts.
        """
        if self.value_added.any():
            self.ghosh_model = np.linalg.inv(self.I - self.B)
            va_transposed = np.transpose(self.value_added)
            delta_va = va_transposed @ self.ghosh_model
            return delta_va

    # -------------------------------------------------------------------
    # Economic Shock Functions Integrated as Methods
    # -------------------------------------------------------------------
    @staticmethod
    def sum_economic_shocks(economic_shocks_list):
        """
        Convert a list of dictionaries to a DataFrame and sum entries per NAICS code if needed.
        """
        df = pd.DataFrame(economic_shocks_list)
        if df.duplicated("NAICS").any():
            df = df.groupby("NAICS").sum().reset_index()
        return df

    @staticmethod
    def pad_missing_naics(df, naics_dict):
        """
        Add missing NAICS codes with zero values.
        """
        missing_naics = set(naics_dict.keys()) - set(df.index)
        for naics in missing_naics:
            df.loc[naics] = 0
        return df

    def get_value_added(self, economic_shocks_list):
        """
        Process economic shocks, calculate impacts using the I-O model, and return the final value-added DataFrame.
        """
        total_econ_shock = self.sum_economic_shocks(economic_shocks_list)
        va = total_econ_shock.copy()
        va.set_index("NAICS", inplace=True)
        va = self.pad_missing_naics(va, NAICSMapping().industries)
        va = va.assign(EST=va.EST.fillna(0), GDP2022=va.GDP2022.fillna(0))
        va = va.sort_index(key=lambda x: x.map(sort_key.get))
        va["NAICS"] = va.index
        va["NAICSIndustries"] = va["NAICS"].map(NAICSMapping().industries)
        io_model = InputOutputModel(
            data_path=self.data_path, value_added=va["GDP2022"].to_numpy()
        )
        delta_va = io_model.ghosh()
        va["total_shocks"] = delta_va
        va["indirect_shocks"] = va["total_shocks"] - va["GDP2022"]
        va = va[["NAICSIndustries", "GDP2022", "indirect_shocks"]]
        va = va.rename(
            columns={"GDP2022": "Direct Impact", "indirect_shocks": "Indirect Impact"}
        )
        va["Total Impact"] = va["Direct Impact"] + va["Indirect Impact"]
        return va

    @staticmethod
    def analyze_impact(stats_df, n_failed_cells, n_samples=100):
        """
        Analyze economic impact given a stats DataFrame.
        """
        gdp_cols = [col for col in stats_df.columns if col.startswith("gdp_")]
        filled_df = stats_df.fillna(
            {"satellite_users": 0, "population": 0, "total_est": 0}
        )
        for col in gdp_cols:
            filled_df[col] = filled_df[col].fillna(0)
        random_indices = np.random.randint(
            0, len(filled_df), size=(n_samples, n_failed_cells)
        )
        failed_cells_matrix = filled_df.iloc[random_indices.flatten()].values.reshape(
            n_samples, n_failed_cells, -1
        )
        sums = failed_cells_matrix.sum(axis=1)
        col_positions = {
            "satellite_users": filled_df.columns.get_loc("satellite_users"),
            "population": filled_df.columns.get_loc("population"),
            "total_est": filled_df.columns.get_loc("total_est"),
        }
        total_affected_users = sums[:, col_positions["satellite_users"]]
        total_population = sums[:, col_positions["population"]]
        impact_proportion = np.divide(
            total_affected_users,
            total_population,
            out=np.zeros_like(total_population),
            where=total_population > 0,
        )
        results = {
            "total_pop_affected": np.median(total_population),
            "total_est_affected": np.median(sums[:, col_positions["total_est"]]),
            "satellite_users_affected": np.median(total_affected_users),
        }
        for col in gdp_cols:
            col_idx = filled_df.columns.get_loc(col)
            gdp_impact = sums[:, col_idx] * impact_proportion
            total_naics_gdp = filled_df[col].sum()
            naics_code = col.split("_")[1]
            results[f"gdp_{naics_code}_affected"] = np.median(gdp_impact)
            results[f"gdp_{naics_code}_percentage"] = (
                (np.median(gdp_impact) / total_naics_gdp * 100)
                if total_naics_gdp > 0
                else 0
            )
        return pd.Series(results)

    @staticmethod
    def prepare_shocks(impact_data):
        """
        Prepare economic shock data by NAICS code.
        """
        economic_shocks = []
        for naics_code in NAICSMapping().industries.keys():
            impact_col = f"gdp_{naics_code}_affected"
            if impact_col in impact_data.index:
                shock = {
                    "NAICS": naics_code,
                    "GDP2022": float(impact_data[impact_col]),
                    "EST": float(impact_data["total_est_affected"]),
                }
            else:
                shock = {"NAICS": naics_code, "GDP2022": 0.0, "EST": 0.0}
            economic_shocks.append(shock)
        return economic_shocks
