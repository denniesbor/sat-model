# %%
import numpy as np
import pandas as pd
from see_rate import load_and_process_data


# %%
# -----------------------------------------------------------
# 1) Convert differential energy spectrum to differential LET spectrum
# -----------------------------------------------------------
def energy_to_let_spectrum_combined(df_stereo, df_epam):
    """
    Convert differential energy spectrum (STEREO, EPAM) to differential LET (dF/dL).
    Uses geometric-mean energies for each band and a power-law relation between E and LET.
    """
    k = 95  # for Silicon in MeV·cm²/g
    let_df = pd.DataFrame(index=df_stereo.index)

    # Define energy bands and their representative geometric-mean energies
    energy_bands = {
        # EPAM bands
        "P1p": np.sqrt(0.047 * 0.068),
        "P2p": np.sqrt(0.068 * 0.115),
        "P3p": np.sqrt(0.115 * 0.195),
        "P4p": np.sqrt(0.195 * 0.321),
        "P5p": np.sqrt(0.321 * 0.580),
        "P6p": np.sqrt(0.587 * 1.06),
        "P7p": np.sqrt(1.06 * 1.90),
        "P8p": np.sqrt(1.90 * 4.80),
        # STEREO bands
        "STEREO_13-15": np.sqrt(13 * 15),
        "STEREO_17-19": np.sqrt(17 * 19),
        "STEREO_21-24": np.sqrt(21 * 24),
        "STEREO_26-30": np.sqrt(26 * 30),
        "STEREO_33-38": np.sqrt(33 * 38),
        "STEREO_40-60": np.sqrt(40 * 60),
        "STEREO_60-100": np.sqrt(60 * 100),
    }

    # Process EPAM bands
    for band in ["P1p", "P2p", "P3p", "P4p", "P5p", "P6p", "P7p", "P8p"]:
        E = energy_bands[band]
        # LET = k / (E^0.7)
        S = k / (E**0.7)
        # dE/dS = -(E^(1.7)) / (0.7 * k)  -> from dS = -k * 0.7 * E^(-0.7) ...
        dE_dS = -(E**1.7) / (0.7 * k)
        let_df[f"LET_{S:.2f}"] = df_epam[band] * abs(dE_dS)

    # Process STEREO bands
    for band in [
        "STEREO_13-15",
        "STEREO_17-19",
        "STEREO_21-24",
        "STEREO_26-30",
        "STEREO_33-38",
        "STEREO_40-60",
        "STEREO_60-100",
    ]:
        E = energy_bands[band]
        S = k / (E**0.7)
        dE_dS = -(E**1.7) / (0.7 * k)
        let_df[f"LET_{S:.2f}"] = df_stereo[band] * abs(dE_dS)

    # Sort columns by their LET value (descending order of S)
    sorted_cols = sorted(
        let_df.columns, key=lambda x: float(x.split("_")[1]), reverse=True
    )
    let_df = let_df[sorted_cols]

    return let_df


# -----------------------------------------------------------
# 2) Convert differential LET spectrum to integral LET spectrum
# -----------------------------------------------------------
def compute_integral_let_spectrum(let_df):
    """
    Convert differential LET spectrum (dF/dL) into integral LET spectrum F(L).
    Sort columns in ascending order of LET, then do a cumulative sum from
    the highest-LET column backward, so each column is F(>= L).
    """
    # Sort ascending by numeric part (e.g., 'LET_12.30' -> 12.30)
    sorted_cols = sorted(let_df.columns, key=lambda c: float(c.split("_")[1]))
    let_df = let_df[sorted_cols]

    # Integrate from high LET to low LET via cumulative sum
    cum_flux = let_df[sorted_cols[::-1]].cumsum(axis=1)
    integral_let_df = cum_flux[sorted_cols[::-1]]
    return integral_let_df


# -----------------------------------------------------------
# 3) Functions for RPP geometry, path distribution, etc.
# -----------------------------------------------------------
def calculate_L_min(X_e_ratio, Qc, p_max):
    """
    Minimum LET threshold for upset (MeV·cm²/g).
    X_e_ratio: eV per charge pair (e.g., 3.6 eV per e-h pair for Si, but stored as eV/C),
    Qc: critical charge in pC,
    p_max: max path length in g·cm⁻².
    """
    Qc_coulomb = Qc * 1e-12
    L_min_eV = (X_e_ratio * Qc_coulomb) / p_max  # eV·cm²/g
    return L_min_eV * 1e-6  # convert eV -> MeV


def calculate_p_L(L, X_e_ratio, Qc):
    """
    Compute path length p corresponding to a given L (MeV·cm²/g).
    L is in MeV·cm²/g -> convert to eV·cm²/g, then p = (X_e_ratio*Qc)/(energy).
    """
    Qc_coulomb = Qc * 1e-12
    L_eV = L * 1e6
    return (X_e_ratio * Qc_coulomb) / L_eV  # g·cm⁻²


def triangular_distribution(p, p_max):
    """
    Triangular distribution from 0 to p_max, peak at p_max/2, area=1 in [0,p_max].
    Returns distribution value at path length p (g·cm⁻²).
    """
    if p < 0 or p > p_max:
        return 0.0
    p_peak = p_max / 2
    height = 2.0 / p_max
    if p <= p_peak:
        return (height / p_peak) * p
    else:
        return height - (height / p_peak) * (p - p_peak)


def weibull_distribution(p, p_max, shape=1.5):
    """
    Truncated Weibull distribution from p=0 to p=p_max with shape=α and scale=λ.
    Ensures total area is 1 over [0, p_max].

    Parameters
    ----------
    p : float
        Current path length, g·cm⁻²
    p_max : float
        Maximum path length, g·cm⁻²
    shape : float
        Weibull shape parameter (α)
    scale : float
        Weibull scale parameter (λ)

    Returns
    -------
    float
        Probability density at p under truncated Weibull(α, λ) on [0, p_max].
    """

    scale = p_max / (np.pi / 2)  # Scale to ensure CDF at p_max is 1
    # Outside [0, p_max], density is zero
    if p < 0 or p > p_max:
        return 0.0

    # Standard Weibull PDF: f(p) = (α/λ) * (p/λ)^(α-1) * exp(-(p/λ)^α)
    # We truncate at p_max, dividing by F(p_max), the CDF at p_max:
    # F(p_max) = 1 - exp(-(p_max/λ)^α)
    cdf_pmax = 1 - np.exp(-((p_max / scale) ** shape))
    if cdf_pmax == 0:
        return 0.0  # Avoid division by zero if scale or shape are not valid

    pdf_full = (
        (shape / scale) * ((p / scale) ** (shape - 1)) * np.exp(-((p / scale) ** shape))
    )
    return pdf_full / cdf_pmax


# %%
# -----------------------------------------------------------
# 4) Main upset-rate computation (Adams-like method)
# -----------------------------------------------------------
def calculate_upset_rate_time_series(let_df, params):
    """
    Calculate upset rate vs time using an RPP model, triangular p(L) distribution, and
    an integral approach (Adams 1983 style). Each row in let_df is a set of fluxes vs. LET.
    params must define geometry, density, Qc, base failure rate, etc.
    """
    results = pd.DataFrame(index=let_df.index, columns=["Upset Rate", "Failure Rate"])

    length = params["length"]  # μm
    width = params["width"]  # μm
    height = params["height"]  # μm
    density = params["density"]  # g/cm³
    X_e_ratio = params["X_e_ratio"]  # eV/C
    Qc = params["Qc"]  # pC
    beta = params["beta"]  # shielding fraction
    lambda_base = params["lambda_base"]
    n_components = params["n_components"]

    # Convert diagonal from μm to cm
    diagonal_cm = np.sqrt(length**2 + width**2 + height**2) * 1e-4
    p_max = diagonal_cm * density

    # Minimum LET needed to deposit Qc
    L_min = calculate_L_min(X_e_ratio, Qc, p_max)
    lambda_eff = lambda_base * (1 - beta)

    # For each time step, sum upset contributions from each LET bin above L_min
    for t_idx, row in let_df.iterrows():
        let_values = [float(c.split("_")[1]) for c in let_df.columns]
        flux_values = row.values

        # Consider only bins with L >= L_min
        valid_idxs = [i for i, L in enumerate(let_values) if L >= L_min]
        if not valid_idxs:
            U = 0.0
        else:
            U = 0.0
            for idx in valid_idxs:
                L = let_values[idx]
                F_L = flux_values[idx]
                p = calculate_p_L(L, X_e_ratio, Qc)
                # Dp = triangular_distribution(p, p_max)
                Dp = weibull_distribution(p, p_max)
                # Summation from Adams: U += [Dp * F(L) / L^2], then multiply constants
                U += Dp * F_L / (L * L)

            # Multiply by geometry & constants
            # A: area in cm² -> length*width in μm² times 1e-8
            A = length * width * 1e-8
            # Qc pC -> Coulomb => *1e-12 is already in calculate_p_L, but here we do for total
            U *= np.pi * A * X_e_ratio * (Qc * 1e-12)

        # System-level failure
        failure_rate = U * lambda_eff * n_components
        results.loc[t_idx] = [U, failure_rate]

    return results


def compute_dose(differential_let_df, dt_seconds):
    """
    Compute approximate dose (J/kg) for each time step.

    The dose is computed as:
        Dose = ∑ (Flux_i * LET_i * conversion_factor * dt)
    where:
        - Flux_i is the differential flux in the LET bin,
        - LET_i (in MeV·cm²/g) is taken from the column label,
        - conversion_factor converts MeV to Joules (1 MeV = 1.602e-13 J),
        - dt_seconds is the time interval over which the flux is integrated.

    Args:
        differential_let_df: DataFrame with columns labeled like 'LET_xx.xx'
                             and differential flux values.
        dt_seconds: Time interval in seconds.

    Returns:
        pandas.Series of dose (J/kg) with the same index as differential_let_df.
    """
    conversion_factor = 1.602e-13  # J/MeV
    # Extract LET values from column names
    let_values = np.array(
        [float(col.split("_")[1]) for col in differential_let_df.columns]
    )
    # Compute dose by summing (flux * LET) over all LET bins, multiplied by conversion and dt.
    dose_series = (
        differential_let_df.multiply(let_values, axis=1).sum(axis=1)
        * conversion_factor
        * dt_seconds
    )
    return dose_series


# %%
if __name__ == "__main__":

    df_soho, df_stereo, df_epam = load_and_process_data()
    datetime = df_soho.datetime

    # 1) Build differential LET (dF/dL)
    let_differential = energy_to_let_spectrum_combined(df_stereo, df_epam)

    # 2) Convert to integral LET (F(>=L))
    let_integral = compute_integral_let_spectrum(let_differential)

    # 3 Parameters
    params = {
        "length": 10,  # μm
        "width": 10,  # μm
        "height": 10,  # μm
        "density": 2.328,  # g/cm³ (Silicon)
        "X_e_ratio": 3.6 / 1.602e-19,  # eV/C, ~2.248e19
        "Qc": 0.1,  # pC
        "beta": 0.5,  # 50% shielding
        "lambda_base": 1 / 1000,  # base fail probability
        "n_components": 50,
    }
    
    # To account for umbra/penumbra, we can use a Weibull distributin
    # To also account for Stormer cutoff, we can use a Gaussian distribution

    # 4) Compute time-series upset/failure rates
    results_df = calculate_upset_rate_time_series(let_integral, params)
        
    dt_seconds = 5 * 60  # 5 minutes
    # Compute dose (J/kg) for each time step
    dose_series = compute_dose(let_differential, dt_seconds)
    print(results_df)

# %%
