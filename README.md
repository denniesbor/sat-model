# Sensitivity Analysis of LEO Satellites to Space Weather Activities

## Overview

This project conducts a sensitivity analysis of LEO satellites under space weather events using three main modules. The radiation model estimates Single Event Effects (SEE) by fitting Weibull or triangular distributions to flux data from SOHO, ACE, and STEREO and computes dosimetry based on differential LET. It also implements a crude version of the CREME96 IRPP approach for radiation analysis. Magnetic rigidity and the impact of Earth's shadow are accounted for by scaling the differential fluxes with a simplified Stormer model. The network analysis evaluates satellite coverage, network capacity degradation, and the cascading economic impacts of satellite failures. The satellite fleet analysis assesses atmospheric drag and propellant usage using the MSIS model and SGP4 propagation. The figure below illustrates the methodological framework of the project.  
![Methodology Figure](viz/figures/box.png)

## Data Requirements

Historical TLE data, for example from the May 2024 Gannion storm, is obtained via the SpaceTrack API. Radiation data is sourced from ACE EPAM, STEREO A, and SOHO ERNE. Additionally, socioeconomic data from the Census Bureau at the ZCTA level and the 2022 Q3 BEA direct requirement matrix are used for economic analysis.

## Setup

First, create a `.env` file in the parent directory and add your SpaceTrack API credentials:

```env
SPACETRACK_USERNAME=<your_username>
SPACETRACK_PASSWORD=<your_password>
```

Next, edit the `config/settings.py` file to specify your desired constellations. The default configuration includes ONEWEB, KUIPER, TELESAT, and STARLINK.

## Execution Steps

### 1. Satellite Fleet and Drag Model

This step requires the `pymsis` library. First, download the TLE data by executing:

```bash
python satellite_fleet/download_tles.py
```

Then, propagate satellite positions using SGP4 at 5-minute intervals and compute drag estimates by running:

```bash
python satellite_fleet/drag_estimates.py
```

### 2. Radiation Model Processing

Prepare the SEE data and merge the flux datasets by executing:

```bash
python radiation_model/process_see_data.py
python radiation_model/see_rate.py
```

### 3. Network Analysis

The network analysis module extracts Uber H3 cells for the contiguous United States to support the capacity analysis model.

### 4. Capacity Degradation and Economic Analysis

The capacity degradation model estimates the reduction in network capacity based on failed satellites identified by the radiation model. It determines the Uber H3 cells served by the satellites and intersects the affected cells with socioeconomic data from the Census Bureau at the ZCTA level. An input-output model is then applied using the 2022 Q3 outputs and the BEA direct requirement matrix to derive the macroeconomic effects.

## Directory Structure

The project directory is organized as follows:

```
.
├── logs
├── viz
│   ├── figures
│   │   ├── epam_soho_stereo.png
│   │   ├── propellant_by_constellation.png
│   │   └── satellite_visibility.png
│   ├── __init__.py
│   ├── viz_econo.py
│   └── viz.py
├── config
│   ├── __init__.py
│   └── settings.py
├── network_analysis
│   ├── h3_grid.py
│   └── __init__.py
├── satellite_fleet
│   ├── download_tles.py
│   └── drag_estimates.py
├── capacity_analysis
│   ├── capacity_model.py
│   ├── coverage_estimator.py
│   ├── economic_impact.py
│   ├── __Init__.py
│   ├── io_model.py
│   └── monte_carlo_run.py
├── radiation_model
│   ├── de421.bsp
│   ├── let.py
│   └── process_see_data.py
└── data
```

## Additional Information

Drag estimates are calculated assuming Xenon fuel with an Isp of 4.4. The radiation model implements a variant of the CREME96 IRPP model, and the network analysis uses the Uber H3 algorithm for spatial discretization. For more information please review the individual scripts.
