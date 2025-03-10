# Customer line plots scripts
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches

import pandas as pd


def custom_line_plots(
    df_epam,
    df_soho,
    df_stereo,
    energy_data,
    annotation=True,
    logger=None,
    filename=None,
):
    """
    Create custom line plots for EPAM, SOHO, and STEREO data using keys from energy_data.

    Parameters:
      df_epam    : DataFrame with EPAM data (datetime index) whose columns are the renamed energy ranges.
      df_soho    : DataFrame with reconciled SOHO data (datetime index) with columns like "SOHO_13-15", etc.
      df_stereo  : DataFrame with reconciled STEREO data (datetime index) with columns like "STEREO_13-15", etc.
      energy_data: An instance of EnergyData containing EPAM and SOHO/STEREO mappings.
      annotation : bool, if True, annotate the x-axis with common limits and formatted ticks.
    """
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Plot EPAM data using the energy_data EPAM keys.
    for key, epam_band in energy_data.epam_energies.items():
        col_name = epam_band.range  # df_epam columns are renamed to these values.
        if col_name in df_epam.columns:
            ax1.plot(
                pd.to_datetime(df_epam["datetime"]), df_epam[col_name], label=col_name
            )
        else:
            logger.warning(f"EPAM column {col_name} not found.")

    # Plot SOHO bands using energy_data.soho_stereo_bands keys.
    for band_key in energy_data.soho_stereo_bands.keys():
        col_name = f"SOHO_{band_key}"
        if col_name in df_soho.columns:
            ax2.plot(
                pd.to_datetime(df_soho["datetime"]), df_soho[col_name], label=band_key
            )
        else:
            logger.warning(f"SOHO column {col_name} not found.")

    # Plot STEREO bands.
    for band_key in energy_data.soho_stereo_bands.keys():
        col_name = f"STEREO_{band_key}"
        if col_name in df_stereo.columns:
            ax3.plot(
                pd.to_datetime(df_stereo["datetime"]),
                df_stereo[col_name],
                label=band_key,
            )
        else:
            logger.warning(f"STEREO column {col_name} not found.")

    # Configure common grid and x-axis formatting.
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, which="major", linestyle="-", alpha=0.5)
        ax.grid(True, which="minor", linestyle="--", alpha=0.2)
        ax.set_ylim(bottom=0)
        # ax.xaxis.set_major_locator(mdates.DayLocator())
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))

    ax1.set_title(
        "(a) ACE/EPAM Differential Flux Spectrum", fontweight="bold", loc="left"
    )
    ax1.set_ylabel("Flux (1/cm²-s-sr-MeV)")
    ax1.legend(
        title="EPAM Energy Channels",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        frameon=False,
    )

    ax2.set_title("(b) SOHO/ERNE Proton Fluxes", fontweight="bold", loc="left")
    ax2.set_ylabel("Flux (1/cm²-s-sr-MeV)")
    ax2.legend(
        title="SOHO Energy Bands",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        frameon=False,
    )

    ax3.set_title("(c) STEREO-A/HET Proton Fluxes", fontweight="bold", loc="left")
    ax3.set_ylabel("Flux (1/cm²-s-sr-MeV)")
    ax3.legend(
        title="STEREO Energy Bands",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        frameon=False,
    )

    if annotation:
        times = list(pd.to_datetime(df_epam.datetime))
        tick_positions = np.linspace(0, len(times) - 1, 10, dtype=int)
        tick_times = [times[i] for i in tick_positions]
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(times[0], times[-1])
            ax.set_xticks(tick_times)
            ax.set_xticklabels([t.strftime("%H:%M\n%m/%d") for t in tick_times])

    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_propellant_by_constellation(grouped_df, file_name=None):
    """
    Plots propellant usage for each constellation in a vertical stack (one column, three rows).
    All subplots share the same x-axis and y-limit (computed globally).
    Altitude bins are generated dynamically and renamed as "Shell X".

    Parameters:
        grouped_df (pd.DataFrame): Aggregated DataFrame with columns including:
            - "Constellation"
            - "Altitude Bin" (e.g., "Bin 1 (266 - 306)")
            - "Quiet Propellant (kg)", "Actual Propellant (kg)", "Storm Propellant (kg)"
            - "Satellite Count"

    Returns:
        None. Displays the plot.
    """

    # Prepare altitude bins dynamically: sort by the numeric part after "Bin"
    altitude_bins = sorted(
        grouped_df["Altitude Bin"].unique(), key=lambda x: int(x.split(" ")[1])
    )
    # Convert "Bin X (range)" → "Shell X"
    shell_names = [
        f"Shell {int(bin_label.split(' ')[1])}" for bin_label in altitude_bins
    ]
    altitude_ranges = [
        label.replace("Bin ", "").replace(" km", "") for label in altitude_bins
    ]

    # Extract unique constellations
    constellations = grouped_df["Constellation"].unique()

    # Define colors for each propellant condition
    colors = {"Quiet": "#440154", "Actual": "#FDE725", "Storm": "#21918c"}

    # Create subplots: vertical stack (nrows = number of constellations, 1 column), share x-axis
    n_const = len(constellations)
    fig, axes = plt.subplots(
        nrows=n_const, ncols=1, figsize=(7, 5 * n_const), dpi=300, sharex=True
    )
    fig.patch.set_facecolor("#f0f0f0")
    if n_const == 1:
        axes = [axes]

    # Precompute global y-limit across all constellations
    global_max = (
        grouped_df[
            ["Quiet Propellant (kg)", "Actual Propellant (kg)", "Storm Propellant (kg)"]
        ]
        .max()
        .max()
    )
    global_ylim = (1e-3, global_max * 1.2 if global_max * 1.2 > 1e-3 else 1e3)

    # Add combined legend (only once)
    legend_elements = [
        mpatches.Patch(color=colors["Quiet"], label="Quiet"),
        mpatches.Patch(color=colors["Actual"], label="Gannon"),
        mpatches.Patch(color=colors["Storm"], label="Severe"),
    ]

    # Loop through each constellation (vertical subplots)
    for idx, (ax, constellation) in enumerate(zip(axes, constellations)):
        data = grouped_df[grouped_df["Constellation"] == constellation].copy()

        # Replace zeros with small placeholders (if needed for log scale)
        for col in [
            "Quiet Propellant (kg)",
            "Actual Propellant (kg)",
            "Storm Propellant (kg)",
        ]:
            data[col] = data[col].replace(0.0, 0.001)

        # Precompute satellite counts per altitude bin for this constellation
        sat_count_by_bin = data.groupby("Altitude Bin")["Satellite Count"].sum()

        # Build shell labels with satellite counts
        subplot_labels = [
            f"{shell}\n n={int(sat_count_by_bin.get(bin_label, 0))}"
            for shell, bin_label in zip(shell_names, altitude_bins)
        ]

        x = np.arange(len(altitude_bins))
        width = 0.25

        # Plot bars for each condition
        ax.bar(
            x - width,
            data["Quiet Propellant (kg)"],
            width,
            label="Quiet",
            color=colors["Quiet"],
            zorder=3,
        )
        ax.bar(
            x,
            data["Actual Propellant (kg)"],
            width,
            label="Gannon",
            color=colors["Actual"],
            zorder=3,
        )
        ax.bar(
            x + width,
            data["Storm Propellant (kg)"],
            width,
            label="Severe",
            color=colors["Storm"],
            zorder=3,
        )

        ax.set_xticks(x)
        # Only label x-axis on the bottom subplot
        if idx == n_const - 1:
            ax.set_xticklabels(subplot_labels)
            ax.set_xlabel("Orbital Shell Classification", fontsize=8)

            # Add legend to the bottom subplot
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(0.75, 1),
                loc="upper left",
                frameon=False,
                fontsize=8,
            )

            # Altitude ranges text
            y_pos = 0.90
            for alt_range in altitude_ranges:
                ax.text(0.4, y_pos, alt_range, transform=ax.transAxes, fontsize=8)
                y_pos -= 0.05

        else:
            ax.set_xticklabels([])

        # Set subplot title as (a), (b), (c), etc.
        ax.set_title(
            f"({chr(97 + idx)}) Estimates of Propellant Usage by Altitude - {constellation}",
            fontsize=10,
            fontweight="bold",
            loc="left",
        )
        ax.set_ylabel("Propellant Mass (kg)", fontsize=8)

        ax.set_ylim(*global_ylim)
        # ax.set_yscale("log")

    if file_name:
        fig.savefig(file_name, dpi=300, bbox_inches="tight")

    plt.subplots_adjust(right=0.85)
    plt.show()


def plot_satellite_visibility(visibility_df, file_name):

    visible_sats = np.ceil(visibility_df["total_satellites"].mean())

    fig = plt.figure(figsize=(8, 4), dpi=500)
    plt.rcParams.update({"font.size": 10})
    ax = plt.gca()
    ax.set_facecolor("#f0f0f0")
    fig.patch.set_facecolor("#f0f0f0")
    ax.grid(True, which="major", linestyle="-", alpha=0.3, zorder=0)

    # Plot the visibility line and scatter points
    ax.plot(
        visibility_df["timestamp"],
        visibility_df["total_satellites"],
        color="#2E86AB",
        linewidth=1,
        zorder=3,
    )
    ax.scatter(
        visibility_df["timestamp"],
        visibility_df["total_satellites"],
        color="#2E86AB",
        s=2,
        zorder=4,
    )

    # Plot the mean line
    ax.axhline(
        y=visible_sats,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Mean ({int(visible_sats)} satellites)",
        zorder=2,
    )

    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    plt.xticks(rotation=0)

    ax.set_title(
        "Satellite Visibility Over Time", fontsize=10, fontweight="bold", loc="left"
    )
    ax.set_xlabel("Time (UTC) on 7 Dec 2024", fontsize=8)
    ax.set_ylabel("Number of Visible Satellites", fontsize=8)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    plt.tight_layout()

    # Save and show the figure
    fig.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.show()
