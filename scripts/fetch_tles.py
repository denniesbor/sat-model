# ---------------------------------------------------------------------------
# Description: This script fetches the latest TLEs from celestrak.com
# and stores them in a file. The file is then read by the main program for use
# in the simulation of the satellites. We begin by downloading the TLEs of the
# SpaceX Starlink satellites.
# Author:  Dennies Bor
# Date:    2024-08-22
# ---------------------------------------------------------------------------

import requests
import json


def get_starlink_tles():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"

    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        return f"Failed to retrieve data. Status code: {response.status_code}"


def main():
    tle_data = get_starlink_tles()

    print(tle_data[:500])  # Print first 500 characters as a preview

    # Optionally, save the data to a file
    with open("starlink_tles.txt", "w") as file:
        file.write(tle_data)

    print("\nFull TLE data has been saved to 'starlink_tles.txt'")


if __name__ == "__main__":
    main()
