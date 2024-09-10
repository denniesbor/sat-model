# %%
from skyfield.api import load
from skyfield.api import EarthSatellite
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# %%

tle_lines = [
    "1 25544U 98067A   21274.92574537  .00001450  00000-0  34120-4 0  9995",
    "2 25544  51.6458  27.4091 0002267  34.2689 325.8411 15.48815303292616",
]
ts = load.timescale()

# Load satellite from TLE
satellite = EarthSatellite(tle_lines[0], tle_lines[1], "ISS (ZARYA)", ts)


times = ts.utc(2024, 9, 5, range(0, 60))  # Example: simulate for 60 seconds

# %%

positions = satellite.at(times).position.km  # Positions in km
x, y, z = positions

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Set up plot limits (size of Earth)
ax.set_xlim([-7000, 7000])
ax.set_ylim([-7000, 7000])
ax.set_zlim([-7000, 7000])

# Labels
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")

# Initialize an empty plot line for the orbit
(line,) = ax.plot([], [], [], "b-", label="Satellite Orbit", lw=2)
(point,) = ax.plot([], [], [], "ro", label="Satellite", markersize=6)


# %%
def update(num, x, y, z, line, point):
    # Update the line (orbit)
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])

    # Update the point (satellite)
    point.set_data(x[num], y[num])
    point.set_3d_properties(z[num])

    return line, point


# %%
ani = FuncAnimation(
    fig,
    update,
    frames=len(times),
    fargs=(x, y, z, line, point),
    interval=100,
    blit=False,
)

# Display the plot
plt.legend()
plt.show()

# %%
ani.save("satellite_orbit.mp4", writer="ffmpeg", fps=10)

# %%
