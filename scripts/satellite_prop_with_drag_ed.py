import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# set matplotlib font to arial
import datetime as dt
from skyfield.api import Distance, load, wgs84
from skyfield.positionlib import Geocentric
import math
from pymsis import msis
import time
import matplotlib.pyplot as plt

def main():
    # USER INPUTS
    start_date = dt.datetime(2024, 5, 10, 0, 0)
    end_date = dt.datetime(2024, 5, 10, 3, 0)
    dens_model = 'msis' # options are 'msis', 'expo', or 'none'
    circ_orbit_alt = 500 # km
    incl = 53.2 # inclination in degrees
    mass = 100 #kg
    cd = 2.2
    area = 5 #m^2

    # CONSTANTS
    r_earth = 6378.137 # km 
    mu_earth = 398600.4418 # km^3/s^2
    j2 = 0.0010826269

    # Set up the initial satellite state in ECI frame 
    p1 = [circ_orbit_alt+r_earth, 0, 0]
    v1 = circular_orbit_vel(p1, incl, mu_earth)
    state1 = np.array([p1[0], p1[1], p1[2], v1[0], v1[1], v1[2]])

    # Set up time vector
    t0 = 0
    tstep = 10 # sec
    total_seconds = (end_date - start_date).total_seconds()
    t = np.arange(t0, total_seconds + tstep, tstep)
    t1 = time.time()

    # Propagate the satellite state
    yy = odeint(propagator_j2_drag, state1, t, args = (mu_earth, r_earth, j2, cd, area, mass, dens_model, start_date),rtol=1e-12, atol=1e-12)
    print('propagation took', time.time() - t1, 's')

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot a 3D sphere to represent the earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha = 0.1)
    ax.plot(yy[:, 0], yy[:, 1], yy[:, 2], 'r')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_xlim(-7000, 7000)
    ax.set_ylim(-7000, 7000)
    ax.set_zlim(-7000, 7000)
    ax.set_box_aspect([1,1,1])
    plt.show()

    # convert ECI to LLA and plot a ground track plot for the satellite, with the color of the plot being the altitude
    lats = []
    lons = []
    alts = []
    for i in range(len(yy)):
        lat, lon, alt = eci2lla(yy[i][0]*1e3, yy[i][1]*1e3, yy[i][2]*1e3, start_date + dt.timedelta(seconds=int(t[i])))        
        lats.append(lat)
        lons.append(lon)
        alts.append(alt/1000)
    
    plt.figure(figsize = (5,4))
    plt.scatter(lons, lats, c=alts)
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.colorbar(label = 'Altitude [km]')
    plt.show()

    # get density along the path using the LLA coordinates. Create a plot of density vs time from MSIS
    t_dt = [start_date + dt.timedelta(seconds=int(t[i])) for i in range(len(t))]
    msis_op = msis.run(t_dt, lats, lons, alts, geomagnetic_activity=-1)
    rho = msis_op[:,0]
    plt.figure(figsize = (5,3))
    plt.plot(t_dt, rho, 'k')
    plt.xlabel('Date')
    plt.ylabel('Density [kg/m^3]')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def circular_orbit_vel(r_init, incl, mu):
    """
    Computes the velocity vector for a circular orbit given an initial position in ECI
    and a desired inclination.
    
    Parameters:
    r_init : numpy array
        Initial position vector in ECI (km)
    incl : float
        Desired inclination in degrees
        
    Returns:
    v_init : numpy array
        Velocity vector in ECI (km/s)
    """
    r_mag = np.linalg.norm(r_init)  # Compute the orbital radius
    v_mag = np.sqrt(mu / r_mag)  # Compute circular orbital velocity

    # Unit vector in the orbital plane (assuming prograde orbit)
    r_hat = r_init / r_mag  # Radial direction
    z_hat = np.array([0, 0, 1])  # K-hat (z-axis unit vector)
    v_dir = np.cross(z_hat, r_hat)  # Get velocity direction (perpendicular to r and z)

    # Normalize velocity direction
    v_dir = v_dir / np.linalg.norm(v_dir)

    # Compute initial velocity vector in the equatorial plane
    v_init = v_mag * v_dir

    # Rotate velocity vector to match inclination
    incl_rad = np.radians(incl)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(incl_rad), -np.sin(incl_rad)],
        [0, np.sin(incl_rad), np.cos(incl_rad)]
    ])
    
    v_init = rotation_matrix @ v_init  # Rotate to desired inclination

    return v_init

def propagator_j2_drag(state, t, mu_earth, r_earth, j2, cd, area, mass, model, start_date):
    # Define the differential equation
    x, y, z, xdot, ydot, zdot = state
    r = np.sqrt(x**2 + y**2 + z**2)
    v = np.sqrt(xdot**2 + ydot**2 + zdot**2)
    a_gravity = -mu_earth / r**3 * np.array([x, y, z])
    a_j2 = j2_model(x,y,z,r, j2, mu_earth, r_earth)
    rho = dens_model(x,y,z,r,r_earth,t, model, start_date)
    a_drag = -0.5 * cd * area / mass * rho * v**2 * (np.array([xdot, ydot, zdot])/np.linalg.norm([xdot, ydot, zdot]))
    a = a_gravity + a_j2 + a_drag
    # print(t)
    return [xdot, ydot, zdot, a[0], a[1], a[2]]

def j2_model(x,y,z,r, j2, mu_earth, r_earth):
    a_j2_x = (3*j2*mu_earth*r_earth**2)/(2*r**5)*(5*(z**2/r**2)-1)*x# see https://www.vcalc.com/wiki/eng/j2+Perturbation+Acceleration
    a_j2_y = (3*j2*mu_earth*r_earth**2)/(2*r**5)*(5*(z**2/r**2)-1)*y
    a_j2_z = (3*j2*mu_earth*r_earth**2)/(2*r**5)*(5*(z**2/r**2)-3)*z
    a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z])
    return a_j2

def eci2lla(x,y,z, t_dt):
    ts = load.timescale()
    year, month, day, hour, minute = t_dt.year, t_dt.month, t_dt.day, t_dt.hour, t_dt.minute
    t = ts.utc(year, month, day, hour, minute)
    d = Distance(m=[x, y, z])
    p = Geocentric(d.au, t=t)
    g = wgs84.subpoint(p)
    return g.latitude.degrees, g.longitude.degrees, g.elevation.m

def dens_model(x,y,z,r,r_earth,t, model, start_date):
    # a_drag = 0 #-0.5 * cd * A / mass * np.exp(-(r-r_earth)/7.5) * v * np.array([xdot, ydot, zdot])
    if model == 'expo':
        rho = den_expo(r-r_earth)

    elif model == 'none':
        rho = 0

    elif model == 'msis':
        # compute the current time from start date and time delta of t
        t_dt = start_date + dt.timedelta(seconds=int(t))

        # represent this time as a string in UTC format
        t_prt = t_dt.strftime('%Y-%m-%d %H:%M:%S')
        print(t_prt) # comment this if it slows down code too much!

        # compute the lat, lon, and alt of the satellite from x, y, and z in ECI
        lat,lon,alt_m = eci2lla(x*1e3,y*1e3,z*1e3, t_dt); # everything is in degrees and meters, need to correct alt to be in m
        alt = alt_m/1e3 # km

        # run MSIS model
        data = msis.run([t_dt], [lat], [lon], [alt], geomagnetic_activity=-1)
        rho = data[0][0]
    return rho

def den_expo(h):
    # static exponential atmospheric density from vallado's Astrodynamics book
    params = [
        (0, 25, 0, 1.225, 7.249),
        (25, 30, 25, 3.899e-2, 6.349),
        (30, 40, 30, 1.774e-2, 6.682),
        (40, 50, 40, 3.972e-3, 7.554),
        (50, 60, 50, 1.057e-3, 8.382),
        (60, 70, 60, 3.206e-4, 7.714),
        (70, 80, 70, 8.77e-5, 6.549),
        (80, 90, 80, 1.905e-5, 5.799),
        (90, 100, 90, 3.396e-6, 5.382),
        (100, 110, 100, 5.297e-7, 5.877),
        (110, 120, 110, 9.661e-8, 7.263),
        (120, 130, 120, 2.438e-8, 9.473),
        (130, 140, 130, 8.484e-9, 12.636),
        (140, 150, 140, 3.845e-9, 16.149),
        (150, 180, 150, 2.070e-9, 22.523),
        (180, 200, 180, 5.464e-10, 29.74),
        (200, 250, 200, 2.789e-10, 37.105),
        (250, 300, 250, 7.248e-11, 45.546),
        (300, 350, 300, 2.418e-11, 53.628),
        (350, 400, 350, 9.518e-12, 53.298),
        (400, 450, 400, 3.725e-12, 58.515),
        (450, 500, 450, 1.585e-12, 60.828),
        (500, 600, 500, 6.967e-13, 63.822),
        (600, 700, 600, 1.454e-13, 71.835),
        (700, 800, 700, 3.614e-14, 88.667),
        (800, 900, 800, 1.17e-14, 124.64),
        (900, 1000, 900, 5.245e-15, 181.05),
        (1000, float('inf'), 1000, 3.019e-15, 268)
    ]
    
    dens = np.zeros(len(h))
    
    for i, h_ellp in enumerate(h):
        for (h_min, h_max, h_0, rho_0, H) in params:
            if h_min <= h_ellp < h_max:
                dens[i] = rho_0 * math.exp(-(h_ellp - h_0) / H)
                break
    
    return dens

main()