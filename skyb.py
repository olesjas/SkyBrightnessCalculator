#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.interpolate import griddata
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord, EarthLocation, get_sun
from astropy.time import Time, TimeDelta
import astroplan
from argparse import ArgumentParser
from math import ceil, floor
import sys
import os
import shutil

# zenith extinction coef. in V-band for differnt sites
site_data = {
        1: {'name': 'gemini_north', 'k_V': 0.111},
        2: {'name': 'lapalma', 'k_V': 0.13}
        }

# Typical F10.7 index values (in MJy) for the min, mid and max of solar cycle
f107_values = {
        'min': 0.8,
        'med': 1.4,
        'max': 2.0
        }

obliq = np.radians(23.439) # obliquity of ecliptic (J2000.0)
origin = 'lower'    # vert. axes direction 


def parse_arguments():
    parser = ArgumentParser(description='Calculate Sky Brightness')
    parser.add_argument('--site', '-s', type=int, default=1, choices=(1, 2),
            help='Observing site, to draw coordinates and altitude and extinction. 1 - Mauna Kea; 2 - Roque de los Muchachos (default: 1)')
    parser.add_argument('--date', '-d', default=None,
            help="UT Date (YYYY-MM-DD). The calculations are based at midnight using this data. Default: the following midnight with respect to current time")
    parser.add_argument('--step', '-t', type=int, default=1,
            help="Grid step (in degrees). Default: 1")
    parser.add_argument('--flux', '-F', type=str, default='min', choices=('min', 'med', 'max'),
            help="Solar activity level. Affects the F10.7 value (min=0.8, med=1.4, max=2.0) in MJy. Default: min")

    return parser.parse_args(sys.argv[1:])

def round_minutes(dtime, fn):
    """
    Takes a date and time and rounds it to the closest fraction of 10 minutes, using
    'fn' to decide if rounding up, or down.

    Inputs:
    ------
    dtime: an Astropy Time object
    fn:    function to apply for rounding
    """

    # Round up the minutes
    dt = dtime.to_datetime()
    orig_mins = dtime.to_datetime().minute
    round_mins = fn(orig_mins / 10) * 10

    # Calculate the difference (in seconds) and apply it
    diff = (round_mins - orig_mins) * 60
    rounded_time = (dtime + TimeDelta(diff, format='sec')).to_datetime()

    # Remove seconds and microseconds
    return Time(rounded_time.strftime('%Y-%m-%d %H:%M:00'))

def create_time_range(ut_date, obs):
    """
    Generates an array of times between astronomical twilights, with a 10
    minute step, for a given observing site.

    Inputs:
    ------
    ut_date:   UT Date of observation
    obs:       An astroplan Observer object

    Returns
    -------
    
    A grid of equally spaced timestamps
    """

    obs_time = Time(ut_date)
    ev_twi = obs.twilight_evening_astronomical(obs_time)
    mo_twi = obs.twilight_morning_astronomical(obs_time)

    if mo_twi < ev_twi:
        mo_twi = obs.twilight_morning_astronomical(obs_time + 1)

    ev_twi_rounded = round_minutes(ev_twi, ceil)
    mo_twi_rounded = round_minutes(mo_twi, floor)

    return np.arange(ev_twi_rounded.to_datetime(),
                    mo_twi_rounded.to_datetime(), dtype='datetime64[10m]')

def moon_altaz_coord(ut_date_range, obs):
    """
    Generates altitude and azimuth (in radians) for the Moon, for a given range
    of dates, at a certain location.

    Input:
    -----

    ut_date_range:  A sequence of UT date/time in a format that can be accepted
                    by astropy.time.Time
    obs:            An astroplan Observer object

    Returns
    -------

    Two arrays, the first one containing the altitudes, the second one the azimuths
    """
    coordinates = [obs.moon_altaz(dt) for dt in ut_date_range]

    return np.radians([coord.alt.value for coord in coordinates]), (np.radians([coord.az.value for coord in coordinates]) + np.pi)

def sun_ecliptic_long(ut_date_range):
    """
    Generates the ecliptic longitudes of the Sun for a given range of dates.

    Input:
    -----
    ut_date_range:  A sequence of UT date/time in a format that can be accepted
                    by astropy.time.Time

    Returns:
    -------
    An array of Sun ecliptic longitudes in degrees

    """
    return np.array([get_sun(Time(dt)).geocentrictrueecliptic.lon.value for dt in ut_date_range])

def moon_phase_angle(ut_date, obs):
    """
    Given a UT date, and an observing site, return the moon phase angle in degrees

    Inputs:
    ------
    ut_date_range:  A of UT date/time in a format that can be accepted
                    by astropy.time.Time
    obs:            An astroplan Observer object

    Returns:
    -------

    Moon phase angle in degrees
    """

    return Angle(obs.moon_phase(ut_date)).deg

def moon_illuminance(phase):
    """
    Calculates Moon illuminance outside the atmosphere, according to
    K&S91 [20]

    Input:
    -----

    phase:    Moon phase angle in degrees

    Returns:
    -------

    Moon illuminance
    """
    return 10**(-0.4 * (3.84 + 0.026 * abs(phase) + 4 * 10**(-9) * abs(phase)**4))

def optical_path(alt):
    """
    Optical pathlength along a line of sight in airmasses for object at a given altitude.
    (K&S91 [3])

    Input:
    -----

    alt:    array of altitudes (in radians)

    Returns:
    -------

    An array of altitudes for the object (in airmasses)
    """

    # zenith distance of the object in radians
    Zrad = (0.5 * np.pi) - alt

    return (1 - 0.96 * (np.sin(Zrad)**2))**(-0.5) 

def read_zod_file(filename):
    """
    Read zodiacal light intensity (in S10 units) distribution from file. 
    The file supplies only one quarter of the sky sphere with the Sun in zero point
    in the lower left corner, relative ecliptic longitude increasing upwards from
    0 to 180 deg, and ecliptic latitude increasing from 0 to 90 deg from left to right.
    Coordinate grid step is 5 deg. Missing data in close proximity to the Sun is
    set to zeros.

    Returns:
    -------

    Three arrays:
    - Full-sky zodiacal light intensity distribution array
    - Grid of relative ecliptic longitudes
    - Grid of ecliptic latitudes

    Both longitude/latitude arrays match the shape of the first one.
    """
    with open(filename) as zod_br:
        data1 = np.genfromtxt(zod_br)   # This data represents only one quarter of the sky

    # Test array format
    nrows, ncols = data1.shape 
    if ((nrows - 1) // 2) != (ncols - 1):
        raise RuntimeError("Wrong size array when reading the zodiacal light intensity distribution file")

    # make 2nd quarter or zod. brightness array by symmetric transformations of the 1st quarter
    data2 = np.zeros(data1.shape) 
    # reverse row order starting with second to last row
    for i in range((len(data1))-1):
        data2[i,:] = data1[(((len(data1))-1)-i),:]      
    data2 = np.delete(data2, ((len(data2))-1), axis=0)  # delete the last row with zeros
    data12 = np.vstack((data2,data1))   # stack together the two quaters, one above another

    data34 = np.zeros(data12.shape) # make 2nd half of brightness array
    # reverse column order of the right half
    for j in range((data12.shape[1])-1):
        data34[:,j] = data12[:,(((data12.shape[1])-1)-j)]
    data34 = np.delete(data34, ((data34.shape[1])-1), axis=1)   # delete the last column with zeros
    # stack together two halfs, side by side  
    data1234 = np.hstack((data34,data12)) 

    # create a grid of relative ecliptic longitudes for the data1234 array
    lamb_deg_rel = np.arange(0,365,5) 
    # create a grid of ecliptic latitudes for the data1234 array
    beta_rad = np.radians(np.arange(-90,95,5))
    beta_rad = beta_rad[None,:] # make a horizontal matrix

    return data1234, lamb_deg_rel, beta_rad


def create_coordinate_grid(grid_step):
    """
    Generates azimuth and altitude coordinate grids to be used in plotting. 

    Inputs:
    ------
    grid_step: Spacing between coordinate points, in degrees.
                The same step is used for each axis.

    Results:
    -------

    Two mesh grids ready for plotting (azimuth and altitude, deg)
    """
    # create the coordinate grid for the plot
    az = np.linspace(-180, 181,((360 // grid_step) + 1))
    alt = np.linspace(0, 90,((90 // grid_step) + 1))

    return np.meshgrid(alt, az)

def S10_to_nL(arr):
    """
    Take an array and convert it from S10 units to nanoLamberts
    """
    return arr * 0.263

def am_scaled(k_V, X_Z, B_zen):
    """
    Calculates sky brightness as a function of zenith distance (K&S91 [23])
    
    Inputs:
    ------
    k_V:      zenith ext. coeff. in V
    X-Z:      zenith distance in radians
    B_zen:    sky brightness in zenith
    
    Results:
    -------
    
    """
    return (X_Z * B_zen) * 10**(-0.4 * k_V * (X_Z - 1)) 

def nL_to_Vmag(arr):
    """
    Convert an array from nL units to V in mag/arcsec^2
    """
    return (20.7233 - (np.log(arr / 34.08))) / 0.92104 

def create_folder(path, name):
    """

    """
    new_path = os.path.join(path, name)
    os.makedirs(new_path)
    return new_path

def eqToHoriz(ra_rad, decl_rad, lst_rad, fi_rad):
    """
    Transforms RA and Decl coordinate arrays from equatorial coord. system to horizontal. 
    
    Inputs:
    ------
    ra_rad:   object's RA (rad)
    decl_rad: object's Decl (rad)
    lst_rad:  Local Siderial Time of observation (rad)
    fi_rad:   observer's latitude (rad)
    
    Results:
    -------
        
        Two arrays, the first one containing the altitudes, the second one the azimuths (in radians)
    """
    hourAngle_rad = lst_rad - ra_rad        # calculate hour angle for the right ascension
    
    # For azimuth ESO convension is used: zero is at South, increasing westvards
    az_rad = (np.arctan2((np.sin(hourAngle_rad)), (np.cos(hourAngle_rad) * 
            np.sin(fi_rad) - np.tan(decl_rad) * np.cos(fi_rad))))
    
    alt_rad = (np.arcsin(np.sin(fi_rad) * np.sin(decl_rad) + np.cos(fi_rad) *
            np.cos(decl_rad) * np.cos(hourAngle_rad)))
    
    return az_rad, alt_rad

def zodiac_grid(latitude, az, alt, lamb_sun, LST_rad, lamb_deg_rel, beta_rad, data1234):
    """
    A function that transforms the coordinates of full sky zod. brightness distribution array from
    relative ecliptic coordinate into horizontal coordinates. A linear interpolation is then done 
    on the resulting array in order to make it evenly spaced to avoid edge effects.
    
    Inputs:
    ------
    latitude: observer's latitude (rad)
    az, alt:  grids of azimuth and altitude coordinates used for the sky brightness plot (deg)
    lamb_sun: ecliptic longitude of the sun (deg)
    LST_rad:  local siderial time of observations (rad)
    lamb_deg_rel: relative ecliptic longitude array for the full sky zod. brightness distr. array (deg)
    beta_rad: ecliptic latitude array for the full sky zod. brightness distr. array (rad)
    data1234: full sky zod. brightness distribution array

    Returns:
    -------
    Zod. light array is in nanoLamberts.
    """
    # convert relative ecliptic long. to absolute ones by addig ecl. longitude of the sun
    lamb_deg = lamb_sun + lamb_deg_rel 
    lamb_rad = np.radians(lamb_deg)
    lamb_rad = lamb_rad[:,None] # make a vertical matrix

    # transform ecliptic coords into equatorial
    alpha_rad = np.arctan2((np.sin(lamb_rad) * np.cos(obliq) - np.tan(beta_rad) *
        np.sin(obliq)),(np.cos(lamb_rad))) 
    
    # making R.A. in range of [0;2pi)
    alpha_neg = alpha_rad < 0
    alpha_rad[alpha_neg] = alpha_rad[alpha_neg] + (2 * np.pi)
    
    delta_rad = np.arcsin(np.sin(beta_rad) * np.cos(obliq) + np.cos(beta_rad) * 
        np.sin(obliq) * np.sin(lamb_rad))

    # transform equatorial coords into horizontal
    az_rad, alt_rad = eqToHoriz(alpha_rad, delta_rad, LST_rad, latitude)

    az_deg = np.degrees(az_rad)
    alt_deg = np.degrees(alt_rad)

    # create linear arrays out of two-dimensional grids of alt, az and zod.
    # light intensity
    alt_az = np.zeros((alt_deg.size,2)) # an array with two columns: alt and az
    data1234_lin = np.zeros(alt_deg.size)   # an array with one column

    for i in range (alt_deg.shape[0]):
        for j in range (alt_deg.shape[1]):
            alt_az[((i * alt_deg.shape[1]) + j),0] = alt_deg[i,j]
            alt_az[((i * alt_deg.shape[1]) + j),1] = az_deg[i,j]
            data1234_lin[((i * alt_deg.shape[1]) + j)] = data1234[i,j]

    # The following coordinate transformations from polar to cartesian system are
    # in order to use a python interpolation function griddata, which works with
    # cartesian coordinates. The interpolation is needed to avoid egde effect around the pole.
    alt_az_cart=np.zeros(alt_az.shape)
    # expressing polar (alt, az) coordinates as cartesian x
    alt_az_cart[:,0] = (90 - alt_az[:,0])*np.cos(np.radians(alt_az[:,1]))
    # expressing polar (alt, az) coordinates as cartesian y
    alt_az_cart[:,1] = (90 - alt_az[:,0])*np.sin(np.radians(alt_az[:,1]))
    # expressing polar coord. grid on which to interpolate in cartesian coord.
    grid_x = (90 - alt) * np.cos(np.radians(az)) 
    grid_y = (90 - alt) * np.sin(np.radians(az))
    # do linear interpolation
    grid_S10 = griddata(alt_az_cart, data1234_lin, (grid_x, grid_y),
                method='linear')

    # convert zodiacal light intensity values from S10 to nL
    grid_zod = S10_to_nL(grid_S10)

    return grid_zod

def moon_grid(az, alt, X_Z, azm, altm, X_Zm, I, k_V):
    """
    Calculate lunar sky brightness (in nanoLamberts) of a point in the sky,
    as a function of the angular distance to the Moon.

    Parameters:
    -----------

    az, alt:     horiz. coordinates of the point
    X_Z:         optical pathlength along the line of sight, for the point (in airmasses)
    azm, altm:   horiz. coordinates of the Moon
    X_Zm:        optical pathlength along the line of sight, for the Moon (in airmasses)
    I:           illuminance of the Moon outside the atmosphere (I*, in footcandles)
    k_V:         zenith extinction coeff. (Vmag)

    Returns:
    --------
    Bmoon:       Lunar sky brightness in nL
    """
    delta_az = az - azm
    # angular distance (in rad) between two points with coordinates (az, alt) and (azm, altm)
    dist_rad = np.arctan2(np.sqrt((np.cos(alt) * np.sin(delta_az))**2 + 
        (np.cos(altm) * np.sin(alt) - np.sin(altm) * np.cos(alt) * 
        np.cos(delta_az))**2), (np.sin(altm) * np.sin(alt) + np.cos(altm) * 
        np.cos(alt) * np.cos(delta_az)))
    dist_deg = np.degrees(dist_rad)
    
    # calculate Mie scattering contribution of the aerosols to the scaterring function
    f_mie = np.empty(dist_deg.shape)
    # set constant brightness for the small angles (for presentation purposes)
    dist_deg5 = dist_deg < 5
    f_mie[dist_deg5] = 20
    # Mie scat. for angular separation from Moon less than 10 deg
    dist_deg10 = np.logical_and(5 <= dist_deg, dist_deg < 10)
    f_mie[dist_deg10] = 6.2 * 10**7 * ((dist_deg[dist_deg10])**(-2))
    # Mie scat. for angular separation >=10 deg
    dist_deg90 = dist_deg >= 10
    f_mie[dist_deg90] = 10**(6.15 - ((dist_deg[dist_deg90]) / 40))
    
    # Rayleigh scattering from atmospheric gases
    f_rayl = (10**5.36) * (1.06 + (np.cos(dist_rad))**2)
    
    # total scattering function
    f_dist = f_rayl + f_mie
    
    # Lunar sky brightness in nanoLamberts
    Bmoon = f_dist * I * 10**(-0.4 * k_V * X_Zm) * (1 - 10**(-0.4 * k_V * X_Z))
    
    return Bmoon

def plot_polar_contour(values, az, alt):
    """
Create a plot of sky brightness distribution in polar coordinates with isophotes

    Parameters:
    -----------

    values:  sky brightness array
    alt,az:  horiz. coordinate arrays (deg)
    """
    theta = az
    r = 90 - alt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location("N")
    plt.ylim((0,90))

    contourf_levels = np.arange(14, 25, 0.1)
    # create filled contour plot 
    cax = plt.contourf(theta, r, abs(values), contourf_levels, cmap=plt.cm.jet_r)

    mask = np.logical_not(np.isnan(values))
    non_nan = values[mask]
    # number of contour levels to draw contour lines for the particular image
    x = int((max(non_nan)-min(non_nan))/0.1)
    cax3 = plt.contour(theta, r, abs(values), x, cmap=plt.cm.jet_r)

    # add red contour levels for each integer magnitude
    contour_levels = np.arange(14, 25, 1)
    cax2 = plt.contour(theta, r, abs(values), contour_levels,
                        colors = 'r',
                        origin=origin)
    plt.clabel(cax2, fmt = '%2.1f', colors = 'black', fontsize=14)

    # create color bar with label for the plot cax
    cb = fig.colorbar(cax)
    cb.set_label("Sky brightening in V-band (mag)")
    # add contour lines to the color bar 
    cb.add_lines(cax3)

class Plotter:
    def __init__(self, mesh_az, mesh_alt):
        self.az = mesh_az
        self.alt = mesh_alt

    def save(self, date, moon_illum, arr, path, fname):
        plot_polar_contour(arr, self.az, self.alt)
        plt.figtext(0.02,0.95, date, ha='left')
        plt.figtext(0.02,0.92, 'Moon illum. ={0:.0f}%'.format(100*moon_illum), ha='left')
        plt.savefig(os.path.join(path, fname))
        plt.close()

def main(site, date, grid_step, F107, k_V):
    
    latitude = site.location.lat.rad
    time_range = create_time_range(date, site)
    phase = moon_phase_angle(time_range[0], site)
    moon_illum = astroplan.moon_illumination(Time(time_range[0]), ephemeris=None)
    alt_m, az_m = moon_altaz_coord(time_range, site)
    lamb_sun = sun_ecliptic_long(time_range)

    # create the coordinate grid for the plot
    mesh_alt_deg, mesh_az_deg = create_coordinate_grid(grid_step)
    mesh_az, mesh_alt = np.radians(mesh_az_deg), np.radians(mesh_alt_deg)

    # calculate Moon illuminance outside the atmosphere
    I = moon_illuminance(phase)
    # optical path in airmasses for Moon
    X_Zm = optical_path(alt_m)

    # create a zod. brightness array and corresponding ecl. coord arrays
    data1234, lamb_deg_rel, beta_rad = read_zod_file("./zod_br.txt")

    # Transform altitude values into airmasses
    X_Z = optical_path(mesh_alt)

    # -------airglow component (same for all the frames of the particular night) -------
    
    # zenith sky brightness due to airglow as function of F107 (B&E98)
    Bzen_ag = S10_to_nL(145 + 130 * (F107 - 0.8) / 1.2)

    # sky backgroung due to airglow, scaled with airmass
    B0_Za = am_scaled(k_V, X_Z, Bzen_ag)
    # sky brightness in V mag due to airglow
    B0Z_Va = nL_to_Vmag(B0_Za)

    path = Time(time_range[0]).strftime('%Y-%m-%d')
    if os.path.exists(path):
        shutil.rmtree(path)

    Bmoon_path = create_folder(path, 'Bmoon')
    Bfinal_path = create_folder(path, 'Bfinal')
    Bzod_path = create_folder(path, 'Bzod')
    Bairglow_path = create_folder(path, 'Bairglow')

    LST_rad = obs.local_sidereal_time(time_range).rad

    plotter = Plotter(mesh_az, mesh_alt_deg)

    for n in range(len(LST_rad)):

        plotting_date = Time(time_range[n]).strftime('%Y-%m-%d %H:%M UT')
    # ----------------------zodiacal light component---------------
        # get grid_zod in nL in shape of (az, alt)
        grid_zod = zodiac_grid(latitude, mesh_az_deg, mesh_alt_deg,
                            lamb_sun[n], LST_rad[n],
                            lamb_deg_rel, beta_rad, data1234)

        # sky backgroung due to zodiacal light, scaled with airmass
        B0_Zz = grid_zod * 10**(-0.4 * k_V * (X_Z - 1))
        # sky brightness in V mag due to zodiacal light, scaled with airmass
        B0Z_Vz = nL_to_Vmag(B0_Zz)

        # ------- sky brightness in V mag due to both airglow AND zodiacal light -------    
        B0Z_V = nL_to_Vmag(B0_Zz + B0_Za)

    #-----------------Moon brightness component-------------------
        if (alt_m[n])>0:
            Bmoon = moon_grid(mesh_az, mesh_alt, X_Z, az_m[n],alt_m[n],X_Zm[n], I, k_V)
            # sky brightness due to Moon, V mag, aimass scaled
            Bmoon_V = nL_to_Vmag(Bmoon)
            # summary sky brightness due to Moon, airglow and zod.light, airmass sc.
            Bfinal_V = nL_to_Vmag(Bmoon + B0_Za + B0_Zz)
            # make a plot (with object)of Moon brightness distribution
            plotter.save(plotting_date, moon_illum, Bmoon_V, Bmoon_path, f"Bmoon_V_{n+100}.png") 
        else:
            Bfinal_V = B0Z_V
            plt.figtext(0.02,0.95, plotting_date, ha='left')
            plt.figtext(0.5, 0.5,'The Moon is below the horizon', ha='center')
            plt.savefig(os.path.join(Bmoon_path, f'Bmoon_V_{n+100}.png'))
            plt.close()

        # make a plot of final sky brightness, with object
        plotter.save(plotting_date, moon_illum, Bfinal_V, Bfinal_path, f'Bfinal_V_{n+100}.png')

        # make a plot of zodiacal light distribution with obj.
        plotter.save(plotting_date, moon_illum, B0Z_Vz, Bzod_path, f'B0Z_Vz_{n+100}.png')

    # make a plot of sky brightness due to airglow (only one)
    plotter.save(plotting_date, moon_illum, B0Z_Va, Bairglow_path, 'B0Z_Va.png')

if __name__ == '__main__':
    args = parse_arguments()

    sd = site_data[args.site]
    obs = astroplan.Observer.at_site(sd['name'])
    k_V = sd['k_V']

    if args.date is None:
        # Get the default date if none provided
        # Base it on current UTC time
        now = Time.now()
        date = obs.midnight(now, which='next')
    else:
        now = Time(args.date)
        date = obs.midnight(now, which='nearest')

    flux = f107_values[args.flux]

    try:
        main(obs, date, args.step, flux, k_V)
    except RuntimeError as e:
        print(e)
        sys.exit(-1)
