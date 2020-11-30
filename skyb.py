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

site_data = {
        1: {'name': 'gemini_north', 'k_V': 0.111},
        2: {'name': 'lapalma', 'k_V': 0.13}
        }

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
    Take a date and time and rounds it to the closest fraction of 10 minutes, using
    'fn' to decide if rounding up, or down.

    Inputs:
    ------
      dtime: an Astropy Time object
      fn:    function to apply for rounding
    """

    # First we round up the minutes
    dt = dtime.to_datetime()
    orig_mins = dtime.to_datetime().minute
    round_mins = fn(orig_mins / 10) * 10

    # Calculate the difference (in seconds) and apply it
    diff = (round_mins - orig_mins) * 60
    rounded_time = (dtime + TimeDelta(diff, format='sec')).to_datetime()

    # Now we want to remove seconds and microseconds
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
    # Times are passed in UT. For longitudes far from 0,
    # Astroplan may get the wrong twilight
    if mo_twi < ev_twi:
        mo_twi = obs.twilight_morning_astronomical(obs_time + 1)

    ev_twi_rounded = round_minutes(ev_twi, ceil)
    mo_twi_rounded = round_minutes(mo_twi, floor)

    return np.arange(ev_twi_rounded.to_datetime(),
                     mo_twi_rounded.to_datetime(), dtype='datetime64[10m]')

def moon_altaz_coord(ut_date_range, obs):
    """
    Generates altitude an azimuth (in radians) for the Moon, for a given range
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
    # REVIEW: This is not giving the exact same longitudes as StarAlt.
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
    K&S91 (eq. number 20)

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
    Optical path in airmasses for object at a given altitude. See
    K&S91 (eq. number 3)

    Input:
    -----

    alt:    array of altitudes (in radians)

    Returns:
    -------

    An array of altitudes for the object (in airmasses)
    """

    # zenith distance of Moon in radians
    Zmrad = (0.5 * np.pi) - alt

    return (1 - 0.96 * (np.sin(Zmrad)**2))**(-0.5) 

def read_zod_file(filename):
    """
    Read Zodiacal Light intensity (in S10 units) distribution from file. The step
    of the grid is 5 deg in both dimensions, with the Sun being at zero point.

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
        # This data represents only one quarter of the sky
        data1 = np.genfromtxt(zod_br)

    # Test format...
    nrows, ncols = data1.shape 
    if ((nrows - 1) // 2) != (ncols - 1):
        raise RuntimeError("Wrong size array when reading the zodiacal light intensity distribution file")

    # make the second quater of brightness array by reversing the order of rows
    # in the first quater
    data2 = np.zeros(data1.shape) 
    for i in range((len(data1))-1):
        # start reversing from the second to last row, to not to dublicate the 
        # central row
        data2[i,:] = data1[(((len(data1))-1)-i),:]      
    # delete the last row, since it is zeros
    data2 = np.delete(data2, ((len(data2))-1), axis=0) 

    # stick together two arrays of quaters, in vert. direction (one above another)
    data12 = np.vstack((data2,data1)) 

    # make the second HALF of brightness array by reversing the order of columns
    # in the first half
    data34 = np.zeros(data12.shape)
    for j in range((data12.shape[1])-1):
        data34[:,j] = data12[:,(((data12.shape[1])-1)-j)]
    # delete the last column, since it is zeros
    data34 = np.delete(data34, ((data34.shape[1])-1), axis=1) 

    # stick together two halfs, in horizontal direction (one after another)  
    data1234 = np.hstack((data34,data12)) 

#TODO take this part out of the function? (and two  
    # create a grid of relative ecliptic longitudes (relatively to the Sun 
    # position, so geocentric) matching zod. light brightness values data1234
    lamb_deg_rel = np.arange(0,365,5) 

    # create a grid of geocentric ecliptic latitudes, matching zod. light values
    beta_rad = np.radians(np.arange(-90,95,5))
    beta_rad = beta_rad[None,:] # making a horizontal matrix
 ##############   
    return data1234, lamb_deg_rel, beta_rad
#-------------------------------------------------------------------------------

def create_coordinate_grid(grid_step):
    """
    Generates azimuth and altitude coordinate grids to be used in plotting. 

    Inputs:
    ------
      grid_step: Spacing between coordinate points, in degrees.
                 The same step is used for each axis.

    Results:
    -------

      Two mesh grids ready for plotting (azimuth and altitude)
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
    Scales sky brightness as a function of zenith distance
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

#-------------------------------------------------------------------------------
#--- A function to do coordinate transformations from equatorial to horizontal
#--- coordinate systems. Takes four parameters: right ascension and declination
#--- of the object, local sideral time, and observatory latitude (obs_lat), 
#--- all in radians. It returns azimuth and altitude in radians.
#-------------------------------------------------------------------------------
def eqToHoriz(ra_rad, decl_rad, lst_rad, fi_rad):
    
    # First calculate hour angle for the right ascension
    hourAngle_rad = lst_rad - ra_rad
    
 #TODO maybe instead use more common North to East? 
    # For azimuth ESO convension is used: zero is at South, increasing westvards
    az_rad = (np.arctan2((np.sin(hourAngle_rad)), (np.cos(hourAngle_rad) * 
            np.sin(fi_rad) - np.tan(decl_rad) * np.cos(fi_rad))))
            
    alt_rad = (np.arcsin(np.sin(fi_rad) * np.sin(decl_rad) + np.cos(fi_rad) *
            np.cos(decl_rad) * np.cos(hourAngle_rad)))
    
    return az_rad, alt_rad
#-------------------------------------------------------------------------------
#--- A function to transform the relative ecliptic coordinate grid used for  
#--- zod. light array data1234 (and in general in literaturen for zod. light) 
#--- into horizontal coordinates, and then to do linear interpolation to make 
#--- the resulting alt_az grid evenly spaced (to avoid edge effects). The output
#--- zod. light array is in nanoLamberts.
#--- Parameters: 

#-------------------------------------------------------------------------------
def zodiac_grid(latitude, az, alt, lamb_sun, LST_rad, lamb_deg_rel, beta_rad, data1234):
    """
    A function to transform the relative ecliptic coordinate grid used for  
    zod. light array (and in general in literature for zod. light) into horizontal
    coordinates, and then to do linear interpolation to make the resulting alt_az
    grid evenly spaced (to avoid edge effects).

    Returns:
    -------
      Zod. light array is in nanoLamberts.
    """
    # calculate the grid of absolute ecliptic longitudes for zod. light
    # brightness grid depending on the Sun's absolute ecliptic longitude
    lamb_deg = lamb_sun + lamb_deg_rel 
    lamb_rad = np.radians(lamb_deg)
    lamb_rad = lamb_rad[:,None] # make a vertical matrix

# TODO make a separate function
    # transform ecliptical coords into equatorial
    # right ascension array IN RADIANS, NOT IN HOURS!!!!
    alpha_rad = np.arctan2((np.sin(lamb_rad) * np.cos(obliq) - np.tan(beta_rad) *
        np.sin(obliq)),(np.cos(lamb_rad))) 
    
    # making R.A. in range of [0;2pi)
    alpha_neg = alpha_rad < 0
    alpha_rad[alpha_neg] = alpha_rad[alpha_neg] + (2 * np.pi)
    
    # declination array
    delta_rad = np.arcsin(np.sin(beta_rad) * np.cos(obliq) + np.cos(beta_rad) * 
        np.sin(obliq) * np.sin(lamb_rad))

# TODO transform straight from ecliptic to horizontal?
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
    # cartesian coordinates. The interpolation is needed in order to make zod.
    # light grid evenly spaced, since after transforming from ecliptic
    # coordinates to horizontal the grid step is not constant anymore. With not
    # constant step there is an unwanted egde effect around the pole. So this 
    # interpolation on evenly spaced grid is done to fix it.
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
  
#TODO why here???
    # convert zodiacal light intensity values from S10 to nL
    grid_zod = S10_to_nL(grid_S10)

    # zod light intensity in V mags?
    #Zod_V_int1 = (20.7233-(np.log(grid_z1/34.08)))/0.92104

    return grid_zod

################################ OLD CODE #####################################
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

    Returns:
    --------
    Bmoon:       Lunar sky brightness
    """
    
    #!!! use astropy SkyCoord.separation(SkyCoord1)
    delta_az = az - azm
    
#TODO separate function for angular distances
    # angular distance (in rad) between two points with coordinates (az, alt)
    # and (azm, altm)
    dist_rad = np.arctan2(np.sqrt((np.cos(alt) * np.sin(delta_az))**2 + 
        (np.cos(altm) * np.sin(alt) - np.sin(altm) * np.cos(alt) * 
        np.cos(delta_az))**2), (np.sin(altm) * np.sin(alt) + np.cos(altm) * 
        np.cos(alt) * np.cos(delta_az)))
    
    dist_deg = np.degrees(dist_rad)

    dist_deg5 = dist_deg < 5

    dist_deg10 = np.logical_and(5 <= dist_deg, dist_deg < 10)
    # why 90????
    dist_deg90 = dist_deg >= 10

    # Mie scattering contribution of the aerosols to the scaterring function
    # (two cases)
    f_mie = np.empty(dist_deg.shape)
    # set constant brightness for the small angles (for presentation purposes)
    f_mie[dist_deg5] = 20
    # for angular separation from Moon less than 10 deg
    f_mie[dist_deg10] = 6.2 * 10**7 * ((dist_deg[dist_deg10])**(-2))
    # for angular separation >=10 deg
    f_mie[dist_deg90] = 10**(6.15 - ((dist_deg[dist_deg90]) / 40))
    
    # Rayleigh scattering from atmospheric gases
    f_rayl = (10**5.36) * (1.06 + (np.cos(dist_rad))**2)
    
    # total scattering function
    f_dist = f_rayl + f_mie
    
    # Lunar sky brightness in nanoLamberts
    Bmoon = f_dist * I * 10**(-0.4 * k_V * X_Zm) * (1 - 10**(-0.4 * k_V * X_Z))
    
    return Bmoon
#-------------------------------------------------------------------------------
def plot_polar_contour(values, az, alt):
    theta = az
    r = 90 - alt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    # fig, ax = subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    plt.ylim((0,90))

    contourf_levels = np.arange(14, 25, 0.1)
    # create filled contour plot (winter_r is for reverse color map "winter")
    cax = plt.contourf(theta, r, abs(values), contourf_levels, cmap=plt.cm.jet_r)

    mask = np.logical_not(np.isnan(values))
    non_nan = values[mask]
    # number of contour levels to draw contour lines for the particular image is
    # (max_val-min_val)*10
    x = int((max(non_nan)-min(non_nan))/0.1)
    cax3 = plt.contour(theta, r, abs(values), x, cmap=plt.cm.jet_r)

    # add red contour levels for each integer magnitude
    contour_levels = np.arange(14, 25, 1)
    cax2 = plt.contour(theta, r, abs(values), contour_levels,
                        colors = 'r',
                        origin=origin)
#                        hold='on')
    plt.clabel(cax2, fmt = '%2.1f', colors = 'black', fontsize=14)

    # create color bar with label for the plot cax
    cb = fig.colorbar(cax)
    cb.set_label("Sky brightening in V-band (mag)")
    # add contour lines to the color bar 
    cb.add_lines(cax3)

    #plot object as a star and it's track
    #ax.plot(az_obj, (90-np.degrees(alt_obj)),'k-', lw=0.3)
    #ax.plot([curr_az_obj], [90-np.degrees(curr_alt_obj)],'k*', ms=10) 

    #return fig, ax, cax, cax2, cax3
    #return fig, ax, cax, cax3
#-------------------------------------------------------------------------------

class Plotter:
    def __init__(self, mesh_az, mesh_alt):
        self.az = mesh_az
        self.alt = mesh_alt

    def save(self, date, arr, path, fname):
        plot_polar_contour(arr, self.az, self.alt)
        plt.figtext(0.02,0.95, date, ha='left')
        plt.savefig(os.path.join(path, fname))
        plt.close()

def main(site, date, grid_step, F107, k_V):
    #-------------------------------------------------------------------------------
    #--- read data for the night from STARALT generated file curr_coord.out, which 
    #--- is a table with following values for every 10 mins starting from the end of
    #--- evening twilight rounded to the next 10 and ending before the morning tw.:  
    #---    LST,  Moon_RA, Moon_Decl,  Suns_ecliptic_longitude

    # Extract the dimensionless value for the site's latitude
    latitude = site.location.lat.rad
    time_range = create_time_range(date, site)
    phase = moon_phase_angle(time_range[0], site)
    alt_m, az_m = moon_altaz_coord(time_range, site)
# The original. See the shape...     lamb_sun = coord_grid[:,3]			# make Sun ecl. latitude array
    lamb_sun = sun_ecliptic_long(time_range)

    # create the coordinate grid for the plot
    # mesh_az_deg, mesh_alt_deg = create_coordinate_grid(grid_step)
    mesh_alt_deg, mesh_az_deg = create_coordinate_grid(grid_step)
    mesh_az, mesh_alt = np.radians(mesh_az_deg), np.radians(mesh_alt_deg)

    # calculate Moon illuminance outside the atmosphere
    I = moon_illuminance(phase)
    # optical path in airmasses for Moon
    X_Zm = optical_path(alt_m)

    # create a zod. brightness array and corresponding ecl. coord arrays
    data1234, lamb_deg_rel, beta_rad = read_zod_file("./zod_br_formated.txt")

    # Transform altitude values into airmasses
    X_Z = optical_path(mesh_alt)

    #---- airglow component (same for all the frames of the particular night -------
    # zenith sky brightness due to airglow, depending on F107
    # And obtain it in nL
    Bzen_ag = S10_to_nL(145 + 130 * (F107 - 0.8) / 1.2)

    # sky backgroung due to airglow, scaled with airmass
    B0_Za = am_scaled(k_V, X_Z, Bzen_ag)
    # sky brightness in V mag due to airglow
    B0Z_Va = nL_to_Vmag(B0_Za)
    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    #---up to here the calculations are same for all the frames of the same night --
    #-------------------------------------------------------------------------------
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
            # sky brightening due to Moon over the background, dV mag
            dV_Bmoon = -2.5 * np.log10((Bmoon + B0_Za + B0_Zz) / (B0_Za + B0_Zz)) 
            # sky brightness due to Moon, V mag, aimass scaled
            Bmoon_V = nL_to_Vmag(Bmoon)
            # summary sky brightness due to Moon, airglow and zod.light, airmass sc.
            Bfinal_V = nL_to_Vmag(Bmoon + B0_Za + B0_Zz)
            # make a plot (with object)of Moon brightness distribution
            plotter.save(plotting_date, Bmoon_V, Bmoon_path, f"Bmoon_V_{n+100}.png") 
        else:
            Bfinal_V = B0Z_V
            plt.figtext(0.02,0.95, plotting_date, ha='left')
            plt.figtext(0.5, 0.5,'The Moon is below the horizon', ha='center')
            plt.savefig(os.path.join(Bmoon_path, f'Bmoon_V_{n+100}.png'))
            plt.close()

        # make a plot of final sky brightness, with object
        plotter.save(plotting_date, Bfinal_V, Bfinal_path, f'Bfinal_V_{n+100}.png')

        #plot_polar_contour(B0Z_Va, az, mesh_alt_deg)
        #plt.savefig('B0Z_Va_%.0f.png' % (n+100))
        # make a plot of zodiacal light distribution with obj.

        plotter.save(plotting_date, B0Z_Vz, Bzod_path, f'B0Z_Vz_{n+100}.png')

    # make a plot of sky brightness due to airglow (only one)
    plotter.save(plotting_date, B0Z_Va, Bairglow_path, 'B0Z_Va.png')

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
        # TODO: Fix this... not pretty
        now = Time(args.date)
        date = obs.midnight(now, which='nearest')

    flux = f107_values[args.flux]

    try:
        main(obs, date, args.step, flux, k_V)
    except RuntimeError as e:
        print(e)
        sys.exit(-1)
