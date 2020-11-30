# Sky Background Brightness Calculator

This program calculates brightness distribution of the main components contributing to the brightness of the night sky (airglow, moonlight, zodiacal light) for a given date and location, and creates a time series of plots of the V-mag sky brightness distribution for each component, and for their sum. It's meant as a visualization tool, but can be potentially used for observation planning. 

There is an additional graphical interface (in Java) that uses the script to create movies of night sky brightness changes from the plots, that will be added to the repository at a later point.

## Dependencies

This program is written in Python. It is meant for Python 3.6 or newer
(Python 2.7 might work). Other dependencies:

* NumPy
* Matplotlib
* SciPy
* Astropy
* Astroplan

Additionally, the program expects a file with zodiacal light brightness
distribution coefficients to be in the same directory where it is run
 (`zod_br.txt`)

## How to use it

The programm accepts a number of command line arguments, which can be inspected as usual,
providing `-h`.

```
usage: skyb.py [-h] [--site {1,2}] [--date DATE] [--step STEP] [--flux {min,med,max}]

Calculate Sky Brightness

optional arguments:
  -h, --help            show this help message and exit
  --site {1,2}, -s {1,2}
                        Observing site, to draw coordinates and altitude and extinction.
                        1 - Mauna Kea; 2 - Roque de los Muchachos (default: 1)
  --date DATE, -d DATE  UT Date (YYYY-MM-DD). The calculations are based at midnight using this data.
                        Default: the following midnight with respect to current time
  --step STEP, -t STEP  Grid step (in degrees). Default: 1
  --flux {min,med,max}, -F {min,med,max}
                        Solar activity level. Affects the F10.7 value (min=0.8, med=1.4, max=2.0) in MJy. Default: min
```

The program could potentially use any arbitrary observing site, but it depends on an additional parameter (zenith extinction coefficient), so at the time it only allows to select one out of a few pre-determined sites for which extinction has been included.

The grid step affects the plots. It needs to be an integer number of degrees.

The script will generate its output on a directory named as the UT date at the beginning of the evening twilight, on the current working directory. If a directory exists with the same name, it's wiped out before the new results are produced.
