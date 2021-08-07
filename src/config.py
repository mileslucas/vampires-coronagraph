# This file holds some global config settings that are used with multiple
# parts of the rest of the code
from astropy.io import fits
import numpy as np
from scipy import ndimage

from paths import datadir

# aperture image
_aperture = fits.getdata(datadir("scexao_pupil.fits"))
# rotating the aperture
rotation_angle = -7
_aperture = ndimage.rotate(_aperture, rotation_angle, reshape=False)
threshold = 0.8
APERTURE = np.where(_aperture < threshold, 0, 1)

# telescope geometry
SUBARU_DIAMETER = 8.2  # meter
APERTURE_DIAMETER = SUBARU_DIAMETER * 0.95  # "clear aperture"

F_RATIO = 12.6
FOCAL_LENGTH = SUBARU_DIAMETER * F_RATIO  # meter

# detector geometry
VAMPIRES_NPIX = 512
VAMPIRES_SHAPE = (VAMPIRES_NPIX, VAMPIRES_NPIX)

OPTICAL_PLATE_SCALE = 1 / FOCAL_LENGTH  # radians / meter
VAMPIRES_PIXEL_PITCH = 16e-6  # meter / px
VAMPIRES_PLATE_SCALE = OPTICAL_PLATE_SCALE * VAMPIRES_PIXEL_PITCH  # radians / px

BEAM_DIAMETER = 7.2e-3  # meter

# filter information
MIN_WAVELENGTH = 600e-9  # meter
MAX_WAVELENGTH = 800e-9  # meter
BANDWIDTH = 50e-9  # meter
LAMBDA_D = MAX_WAVELENGTH / APERTURE_DIAMETER  # radian
