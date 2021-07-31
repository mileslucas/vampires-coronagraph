import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage
from hcipy import * 
from PIL import Image
import skimage.morphology as ski
from tqdm import tqdm

aperture = read_fits('scexao_pupil.fits')
wavelength = 1.6e-6 # meter
mas_pix = 16.2      # milli arcsec per pixel
#----------------------------------------------------------------------
# parameters
#----------------------------------------------------------------------
diameter   = 8.2 * 0.95 # meter (dimeter * clear aperture)

# lambda / diameter 
ld = wavelength / diameter # radians
# ld = np.degrees(ld) * 3600 * 1000 # milli arcsec

oversampling_factor = 1

# pixels size in mas
rad_pix = np.radians(mas_pix / 1000 / 3600) / oversampling_factor

# number of pixels along one axis in the pupil and focal planes
Npix_pup = aperture.shape[0]

Npix_foc = 128 * oversampling_factor

#----------------------------------------------------------------------
# setting grids, mode basis, propagators, etc  
#----------------------------------------------------------------------    
# rotating the aperture
rotation_anlge = -7
aperture = ndimage.rotate(aperture, rotation_anlge, reshape=False)
threshold = 0.8
aperture[aperture<threshold] = 0
aperture[aperture>=threshold] = 1

# generating the grids 
pupil_grid = make_pupil_grid(Npix_pup, diameter=diameter)
focal_grid = make_uniform_grid([Npix_foc, Npix_foc], [(Npix_foc)*rad_pix, (Npix_foc)*rad_pix])


# generating the propagator
propagator = FraunhoferPropagator(pupil_grid, focal_grid)

# generating Lyot stop using the erosion of the aperture
strel = ski.square(10)
lyot_mask = ski.erosion(aperture, strel)
cent_obs = make_obstructed_circular_aperture(diameter*0.85, 0.5)(pupil_grid)
lyot_mask = lyot_mask.ravel() * cent_obs

# rotating the aperture to the correct rotation and making it a field
aperture = Field(aperture.ravel(), pupil_grid)

# plt.ion()
# plt.figure()
# plt.subplot(1,2,1)
# imshow_field(aperture, cmap='gray')

# plt.subplot(1,2,2)
# imshow_field(lyot_mask, cmap='gray')
# plt.show()

# fourier transform of aperture
wavefront = Wavefront(aperture, wavelength=wavelength)
E_ref = propagator(wavefront)

img_ref = E_ref.intensity

# plt.figure()
# imshow_field(np.log10(img_ref / img_ref.max()), cmap='inferno', vmin=-5, vmax=0)
# plt.show()


#----------------------------------------------------------------------
# Coronagraph part
#----------------------------------------------------------------------    
# generating the focal plane mask
fpm_size = 217  # mas
fpm_rad = (np.radians(fpm_size / 1000 / 3600)) # mas -> rad
focal_plane_mask = circular_aperture(fpm_rad)(focal_grid)
focal_plane_mask = abs(focal_plane_mask -1)
lyot_stop = Apodizer(lyot_mask)

# Choose the type of Coronagraph
coro = LyotCoronagraph(pupil_grid, focal_plane_mask=focal_plane_mask, lyot_stop=None)
# coro = VectorVortexCoronagraph(charge=2)
# coro = PerfectCoronagraph(aperture, order=6)

lyot_plane = coro(wavefront)

post_lyot_mask = lyot_stop(lyot_plane)
img = propagator(post_lyot_mask).intensity

# plt.figure()
# plt.subplot(1,2,1)
# imshow_field(post_lyot_mask.intensity, cmap='inferno')
# plt.subplot(1,2,2)
# imshow_field(np.log10(img / img_ref.max()), vmin=-6, vmax=-2.5, cmap='inferno')
# plt.colorbar()
# plt.show()