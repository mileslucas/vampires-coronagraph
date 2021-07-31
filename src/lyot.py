import astropy.units as u
import cv2
import hcipy as hp
import numpy as np
import proplot as pro
from astropy.io import fits
from tqdm import tqdm

from .paths import datadir, figuredir

# ----------------------------------------------------------------------
# Plotting defaults
# ----------------------------------------------------------------------
pro.use_style("ggplot")
pro.rc["image.cmap"] = "inferno"
pro.rc["grid"] = False


# ----------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------
aperture = fits.getdata(datadir("scexao_pupil.fits"))
wavelength = 1.6 * u.micron  # meter
mas_pix = 16.2  # milli arcsec per pixel
diameter = 8.2 * 0.95 * u.meter  # meter (dimeter * clear aperture)

# lambda / diameter
ld = wavelength / diameter  # radians
# ld = np.degrees(ld) * 3600 * 1000 # milli arcsec

oversampling_factor = 1

# pixels size in mas
rad_pix = np.radians(mas_pix / 1000 / 3600) / oversampling_factor

# number of pixels along one axis in the pupil and focal planes
Npix_pup = aperture.shape[0]
Npix_foc = 128 * oversampling_factor

# ----------------------------------------------------------------------
# setting grids, mode basis, propagators, etc
# ----------------------------------------------------------------------
# rotating the aperture
rotation_angle = -7
image_center = (aperture.shape[1] / 2, aperture.shape[0] / 2)
rot_mat = cv2.getRotationMatrix2D(image_center, rotation_angle, 1.0)
aperture = cv2.warpAffine(aperture, rot_mat, tuple(aperture.shape.reverse()))
threshold = 0.8
aperture[aperture < threshold] = 0
aperture[aperture >= threshold] = 1

# generating the grids
pupil_grid = hp.make_pupil_grid(Npix_pup, diameter=diameter)
focal_grid = hp.make_uniform_grid(
    [Npix_foc, Npix_foc], [(Npix_foc) * rad_pix, (Npix_foc) * rad_pix]
)


# generating the propagator
propagator = hp.FraunhoferPropagator(pupil_grid, focal_grid)

# generating Lyot stop using the erosion of the aperture
kernel = np.ones((10, 10), np.uint8)
cent_obs = hp.make_obstructed_circular_aperture(diameter * 0.85, 0.5)(pupil_grid)
lyot_mask = cv2.erode(aperture, kernel).ravel() * cent_obs

# rotating the aperture to the correct rotation and making it a field
aperture = hp.Field(aperture.ravel(), pupil_grid)

# fourier transform of aperture
wavefront = hp.Wavefront(aperture, wavelength=wavelength)
E_ref = propagator(wavefront)

img_ref = E_ref

# ----------------------------------------------------------------------
# Coronagraph part
# ----------------------------------------------------------------------
# generating the focal plane mask
fpm_size = 217  # mas
fpm_rad = np.radians(fpm_size / 1000 / 3600)  # mas -> rad
focal_plane_mask = hp.circular_aperture(fpm_rad)(focal_grid)
focal_plane_mask = np.abs(focal_plane_mask - 1)
lyot_stop = hp.Apodizer(lyot_mask)

# Choose the type of Coronagraph
coro = hp.LyotCoronagraph(
    pupil_grid, focal_plane_mask=focal_plane_mask, lyot_stop=lyot_stop
)

post_lyot_plane = coro(wavefront)

img = propagator(post_lyot_plane)

## plotting
fig, axes = pro.subplots(nrows=2, ncols=4, share=0)

# wavefront
m = hp.imshow_field(wavefront.amplitude, ax=axes[0, 0])
axes[0, 0].colorbar(m, loc="t")
axes[0, 0].format(title="wavefront amplitude", xlabel="x [m]", ylabel="y [m]")
m = hp.imshow_field(wavefront.phase, ax=axes[0, 1], vmin=-np.pi, vmax=np.pi)
axes[0, 1].format(title="wavefront phase", xlabel="x [m]", ylabel="y [m]")
axes[0, 1].colorbar(m, loc="t")

# aperture and unobstructed PSF
hp.imshow_field(aperture, ax=axes[1, 0], cmap="gray")
axes[1, 0].format(title=f"aperture ({diameter} m)", xlabel="x [m]", ylabel="y [m]")
psf_norm = np.log10(img_ref.intensity / img_ref.intensity.max())
m = hp.imshow_field(psf_norm, vmin=-5, ax=axes[1, 1])
axes[1, 1].format(title="unobstructed PSF", xlabel="$\lambda$/D", ylabel="$\lambda$/D")
axes[1, 1].colorbar(m, loc="b")

# coronagraph mask and stop
hp.imshow_field(lyot_mask, ax=axes[0, 2], cmap="gray")
axes[0, 2].format(
    title=f"Lyot mask ({fpm_rad / ld:.1f} $\lambda$/D)",
    xlabel="$\lambda$/D",
    ylabel="$\lambda$/D",
)
hp.imshow_field(lyot_stop, ax=axes[0, 3], cmap="gray")
axes[0, 3].format(
    title=f"Lyot stop ({fpm_rad / ld:.1f} m)", xlabel="x [m]", ylabel="y [m]"
)

# post-coronagraphic focal plane
m = hp.imshow_field(post_lyot_plane.intensity, ax=axes[1, 2])
axes[1, 2].format(title="Lyot plane", xlabel="x [m]", ylabel="y [m]")
axes[1, 2].colorbar(m, loc="b")
contrast = np.log10(img.intensity / img_ref.intensity.max())
m = hp.imshow_field(contrast, ax=axes[1, 3], vmin=-6, vmax=0)
axes[1, 3].format(title="raw contrast", xlabel="$\lambda$/D", ylabel="$\lambda$/D")
axes[1, 3].colorbar(m, loc="b")
fig.save(figuredir("lyot.pdf"))
