import astropy.units as u
import cv2
import hcipy as hp
import numpy as np
import proplot as pro
from astropy.io import fits
from scipy import ndimage
from tqdm import tqdm

from paths import datadir, figuredir

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
# wavelength = 1.6 * u.micron  # meter
wavelength = 7e-7  # meter
mas_pix = 16.2  # milli arcsec per pixel
# diameter = 8.2 * 0.95 * u.meter  # meter (diameter * clear aperture)
diameter = 8.2 * 0.95  # meter (diameter * clear aperture)

# lambda / diameter
ld = wavelength / diameter  # radians
# ld = np.degrees(ld) * 3600 * 1000 # milli arcsec

oversampling_factor = 10

# pixels size in mas
rad_pix = np.radians(mas_pix / 1000 / 3600) / oversampling_factor

# number of pixels along one axis in the pupil and focal planes
Npix_pup = aperture.shape[0]
zoom_factor = 4
npix = 128 / zoom_factor
Npix_foc = npix * oversampling_factor

# ----------------------------------------------------------------------
# setting grids, mode basis, propagators, etc
# ----------------------------------------------------------------------
# rotating the aperture
rotation_angle = -7
aperture = ndimage.rotate(aperture, rotation_angle, reshape=False)
threshold = 0.8
aperture[aperture < threshold] = 0
aperture[aperture >= threshold] = 1

# generating the grids
pupil_grid = hp.make_pupil_grid(Npix_pup, diameter=diameter)
focal_grid = hp.make_uniform_grid(
    [Npix_foc, Npix_foc], [Npix_foc * rad_pix, Npix_foc * rad_pix]
)

# generating the propagator
propagator = hp.FraunhoferPropagator(pupil_grid, focal_grid)

# generating Lyot stop using the erosion of the aperture
cent_obs = hp.make_obstructed_circular_aperture(diameter * 0.8, 0.6)(pupil_grid)
kernel = np.ones((10, 10), np.uint8)
lyot_mask = cv2.erode(aperture, kernel).ravel() * cent_obs

# rotating the aperture to the correct rotation and making it a field
aperture = hp.Field(aperture.ravel(), pupil_grid)

# fourier transform of aperture
wavefront = hp.Wavefront(aperture, wavelength=wavelength)
img_ref = propagator(wavefront)

# ----------------------------------------------------------------------
# Coronagraph part
# ----------------------------------------------------------------------
# generating the focal plane mask
fpm_rad = 5 * ld  # rad
focal_plane_mask = hp.circular_aperture(fpm_rad)(focal_grid)
focal_plane_mask = np.abs(focal_plane_mask - 1)
lyot_stop = hp.Apodizer(lyot_mask)
lyot_reflect = hp.Apodizer(np.abs(lyot_mask - 1))

# Choose the type of Coronagraph
coro = hp.LyotCoronagraph(pupil_grid, focal_plane_mask=focal_plane_mask)

lyot_plane = coro(wavefront)
post_lyot_plane = lyot_stop(lyot_plane)
img = propagator(post_lyot_plane)

reflected = lyot_reflect(lyot_plane)
reformed = propagator(reflected)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
fig, axes = pro.subplots(nrows=2, ncols=4, share=0)

# wavefront
m = hp.imshow_field(wavefront.amplitude, ax=axes[0, 0])
axes[0, 0].colorbar(m, loc="t")
axes[0, 0].format(title="wavefront amplitude", xlabel="x [m]", ylabel="y [m]")
hp.imshow_field(lyot_mask, ax=axes[0, 1], cmap="gray")
axes[0, 1].format(title=f"Lyot stop", xlabel="x [m]", ylabel="y [m]")

# coronagraph mask and stop

m = hp.imshow_field(wavefront.phase, ax=axes[1, 0], vmin=-np.pi, vmax=np.pi)
axes[1, 0].format(title="wavefront phase", xlabel="x [m]", ylabel="y [m]")
axes[1, 0].colorbar(m, loc="b")

hp.imshow_field(focal_plane_mask, ax=axes[1, 1], cmap="gray")
focal_ticks = axes[1, 1].get_xticks()
focal_ticklabs = [f"{v:.1f}" for v in np.degrees(focal_ticks) * 3600]
axes[1, 1].format(
    title=f"Lyot mask ({fpm_rad / ld:.1f} $\lambda$/D)",
    # xlabel="$\lambda$/D",
    # ylabel="$\lambda$/D",
    xlabel="arcsec",
    ylabel="arcsec",
    xticks=focal_ticks,
    xticklabels=focal_ticklabs,
    yticks=focal_ticks,
    yticklabels=focal_ticklabs,
)
axes[1, 1].grid(True, color="k", alpha=0.2)



# post-coronagraphic focal plane
m = hp.imshow_field(lyot_plane.intensity, ax=axes[0, 2])
axes[0, 2].format(title="Lyot plane", xlabel="x [m]", ylabel="y [m]")
axes[0, 2].colorbar(m, loc="t")

m = hp.imshow_field(post_lyot_plane.intensity, ax=axes[0, 3])
axes[0, 3].format(title="post Lyot plane", xlabel="x [m]", ylabel="y [m]")
axes[0, 3].colorbar(m, loc="t")

# unobstructed PSF & contrast
psf_norm = np.log10(img_ref.intensity / img_ref.intensity.max())
m = hp.imshow_field(psf_norm, vmin=-8, ax=axes[1, 2])
axes[1, 2].format(
    title="unobstructed PSF",
    # xlabel="$\lambda$/D",
    # ylabel="$\lambda$/D",
    xlabel="arcsec",
    ylabel="arcsec",
    xticks=focal_ticks,
    xticklabels=focal_ticklabs,
    yticks=focal_ticks,
    yticklabels=focal_ticklabs,
)
axes[1, 2].colorbar(m, loc="b")

contrast = np.log10(img_ref.intensity.max() / img.intensity)
contrast[~focal_plane_mask.astype(bool)] = np.nan
m = hp.imshow_field(contrast, ax=axes[1, 3], cmap="inferno_r", vmax=10, vmin=0)
axes[1, 3].format(
    title="raw contrast",
    # xlabel="$\lambda$/D",
    # ylabel="$\lambda$/D",
    xlabel="arcsec",
    ylabel="arcsec",
    xticks=focal_ticks,
    xticklabels=focal_ticklabs,
    yticks=focal_ticks,
    yticklabels=focal_ticklabs,
)
axes[1, 3].colorbar(m, loc="b", label="log10(max(PSF) / img)")

fig.save(figuredir("lyot.pdf"))

# ----------------------------------------------------------------------
# Contrast
