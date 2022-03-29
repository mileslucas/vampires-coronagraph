
import hcipy as hp
import numpy as np
import proplot as pro
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
import tqdm

from config import *
from lyot import (
    create_grids,
    make_wavefront,
)
from paths import figuredir, datadir

def add_noise(image):
    noise = poisson(image)
    return noise.rvs()

def make_images(wavelength=750e-9, rms=8, N=1000, q=0.9):
    pupil_grid, focal_grid = create_grids(wavelength)
    tip_tilt_dist = norm(scale=rms)
    propagator = hp.FraunhoferPropagator(
        pupil_grid, focal_grid, focal_length=EFF_FOCAL_LENGTH
    )
    wavelengths = np.linspace(725e-9, 775e-9, 10)
    original_psf = 0
    for wave in tqdm.tqdm(wavelengths, leave=False):
        wavefront = make_wavefront(pupil_grid, wave, (0, 0))
        orig_psf = propagator(wavefront)

        original_psf += orig_psf.intensity
    return make_wavefront(pupil_grid, wavelength, (0, 0)), original_psf

def plot(pupil, psf):
    pupil_grid, focal_grid = create_grids(750e-9)
    fig, axes = pro.subplots(ncols=2, share=0, refwidth="2in")
    hp.imshow_field(
        pupil.intensity,
        pupil_grid,
        vmin=0,
        vmax=1,
        ax=axes[0],
        cmap="gray"
    )
    axes[0].format(
        xlabel="x [m]",
        ylabel="y [m]",
    )

    fpm_units = 1 / (np.degrees(VAMPIRES_FOCAL_PLATE_SCALE) * 3600)
    psf_norm = np.log10(psf / psf.max())
    m = hp.imshow_field(
        psf_norm,
        focal_grid,
        vmin=-8,
        grid_units=fpm_units,
        ax=axes[1],
        cmap="magma"
    )
    axes[1].format(
        xlabel="x [arcsec]",
        ylabel="y [arcsec]",
    )
    # add circles
    radii = np.array([2, 3, 5, 7])
    radii_arcsec = np.rad2deg(radii * 750e-9/(8.2 * 0.95)) * 3600
    colors = ["1.0", "0.95", "0.8", "0.7"]
    for r, c in zip(radii_arcsec, colors):
        patch = plt.Circle((fpm_units, fpm_units), r, fill=False, color=c, lw=1)
        axes[1].add_patch(patch)

    axes.format(
        abc=True,
        abcbbox=True,
        abcloc="ul",
        grid=False,
    )
    fig.colorbar(m, loc="r", label="log10(norm. intensity)")
    fig.savefig(figuredir("aperture_psf.pdf"))
    pro.close(fig)

if __name__ == "__main__":
    imgs = make_images()
    plot(*imgs)