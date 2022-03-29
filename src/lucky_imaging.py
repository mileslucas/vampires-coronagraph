
import hcipy as hp
import numpy as np
import proplot as pro
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

    tip_tilt_list = []
    psf_images = []
    for i in tqdm.trange(N):
        tip_tilt = tip_tilt_dist.rvs(2)
        tip_tilt_list.append(tip_tilt)
        psf_image = 0
        for wave in wavelengths:
            wavefront = make_wavefront(
                pupil_grid, wave, np.radians(np.array(tip_tilt) / 36e5)
            )
            psf_ref = propagator(wavefront)
            psf_image += psf_ref.intensity
        noisy_image = add_noise(psf_image)
        psf_images.append(noisy_image)

    psf_images = np.asarray(psf_images)
    tip_tilt_list = np.asarray(tip_tilt_list)
    tip_tilt_rms = np.hypot(tip_tilt_list[:, 0], tip_tilt_list[:, 1])

    cutoff = np.quantile(tip_tilt_rms, 1 - q)
    cutoff_inds = tip_tilt_rms < cutoff
    best_ind = np.argmin(tip_tilt_rms)
    print(f"best rms: {tip_tilt_rms.min()}")
    print(f"10th percentile: {cutoff}")
    print(f"median: {np.median(tip_tilt_rms)}")
    smeared = np.mean(psf_images, axis=0)
    single = psf_images[best_ind]
    many = np.mean(psf_images[cutoff_inds], axis=0)
    return smeared, single, many

def plot(smeared, single, many):
    pupil_grid, focal_grid = create_grids(750e-9)
    fig, axes = pro.subplots(ncols=2, refwidth="2in")
    fpm_units = 1 / (np.degrees(VAMPIRES_FOCAL_PLATE_SCALE) * 3600)
    smeared_norm = np.log10(smeared / many.max())
    m = hp.imshow_field(
        smeared_norm,
        focal_grid,
        vmin=-5,
        grid_units=fpm_units,
        ax=axes[0],
        cmap="magma"
    )
    many_norm = np.log10(many / many.max())
    m = hp.imshow_field(
        many_norm,
        focal_grid,
        vmin=-5,
        grid_units=fpm_units,
        ax=axes[1],
        cmap="magma"
    )
    [ax.format(title=t) for ax, t in zip(axes, ["5s exposure","top 10% 5ms exposures"])]
    axes.format(
        abc=True,
        abcbbox=True,
        abcloc="ul",
        grid=False,
        xlabel="x [arcsec]",
        ylabel="y [arcsec]",
        xlim=(-0.1, 0.1),
        ylim=(-0.1, 0.1),
    )
    fig.colorbar(m, loc="r", label="log10(norm. intensity)")
    fig.tight_layout()
    fig.savefig(figuredir("lucky_imaging.pdf"))
    pro.close(fig)

if __name__ == "__main__":
    smeared, single, many = make_images()
    plot(smeared, single, many)