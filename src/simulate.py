# Take an input PSD of wavefront errors and produce a Lyot coronagraph
# with the given parameters. Many PSDs are simulated and averaged together to
# measure the contrast. Each report is logged alongside plots.
from PyPDF2 import PdfFileMerger
from itertools import product
import hcipy as hp
import logging
import numpy as np
from scipy import ndimage
from scipy.stats import norm
from tqdm import tqdm
import os

from config import *
from lyot import (
    create_grids,
    make_coronagraph,
    make_wavefront,
    make_lyot_stop,
    eroded_lyot_stop,
)
from plotting import plot_lyot_mosaic, plot_attenuation_curves
from paths import figuredir, datadir

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def simulate():
    tip_tilt_rms = 2  # lam/D
    tip_tilt_dist = norm(scale=tip_tilt_rms)
    fpm_size_ld = 5
    fpm_size = fpm_size_ld * LAMBDA_D
    outer_scale = 0.95

    logger.info(f"fpm size [lam/D]: {fpm_size_ld:d}")
    logger.info(f"lyot stop scale: {outer_scale}")

    # define grids and propagator
    pupil_grid, focal_grid = create_grids(MAX_WAVELENGTH)
    propagator = hp.FraunhoferPropagator(
        pupil_grid, focal_grid, focal_length=FOCAL_LENGTH
    )

    # create optical elements
    coronagraph = make_coronagraph(pupil_grid, focal_grid, fpm_size)
    # lyot_stop = make_lyot_stop(pupil_grid, outer_scale)
    lyot_stop = eroded_lyot_stop(pupil_grid, outer_scale=outer_scale)

    wavelengths = np.linspace(725e-9, 775e-9, 10)

    original_psf = 0
    original_coro = 0
    for wave in tqdm(wavelengths, leave=False):
        wavefront = make_wavefront(pupil_grid, wave, (0, 0))
        orig_psf = propagator(wavefront)
        masked_psf = coronagraph.focal_plane_mask(orig_psf)
        lyot_plane = propagator.backward(masked_psf)
        post_lyot_plane = lyot_stop(lyot_plane)
        orig_coro = propagator(post_lyot_plane)
        original_psf += orig_psf.intensity
        original_coro += orig_coro.intensity
    
    mean_psf = original_psf.copy()
    mean_img = original_coro.copy()
    tip_tilt_list = [np.array([0, 0])]
    weights = [tip_tilt_dist.pdf(tip_tilt_list[0]).prod()]
    bin_size = LAMBDA_D / OPTICAL_PLATE_SCALE / 2
    psf_curves = [hp.radial_profile(original_psf, bin_size)[1]]
    img_curves = [hp.radial_profile(original_coro, bin_size)[1]]
    names = []
    rmax = 0


    nbranches = 12
    nradii = 9
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / nbranches)
    rs = np.linspace(0, 2 * tip_tilt_rms, nradii + 1)[1:]
    coords = product(rs, angles)
    pbar = tqdm(enumerate(coords), total=nradii * nbranches)
    for i, (r, ang) in pbar:
        tip_tilt = np.array([r * np.cos(ang), r * np.sin(ang)])
        tip_tilt_list.append(tip_tilt)
        weight = tip_tilt_dist.pdf(tip_tilt).prod()
        pbar.set_description_str(
            f"tip-tilt error ({tip_tilt[0]:01.2f}, {tip_tilt[1]:01.2f}) [lam/D]"
        )

        # pre-allocate outputs
        wavefront_image = 0
        wavefront_phase = 0
        psf_image = 0
        masked_psf_image = 0
        lyot_image = 0
        post_lyot_image = 0
        output_image = 0

        desc = f"Simulating broadband image from {len(wavelengths)} samples of {wavelengths.min()*1e9:.2f} nm - {wavelengths.max()*1e9:.2f} nm"
        for wave in tqdm(wavelengths, desc, leave=False):
            wavefront = make_wavefront(pupil_grid, wave, np.array(tip_tilt) * LAMBDA_D)

            # Fourier propagation
            psf_ref = propagator(wavefront)
            masked_psf = coronagraph.focal_plane_mask(psf_ref)
            lyot_plane = propagator.backward(masked_psf)
            post_lyot_plane = lyot_stop(lyot_plane)
            psf = propagator(post_lyot_plane)

            wavefront_image += wavefront.intensity / len(wavelengths)
            wavefront_phase += wavefront.phase / len(wavelengths)
            psf_image += psf_ref.intensity
            masked_psf_image += masked_psf.intensity
            lyot_image += lyot_plane.intensity
            post_lyot_image += post_lyot_plane.intensity
            output_image += psf.intensity
        fname = f"lyot_{i:02d}.pdf"
        radii, psf_curve, img_curve, rmax = plot_lyot_mosaic(
            wavefront_image,
            wavefront_phase,
            coronagraph.focal_plane_mask.apodization,
            masked_psf_image,
            lyot_stop.apodization,
            lyot_image,
            post_lyot_image,
            psf_image,
            output_image,
            original_psf,
            fpm_size=np.degrees(fpm_size) * 3600,
            savename=fname,
        )
        mean_psf += psf_image * weight
        mean_img += output_image * weight
        weights.append(weight)
        psf_curves.append(psf_curve)
        img_curves.append(img_curve)
        names.append(fname)

    pbar.close()

    merge_pdfs(names, "lyot_eroded_stop=0.9.pdf")

    tip_tilts = np.array(tip_tilt_list)
    weights = np.array(weights)
    psf_curves = np.array(psf_curves)
    img_curves = np.array(img_curves)
    mean_psf /= len(tip_tilts) * weights.sum()
    mean_img /= len(tip_tilts) * weights.sum()
    np.savez(
        datadir("lyot_data.npz"),
        tip_tilts=tip_tilts,
        radii=radii,
        img_curves=img_curves,
        weights=weights
    )
    plot_attenuation_curves(
        radii,
        hp.radial_profile(original_psf, bin_size)[1],
        hp.radial_profile(original_coro, bin_size)[1],
        psf_curves,
        img_curves,
        weights,
        mean_psf,
        mean_img,
        fpm_size=np.degrees(fpm_size) * 3600,
        rmax=rmax,
        savename="attenuation_curve_eroded_stop=0.9.pdf",
    )

def merge_pdfs(names, outname="lyot.pdf"):
    merger = PdfFileMerger()
    [merger.append(figuredir(name)) for name in names]
    with open(figuredir(outname), "wb") as fh:
        merger.write(fh)

    for name in names:
        os.remove(figuredir(name))

    return figuredir(outname)


if __name__ == "__main__":
    simulate()
