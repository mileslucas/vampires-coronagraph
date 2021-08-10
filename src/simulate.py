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
import tqdm
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


def print_params(fpm_size_ld, wavelength, lyot_type, **lyot_params):
    lines = [
        f"wavelengths: {wavelength*1e9:.0f}+-{BANDWIDTH*1e9/2:.0f} nm",
        f"fpm size [lam/D]: {fpm_size_ld:d}",
        f"fpm size [mas]: {np.degrees(fpm_size_ld * wavelength / APERTURE_DIAMETER) * 36e5:.0f}",
    ]
    if lyot_type == "mirror":
        lines.extend(
            [
                "mirror Lyot stop",
                f"outer scale [D]: {lyot_params['outer_scale']:.2f}",
            ]
        )
    elif lyot_type == "glass":
        lines.extend(
            [
                "glass Lyot stop",
                f"outer scale [D]: {lyot_params['outer_scale']:.2f}",
                f"inner scale [D]: {lyot_params['inner_scale']:.2f}",
                f"erosion size: {lyot_params['erosion_size']:d}",
            ]
        )
    logger.info("\n".join(lines))


def format_filename(fpm_size_ld, wavelength, lyot_type, **lyot_params):
    tokens = [
        f"wave-{wavelength*1e9:.0f}nm",
        f"fpm-{fpm_size_ld:.1f}",
    ]
    if lyot_type == "mirror":
        tokens.extend(["mirror", f"outer-{lyot_params['outer_scale']:.2f}"])
    elif lyot_type == "glass":
        tokens.extend(
            [
                "glass",
                f"outer-{lyot_params['outer_scale']:.2f}",
                f"inner-{lyot_params['inner_scale']:.2f}",
                f"erosion-{lyot_params['erosion_size']:02d}",
            ]
        )
    return "_".join(tokens)


def simulate(
    tip_tilt_rms=3.6,
    fpm_size_ld=2,
    wavelength=750e-9,
    lyot_type="mirror",
    N_tiptilts=1000,
    N_wavelengths=10,
    **lyot_params,
):
    tip_tilt_dist = norm(scale=tip_tilt_rms)
    fpm_size = fpm_size_ld * wavelength / APERTURE_DIAMETER

    print_params(fpm_size_ld, wavelength, lyot_type, **lyot_params)
    filename = format_filename(fpm_size_ld, wavelength, lyot_type, **lyot_params)

    # define grids and propagator
    pupil_grid, focal_grid = create_grids(775e-9)
    propagator = hp.FraunhoferPropagator(
        pupil_grid, focal_grid, focal_length=FOCAL_LENGTH
    )
    nbins = 18
    bin_size = focal_grid.x.max() / nbins

    # create optical elements
    if lyot_type == "mirror":
        lyot_stop = make_lyot_stop(pupil_grid, **lyot_params)
    elif lyot_type == "glass":
        lyot_stop = eroded_lyot_stop(pupil_grid, **lyot_params)
    else:
        raise ValueError(f"invalid Lyot stop type {lyot_type}")
    coronagraph = make_coronagraph(pupil_grid, focal_grid, fpm_size, lyot_stop)

    wavelengths = np.linspace(725e-9, 775e-9, N_wavelengths)

    original_psf = 0
    original_mask_psf = 0
    original_lyot = 0
    original_post_lyot = 0
    original_img = 0
    for wave in tqdm.tqdm(wavelengths, leave=False):
        wavefront = make_wavefront(pupil_grid, wave, (0, 0))
        orig_psf = propagator(wavefront)
        masked_psf = coronagraph.focal_plane_mask(orig_psf)
        lyot_plane = propagator.backward(masked_psf)
        # post_lyot_plane = lyot_stop(lyot_plane)
        post_lyot_plane = coronagraph(wavefront)
        orig_img = propagator(post_lyot_plane)

        original_psf += orig_psf.intensity
        original_mask_psf += masked_psf.intensity
        original_lyot += lyot_plane.intensity
        original_post_lyot += post_lyot_plane.intensity
        original_img += orig_img.intensity

    radii, original_psf_mean, original_coro_mean, rmax = plot_lyot_mosaic(
        wavefront.intensity,
        wavefront.phase,
        coronagraph.focal_plane_mask.apodization,
        original_psf,
        original_mask_psf,
        lyot_stop.apodization,
        lyot_plane.intensity,
        original_post_lyot,
        original_img,
        fpm_size=np.degrees(fpm_size) * 3600,
        bin_size=bin_size,
        savename=filename + "_mosaic.pdf",
    )

    tip_tilt_list = []
    weights = []
    mean_psf = 0
    mean_img = 0
    psf_curves = []
    img_curves = []

    pbar = tqdm.trange(N_tiptilts)
    for i in pbar:
        tip_tilt = tip_tilt_dist.rvs(2)
        tip_tilt_list.append(tip_tilt)
        weight = tip_tilt_dist.pdf(tip_tilt).prod()
        pbar.set_description_str(
            f"tip-tilt error ({tip_tilt[0]: 02.2f}, {tip_tilt[1]: 02.2f}) [mas]"
        )

        # pre-allocate outputs
        # wavefront_image = 0
        # wavefront_phase = 0
        psf_image = 0
        # masked_psf_image = 0
        # lyot_image = 0
        # post_lyot_image = 0
        output_image = 0

        desc = f"Simulating broadband image from {len(wavelengths)} samples of {wavelengths.min()*1e9:.2f} nm - {wavelengths.max()*1e9:.2f} nm"
        for wave in tqdm.tqdm(wavelengths, desc, leave=False):
            wavefront = make_wavefront(
                pupil_grid, wave, np.radians(np.array(tip_tilt) / 36e5)
            )

            # Fourier propagation
            psf_ref = propagator(wavefront)

            # masked_psf = coronagraph.focal_plane_mask(psf_ref)
            # lyot_plane = propagator.backward(masked_psf)
            # post_lyot_plane = lyot_stop(lyot_plane)
            # psf = propagator(post_lyot_plane)

            post_lyot_plane = coronagraph(wavefront)
            psf = propagator(post_lyot_plane)

            # wavefront_image += wavefront.intensity / len(wavelengths)
            # wavefront_phase += wavefront.phase / len(wavelengths)
            psf_image += psf_ref.intensity
            # masked_psf_image += masked_psf.intensity
            # lyot_image += lyot_plane.intensity
            # post_lyot_image += post_lyot_plane.intensity
            output_image += psf.intensity
        bins, psf_curve, _, _ = hp.radial_profile(psf_image, bin_size)
        _, img_curve, _, _ = hp.radial_profile(output_image, bin_size)
        radii = bins * np.degrees(OPTICAL_PLATE_SCALE) * 3600
        mean_psf += psf_image * weight
        mean_img += output_image * weight
        weights.append(weight)
        psf_curves.append(psf_curve)
        img_curves.append(img_curve)
    pbar.close()

    tip_tilts = np.array(tip_tilt_list)
    weights = np.array(weights)
    psf_curves = np.array(psf_curves)
    img_curves = np.array(img_curves)
    mean_psf /= len(tip_tilts) * weights.sum()
    mean_img /= len(tip_tilts) * weights.sum()
    np.savez(
        datadir(filename + "_data.npz"),
        tip_tilts=tip_tilts,
        radii=radii,
        mean_psf=mean_psf,
        mean_image=mean_img,
        psf_curves=psf_curves,
        img_curves=img_curves,
        weights=weights,
        original_psf_curve=original_psf_mean,
        original_coro_curve=original_coro_mean,
    )
    plot_attenuation_curves(
        radii,
        original_psf_mean,
        original_coro_mean,
        psf_curves,
        img_curves,
        weights,
        mean_psf,
        mean_img,
        fpm_size=np.degrees(fpm_size) * 3600,
        rmax=rmax,
        savename=filename + "_attenuation_curve.pdf",
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
    outer_scales = [0.99, 0.95, 0.9, 0.85, 0.8]
    inner_scales = [0.31, 0.35, 0.4, 0.45, 0.5]
    erosion_sizes = list(filter(lambda x: x % 2 == 1, range(3, 16)))
    for scale_idx in tqdm.trange(len(outer_scales), desc="scales"):
        for erosion_size in tqdm.tqdm(erosion_sizes, desc="erosion size"):
            simulate(
                tip_tilt_rms=7.0,
                fpm_size_ld=3,
                wavelength=750e-9,
                lyot_type="glass",
                outer_scale=outer_scales[scale_idx],
                inner_scale=inner_scales[scale_idx],
                erosion_size=erosion_size,
                N_tiptilts=1000,
            )
