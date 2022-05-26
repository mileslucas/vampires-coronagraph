from pathlib import Path

from astropy.io import fits
import hcipy as hp
import numpy as np
import tqdm.auto as tqdm

rootdir = Path(__file__).parent.parent
datadir = rootdir / "data"

# load aperture and lyot stop pupils
entrance_pupil = fits.getdata(datadir / "scexao_pupil.fits")
lyot_stop_pupil = fits.getdata(datadir / "lyot_stop_pupil.fits")
reference_wavelength = 750e-9

# define grids and propagator
beam_width = 7.032e-3
focal_length = 200e-3
Δx = reference_wavelength * focal_length / beam_width
transmission = 1e-3
# oversample factor
q = 4

pupil_grid = hp.make_pupil_grid(entrance_pupil.shape, diameter=beam_width)
aperture = hp.Field(entrance_pupil.ravel(), pupil_grid)

focal_grid = hp.make_focal_grid(q, num_airy=80, spatial_resolution=Δx)
lyot_stop = hp.Field(lyot_stop_pupil.ravel(), pupil_grid)

propagator = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

for fpm_size_ld in tqdm.tqdm((2, 3, 5, 7), desc="FPM size"):
    fpm_size_rad = fpm_size_ld * (reference_wavelength / beam_width)
    fpm_size = fpm_size_rad * focal_length
    # generate small circular hole
    focal_plane_mask = hp.circular_aperture(fpm_size * 2)(focal_grid)
    # invert hole to become obscuration
    fpm = hp.Field(np.where(focal_plane_mask != 0, transmission, 1), focal_grid)

    coronagraph = hp.LyotCoronagraph(pupil_grid, fpm, lyot_stop)

    wavelengths = np.linspace(725e-9, 775e-9, 50)

    unocculted_psf = 0
    occulted_psf = 0
    for wave in tqdm.tqdm(wavelengths, leave=False):
        wavefront = hp.Wavefront(aperture, wavelength=wave)
        ref_psf = propagator(wavefront)
        post_lyot_plane = coronagraph(wavefront)
        psf = propagator(post_lyot_plane)

        unocculted_psf += ref_psf.intensity.shaped
        occulted_psf += psf.intensity.shaped

    broadband_unocculted_psf = unocculted_psf.astype("f4")
    broadband_occulted_psf = occulted_psf.astype("f4")
    np.savez(datadir / f"fpm-{fpm_size_ld}_data.npz", psf=broadband_unocculted_psf, occulted_psf=broadband_occulted_psf)
