import cv2
import hcipy as hp
import numpy as np

from config import *


# ----------------------------------------------------------------------
# setting grids, mode basis, propagators, etc
# ----------------------------------------------------------------------


def create_grids(wavelength):
    # generating the grids
    pupil_grid = hp.make_pupil_grid(APERTURE.shape, diameter=APERTURE_DIAMETER)
    Δx = wavelength * F_RATIO
    # q = Δx / VAMPIRES_PIXEL_PITCH
    q = 4
    focal_grid = hp.make_focal_grid(q, num_airy=16, spatial_resolution=Δx)
    return pupil_grid, focal_grid


def eroded_lyot_stop(pupil_grid, outer_scale, inner_scale, erosion_size):
    """
    Lyot stop with design generated from Subaru aperture using an erosion kernel

    Parameters
    ----------
    outer_scale : float, optional
        The outer scale of the circular stop, in units of diameter. Default is 0.8.
    inner_scale : float, optional
        The inner scale of the circular stop, in units of diameter. Default is 0.5.
    erosion_size : int, optional
        The size of the erosion kernel. Default is 10.

    Returns
    -------
    lyot_mask : ndarray
    """
    cent_obs = hp.make_obstructed_circular_aperture(
        APERTURE_DIAMETER * outer_scale, inner_scale
    )(pupil_grid)
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    stop_mask = cv2.erode(APERTURE.astype("f8"), kernel).ravel() * cent_obs
    return hp.Apodizer(stop_mask)


def make_lyot_stop(pupil_grid, outer_scale):
    """
    Classic Lyot stop with empty hole

    Parameters
    ----------
    outer_scale : float, optional
        The outer scale of the circular stop, in units of diameter. Default is 0.8.

    Returns
    -------
    lyot_mask : ndarray
    """
    stop_mask = hp.circular_aperture(outer_scale * APERTURE_DIAMETER)(pupil_grid)
    return hp.Apodizer(stop_mask)


# ----------------------------------------------------------------------
# Coronagraph part
# ----------------------------------------------------------------------


def focal_plane_mask(focal_grid, size):
    """
    Generate focal plane mask using a uniform disk

    Parameters
    ----------
    size : float
        Angular size of the focal plane mask in radians

    Returns
    -------
    focal_plane_mask : ndarray
    """
    # convert size to meters
    size *= 2 * FOCAL_LENGTH
    # generate small circular hole
    focal_plane_mask = hp.evaluate_supersampled(
        hp.circular_aperture(size), focal_grid, 4
    )
    # invert hole to become obscuration
    return np.abs(focal_plane_mask - 1)


def make_coronagraph(pupil_grid, focal_grid, fpm_size, lyot_stop=None):
    fpm = focal_plane_mask(focal_grid, fpm_size)
    size = fpm_size * 2 * FOCAL_LENGTH
    coronagraph = hp.LyotCoronagraph(
        pupil_grid, fpm, lyot_stop=lyot_stop, focal_length=FOCAL_LENGTH
    )
    return coronagraph


def make_wavefront(pupil_grid, wavelength, tip_tilt):
    # fourier transform of aperture
    phase = (
        (tip_tilt[0] * pupil_grid.x + tip_tilt[1] * pupil_grid.y)
        * 2
        * np.pi
        / wavelength
    )
    ap = hp.Field(APERTURE, pupil_grid).ravel() * np.exp(1j * phase)
    wavefront = hp.Wavefront(ap, wavelength=wavelength)
    return wavefront
