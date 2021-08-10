import os
import hcipy as hp
import logging
import numpy as np
import proplot as pro

from paths import figuredir
from config import *

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ----------------------------------------------------------------------
# Plotting defaults
# ----------------------------------------------------------------------
pro.use_style("ggplot")
pro.rc["image.cmap"] = "inferno"
pro.rc["grid"] = False


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_lyot_mosaic(
    wavefront,
    wavefront_phase,
    focal_plane_mask,
    psf_ref,
    masked_psf,
    lyot_stop,
    lyot_plane,
    post_lyot_plane,
    psf,
    bin_size,
    savename="lyot.pdf",
    fpm_size=None,
):
    layout = [[3, 4, 5, 6, 7], [1, 10, 10, 10, 8], [2, 10, 10, 10, 9]]
    fig, axes = pro.subplots(layout, share=0, width="15in")
    axes.format(abc=True, titlesize=11)

    # wavefront
    m = hp.imshow_field(wavefront, ax=axes[0])
    axes[0].colorbar(m, loc="l")
    axes[0].format(title="wavefront intensity", xlabel="x [m]", ylabel="y [m]")

    m = hp.imshow_field(wavefront_phase * (wavefront > 0), ax=axes[1], cmap="Vlag")
    axes[1].format(title="wavefront phase", xlabel="x [m]", ylabel="y [m]")
    axes[1].colorbar(m, loc="l", label="rad")

    # unobstructed PSF & contrast
    psf_norm = np.log10(psf_ref / psf_ref.max())
    m = hp.imshow_field(psf_norm, vmin=-8, ax=axes[2])
    focal_ticks = axes[2].get_xticks()
    focal_ticklabs = [
        f"{v:.1f}" for v in np.degrees(focal_ticks * OPTICAL_PLATE_SCALE) * 3600
    ]
    axes[2].format(
        title="unobstructed PSF",
        xlabel="arcsec",
        ylabel="arcsec",
        xticks=focal_ticks,
        xticklabels=focal_ticklabs,
        yticks=focal_ticks,
        yticklabels=focal_ticklabs,
    )
    axes[2].colorbar(m, loc="t")

    hp.imshow_field(focal_plane_mask, ax=axes[3], cmap="gray")
    axes[3].format(
        title=f"focal plane mask\n({np.radians(fpm_size / 3600) / LAMBDA_D:.0f} $\lambda$/D; {fpm_size * 1e3:.0f} mas)",
        xlabel="arcsec",
        ylabel="arcsec",
        xticks=focal_ticks,
        xticklabels=focal_ticklabs,
        yticks=focal_ticks,
        yticklabels=focal_ticklabs,
    )
    axes[3].grid(True, color="k", alpha=0.2)
    with np.errstate(divide="ignore"):
        obs_psf_norm = np.log10(masked_psf / psf_ref.max())
    m = hp.imshow_field(obs_psf_norm, vmin=-8, vmax=0, ax=axes[4], zorder=999)
    axes[4].format(
        title="obstructed PSF",
        xlabel="arcsec",
        ylabel="arcsec",
        xticks=focal_ticks,
        xticklabels=focal_ticklabs,
        yticks=focal_ticks,
        yticklabels=focal_ticklabs,
        grid=True,
    )
    axes[4].colorbar(m, loc="t")

    m = hp.imshow_field(lyot_plane, ax=axes[5])
    axes[5].format(title="Lyot plane", xlabel="x [m]", ylabel="y [m]")
    axes[5].colorbar(m, loc="t")

    hp.imshow_field(lyot_stop, ax=axes[6], cmap="gray")
    axes[6].format(title=f"Lyot stop", xlabel="x [m]", ylabel="y [m]")

    m = hp.imshow_field(post_lyot_plane, vmax=lyot_plane.max(), ax=axes[7])
    axes[7].format(title="post Lyot plane", xlabel="x [m]", ylabel="y [m]")
    axes[7].colorbar(m, loc="r")

    post_psf_norm = np.log10(psf / psf_ref.max())
    m = hp.imshow_field(post_psf_norm, vmin=-8, vmax=0, ax=axes[8])
    axes[8].format(
        title="post-Lyot PSF",
        xlabel="arcsec",
        ylabel="arcsec",
        xticks=focal_ticks,
        xticklabels=focal_ticklabs,
        yticks=focal_ticks,
        yticklabels=focal_ticklabs,
    )
    axes[8].colorbar(m, loc="r")

    rmax = psf_ref.grid.x.max() * np.degrees(OPTICAL_PLATE_SCALE) * 3600
    bins, psf_mean, _, _ = hp.radial_profile(psf_ref, bin_size)
    _, img_mean, _, _ = hp.radial_profile(psf, bin_size)

    radii = bins * np.degrees(OPTICAL_PLATE_SCALE) * 3600
    axes[9].plot(radii, psf_mean / psf_mean.max(), lw=2, label="PSF")
    axes[9].plot(radii, img_mean / psf_mean.max(), lw=2, label="post-coronagraphic")
    axes[9].format(
        title="attenuation curve",
        yscale="log",
        yformatter="log",
        ylabel="image / max(PSF)",
        xlabel="separation [arcsec]",
        xlim=(0, rmax),
        grid=True,
    )
    ylim = axes[9].get_ylim()
    axes[9].vlines(fpm_size, *ylim, color="k", ls="--", alpha=0.5, label="FPM radius")
    axes[9].set_ylim(ylim[0], 1)
    axes[9].legend(ncol=1)

    fig.save(figuredir(savename))
    pro.close(fig)
    logger.debug(f"saved image to {os.path.normpath(figuredir(savename))}")
    return radii, psf_mean, img_mean, rmax


def plot_attenuation_curves(
    radii,
    psf_curve,
    coro_curve,
    psf_curves,
    img_curves,
    weights,
    mean_psf,
    mean_img,
    fpm_size,
    rmax,
    savename="attenuation_curve.pdf",
):
    psfmax = psf_curve.max()
    psf_mean_curve = np.average(psf_curves / psfmax, weights=weights, axis=0)
    img_mean_curve = np.average(img_curves / psfmax, weights=weights, axis=0)
    img_std_curve = np.sqrt(
        np.average((img_curves / psfmax - img_mean_curve) ** 2, weights=weights, axis=0)
    )
    mask = (img_mean_curve - img_std_curve) < 0
    std_hi = img_std_curve
    std_lo = np.where(mask, 0, img_std_curve)

    layout = [[1, 3, 3], [2, 3, 3]]
    fig, axes = pro.subplots(layout, width="10in", height="6in", share=0)

    psf_norm = np.log10(mean_psf / mean_psf.max())
    m = hp.imshow_field(psf_norm, vmin=-8, ax=axes[0])
    focal_ticks = axes[0].get_xticks()
    focal_ticklabs = [
        f"{v:.1f}" for v in np.degrees(focal_ticks * OPTICAL_PLATE_SCALE) * 3600
    ]
    axes[0].format(
        title="unobstructed PSF",
        xlabel="arcsec",
        ylabel="arcsec",
        xticks=focal_ticks,
        xticklabels=focal_ticklabs,
        yticks=focal_ticks,
        yticklabels=focal_ticklabs,
    )

    img_norm = np.log10(mean_img / mean_psf.max())
    m = hp.imshow_field(img_norm, vmin=-8, vmax=0, ax=axes[1])
    axes[1].format(
        title="post-coronagraphic PSF",
        xlabel="arcsec",
        ylabel="arcsec",
        xticks=focal_ticks,
        xticklabels=focal_ticklabs,
        yticks=focal_ticks,
        yticklabels=focal_ticklabs,
    )
    fig.colorbar(m, loc="l", label="log10[image / max(PSF)]")

    axes[2].plot(radii, psf_curve / psfmax, lw=2, c="C3", label="PSF (no jitter)")
    axes[2].plot(radii, psf_mean_curve, lw=2, c="C0", label="PSF (jitter)")
    axes[2].plot(
        radii, coro_curve / psfmax, lw=2, c="C5", label="post-coro. (no jitter)"
    )
    axes[2].plot(
        radii,
        img_mean_curve,
        lw=2,
        c="C1",
        fadedata=(std_lo, std_hi),
        label="post-coro. (jitter)",
    )
    axes[2].format(
        title="attenuation curve",
        yscale="log",
        yformatter="log",
        ylabel="image / max(PSF)",
        xlabel="separation [arcsec]",
        grid=True,
        xlim=(0, rmax),
    )
    ylim = axes[2].get_ylim()
    ylo = img_mean_curve[: np.argmax(radii > rmax)].min() / 3
    axes[2].vlines(fpm_size, *ylim, color="k", ls="--", alpha=0.3, label="FPM radius")
    axes[2].set_ylim(ylo, 1)
    axes[2].legend(ncol=1)

    fig.save(figuredir(savename))
    pro.close(fig)
    logger.info(f"saved image to {os.path.normpath(figuredir(savename))}")

    # return radii, psf_mean_curve, img_mean_curve, img_std_curve, rmax
