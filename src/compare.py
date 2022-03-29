import argparse
import hcipy as hp
import numpy as np
import proplot as pro
import logging
import re

from paths import figuredir

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs="+", help="npz files output from simulate.py")
parser.add_argument("-v", "--variable", help="the variable to compare between data")

pro.use_style("ggplot")
pro.rc["image.cmap"] = "inferno"
pro.rc["grid"] = False


def compare_plot(radii, psf_curve, img_curves, img_curves_jitter, labels):
    fig, axes = pro.subplots(width="8in", refaspect=16 / 10)

    cycle = pro.Cycle("viridis", len(img_curves), left=0.1, right=0.9)

    axes.plot(radii, psf_curve, lw=2, c="k", label="PSF (no jitter)")
    for img, jit, lab, color in zip(
        img_curves, img_curves_jitter, labels, cycle.by_key()["color"]
    ):
        axes.plot(radii, img, c=color, label=lab)
        axes.plot(radii, jit, c=color, ls="--")

    axes.format(
        title="attenuation curve",
        yscale="log",
        yformatter="log",
        ylabel="image / max(PSF)",
        xlabel="separation [arcsec]",
        grid=True,
        xlim=(0, 0.30703899893336833),
    )
    # ylim = axes.get_ylim()
    # ylo = img_mean_curve[:np.argmax(radii > rmax)].min() / 3
    # axes.vlines(fpm_size, *ylim, color="k", ls="--", alpha=0.3, label="FPM radius")
    # axes.set_ylim(ylo, 1)
    axes.legend(ncol=3)

    fig.save(figuredir("compare_erosion_outer-0.99_inner-0.31.pdf"))
    pro.close(fig)
    # logger.info(f"saved image to {os.path.normpath(figuredir(savename))}")


def parse_label(filename, variable):
    m = re.search(f"_{variable}-(.+)_", filename)
    return f"{variable} {m[1]}"


if __name__ == "__main__":
    args = parser.parse_args()

    psf_curve = 0
    labels = []
    img_curves = []
    img_curves_jitter = []
    for filename in sorted(args.filenames):
        data = np.load(filename)
        psf = data["original_psf_curve"]
        psfmax = psf.max()
        psf_curve += psf / psfmax / len(args.filenames)
        img_curve = data["original_coro_curve"] / psfmax
        img_curve_jitter = data["img_curves"] / psfmax
        weights = data["weights"]
        radii = data["radii"]
        img_curve_jitter = np.average(img_curve_jitter, weights=weights, axis=0)
        img_curves.append(img_curve)
        img_curves_jitter.append(img_curve_jitter)
        labels.append(parse_label(filename, args.variable))

    compare_plot(radii, psf_curve, img_curves, img_curves_jitter, labels)
