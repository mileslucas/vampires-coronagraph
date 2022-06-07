import numpy as np
from astropy.io import fits
from astropy.modeling.models import Moffat2D
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from scipy.optimize import curve_fit
from pathlib import Path


datadir = Path("/Volumes/mlucas SSD1/vampires-coronagraph-data/bench_20220526")
figdir = Path(__file__) / ".." / "paper" /"figures"
procdir = datadir / "processed"
procdir.mkdir(exist_ok=True)

# make dark frame
dark_filename = datadir / "darks_em0_20ms_750-50_Mirror_0_cam1.fits"
dark_cube, dark_hdr = fits.getdata(dark_filename, header=True)
dark_frame = np.median(dark_cube, axis=0)
fits.writeto(procdir / "master_dark_em0_20ms.fits", dark_frame, header=dark_hdr, overwrite=True)

# calibrate
sci_filename = datadir / "Open_ag50nm-b2_750-50_EmptySlot2_0_cam1.fits"
sci_cube, sci_hdr = fits.getdata(sci_filename, header=True)
calib_cube = np.flip(sci_cube - dark_frame, axis=-2)
calib_frame = np.median(calib_cube, axis=0)
fits.writeto(procdir / "Open_ag50nm-b2_750-50_EmptySlot2_0_cam1_calib_collapsed.fits", calib_frame, header=sci_hdr, overwrite=True)

# find de-facto center and estimate sat spots
ctr = np.unravel_index(calib_frame.argmax(), calib_frame.shape)
sep = 15.5 * np.rad2deg(750e-9 / 7.79) * 3.6e6 / 6.24 # 15.5 lam/d
theta0 = np.deg2rad(-4)
angles = (np.linspace(0, 2*np.pi, 4, endpoint=False) + theta0) % (2 * np.pi)
cy = sep * np.sin(angles) + ctr[0]
cx = sep * np.cos(angles) + ctr[1]
# add center psf
cy = np.append(cy, ctr[0])
cx = np.append(cx, ctr[1])

# get 30x30 window cutouts
window_size = 30
half_width = window_size // 2
ly = np.floor(cy - half_width)
uy = np.floor(cy + half_width)
lx = np.floor(cx - half_width)
ux = np.floor(cx + half_width)
ys = np.concatenate([np.arange(l, u, dtype=int) for l, u in zip(ly, uy)])
xs = np.concatenate([np.arange(l, u, dtype=int) for l, u in zip(lx, ux)])
# X = np.vstack((xs, ys))
ys, xs = np.mgrid[:512, :512]
X = np.vstack((xs.ravel(), ys.ravel()))

# create hierarchical moffat model
def model(X, x0, y0, sep, t0, amp, gamma, alpha, contrast):
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False) + t0
    cy = sep * np.sin(angles) + y0
    cx = sep * np.cos(angles) + x0
    satamp = amp * 10**contrast
    cal_right = Moffat2D.evaluate(X[0, :], X[1, :], satamp, cx[0], cy[0], gamma, alpha)
    cal_top = Moffat2D.evaluate(X[0, :], X[1, :], satamp, cx[1], cy[1], gamma, alpha)
    cal_left = Moffat2D.evaluate(X[0, :], X[1, :], satamp, cx[2], cy[2], gamma, alpha)
    cal_bot = Moffat2D.evaluate(X[0, :], X[1, :], satamp, cx[3], cy[3], gamma, alpha)
    psf = Moffat2D.evaluate(X[0, :], X[1, :], amp, x0, y0, gamma, alpha)
    return cal_right + cal_top + cal_left + cal_bot + psf

# set default arguments
amp = calib_frame[ctr]
gamma = 2
alpha = 1
contrast = -2

P0 = [ctr[1], ctr[0], sep, theta0, amp, gamma, alpha, contrast]

print(f"Initial params:\n{P0}")

# fit
popt, pcov = curve_fit(model, X, calib_frame.ravel(), P0, maxfev=int(1e5))
sigs = np.sqrt(np.diagonal(pcov))
print(f"Fit params:\n{popt}")
print(f"Fit stddev (est):\n{sigs}")

# now calculate contrast using aperture photometry
aprad = 10 * popt[5]
annrad = 4 * popt[5], 6 * popt[5]
angles = np.linspace(0, 2*np.pi, 4, endpoint=False) + popt[3]
cy = popt[2] * np.sin(angles) + popt[1]
cx = popt[2] * np.cos(angles) + popt[0]
ctrs = list(zip(cx, cy))
ctrs.append((popt[0], popt[1]))
aps = CircularAperture(ctrs, aprad)
bkg_aps = CircularAnnulus(ctrs, annrad[0], annrad[1])
phot = aperture_photometry(calib_frame, aps)
bkg_phot = aperture_photometry(calib_frame, bkg_aps)
avg_bkg = bkg_phot["aperture_sum"] / bkg_aps.area
spot_phot = phot["aperture_sum"][0:4] - avg_bkg[0:4] * aps.area
cent_phot = phot["aperture_sum"][-1] - avg_bkg[-1] * aps.area
contrasts = np.log10(spot_phot) - np.log10(cent_phot)
contrast = np.median(contrasts)

report = f"""
center (x, y): ({popt[0]:.1f}, {popt[1]:.1f})
separation [px]: {popt[2]:.1f}
          [mas]: {popt[2] * 6.24:.0f}
          [l/d]: {popt[2] * 6.24 / (np.rad2deg(750e-9 / 7.79) * 3.6e6):.1f}
angle offset (°): {np.rad2deg(popt[3]):.0f})
contrast: 10^{contrast:.2f}
"""
print(report)
with open(procdir / "satspot_fit.txt", "w") as fh:
    fh.write(report)

fit = model(X, *popt)
fit_frame = np.reshape(fit, (512, 512))
fits.writeto(procdir / "Open_ag50nm-b2_model.fits", fit_frame, overwrite=True)
