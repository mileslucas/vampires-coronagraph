from pathlib import Path
import numpy as np
from astropy.io import fits
import proplot as pro

pro.rc["font.size"] = 12
pro.rc["legend.facecolor"] = "0.90"
pro.rc["font.name"] = "Times New Roman"
pro.rc["image.origin"] = "lower"
pro.rc["image.cmap"] = "magma"
pro.rc["axes.grid"] = False


datadir = Path("/Volumes/mlucas SSD1/vampires-coronagraph-data")
figdir = Path(__file__).parent / ".." / "paper" /"figures"
procdir = datadir / "processed"
procdir.mkdir(exist_ok=True)

# # make dark frame
# bench_dark_filename = datadir / "bench_20220526" / "darks_em110_20ms_750-50_Mirror_0_cam1.fits"
# bench_dark_cube, bench_dark_hdr = fits.getdata(bench_dark_filename, header=True)
# bench_dark_frame = np.median(bench_dark_cube, axis=0)
# fits.writeto(procdir / "master_dark_em0_20ms.fits", bench_dark_frame, header=bench_dark_hdr, overwrite=True)

# bench data
bench_filename = datadir / "bench_20220526" / "CLC-5_ag50nm-b2_750-50_LyotStop_0_cam1.fits"
bench_cube, bench_hdr = fits.getdata(bench_filename, header=True)
bench_calib_cube = np.flip(bench_cube, axis=-2)
bench_calib_frame = np.median(bench_calib_cube, axis=0)
fits.writeto(procdir / "CLC-5_ag50nm-b2_750-50_LyotStop_0_cam1_calib_collapsed.fits", bench_calib_frame, header=bench_hdr, overwrite=True)

# # make dark frame
# sky_dark_filename = datadir / "20220512" / "dark_em300_200ms_20220516_Open_Mirror_00_cam1.fits"
# sky_dark_cube, sky_dark_hdr = fits.getdata(sky_dark_filename, header=True)
# sky_dark_frame = np.median(sky_dark_cube, axis=0)
# fits.writeto(procdir / "master_dark_em300_200ms.fits", sky_dark_frame, header=sky_dark_hdr, overwrite=True)

# bench data
sky_filename = datadir / "20220512" / "HIP56083_CLC-5_20220513_750-50_LyotStop_010_cam1.fits"
sky_cube, sky_hdr = fits.getdata(sky_filename, header=True)
sky_calib_cube = np.flip(sky_cube, axis=-2)
maxdx = np.argmax(np.var(sky_calib_cube, axis=(1, 2)))
sky_calib_frame = sky_calib_cube[maxdx]
fits.writeto(procdir / "HIP56083_CLC-5_20220513_750-50_LyotStop_010_cam1_calib_collapsed.fits", sky_calib_frame, header=sky_hdr, overwrite=True)

# crop
bench_frame = bench_calib_frame[int(249.4 - 123):int(249.4 + 123), int(259.4 - 123):int(259.4 + 123)]
sky_frame = sky_calib_frame[int(126 - 123):int(126 + 123), int(123 - 123):int(123 + 123)]

# fix bench frame
# bench_frame[bench_frame < 1] = 5 * np.random.randn(246, 246)[bench_frame < 1] + 18

fig, axes = pro.subplots(ncols=2, share=0, refwidth="3in", refheight="3in", abc=True, abcloc="ul", abcbbox=True)

c = axes[0].imshow(bench_frame, norm="log")
axes[0].colorbar(c, formatter="log")
c = axes[1].imshow(sky_frame, norm="log")
axes[1].colorbar(c, formatter="log")

ticks = [f"{a:.1f}" for a in np.linspace(-113, 113, 5) * 6.24e-3]

axes.format(
    xlabel="x [\"]",
    ylabel="y [\"]",
    xlocator=np.linspace(-113, 113, 5) + 123,
    ylocator=np.linspace(-113, 113, 5) + 123,
    xticklabels=ticks,
    yticklabels=ticks,
)

fig.savefig(figdir / "astrogrid_psf.pdf")