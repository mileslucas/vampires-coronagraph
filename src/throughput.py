import numpy as np
import proplot as pro
from pathlib import Path
from scipy.optimize import curve_fit

rootdir = Path(__file__).parent.parent
datadir = rootdir / "data"
figdir = rootdir / "paper" / "figures"

pro.rc["style"] = "ggplot"
pro.rc["legend.facecolor"] = "0.90"
pro.rc["font.name"] = "Times New Roman"

files = sorted(datadir.glob("CLC-*_throughput.npz"))
data = map(np.load, files)

fig, axes = pro.subplots(refwidth="4.5in", refheight="3in")

cycler = pro.Cycle("magma", 4, left=0.3, right=0.8)
colors = list(cycler)
colors.reverse()
labels = [f"CLC-{i}" for i in (2, 3, 5, 7)]
expected = (35, 52, 90, 127)

focal_length = 200 # 200 mm
magnification = 7.79 / 7.032e-3

def logistic(x, x0, k):
    return 1 / (np.exp(-k * (x - x0)) + 1)

for i, datum in enumerate(data):
    offset_mas = np.rad2deg(datum["offsets"] / focal_length / magnification) * 3.6e6
    # correct so maximum is 1
    mask = offset_mas > (expected[i] + 20)
    factor = np.median(datum["throughput"][mask])
    # correct throughput so bottom is 0
    extent = factor - datum["throughput"].min()
    through = (datum["throughput"] - datum["throughput"].min()) / extent

    # fit logistic function
    popt, pcov = curve_fit(logistic, offset_mas, through, p0=[50, 1])

    line_fit = logistic(offset_mas, *popt)
    
    axes.plot(offset_mas, line_fit, c="0.8")
    axes.axvline(popt[0], c=colors[i]["color"], ls=":")
    axes.plot(offset_mas, through, c=colors[i]["color"], label=labels[i] + f" ({popt[0]:.0f} mas)")

axes.axhline([0.5], c="0.3", ls="--")
axes[0].legend(loc="t", ncols=2)
axes.format(
    xlabel="separation [mas]",
    ylabel="throughput",
    xlim=(0, None),
    ylim=(-0.05, 1.05)
)
fig.savefig(figdir / "throughput_curves.pdf")