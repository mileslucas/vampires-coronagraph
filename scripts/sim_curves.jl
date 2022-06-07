using PythonCall
using ADI
using NPZ
using Glob
using ProgressMeter

pro = pyimport("proplot")
pro.rc["style"] = "ggplot"
pro.rc["font.size"] = 10
pro.rc["legend.facecolor"] = "0.90"
pro.rc["font.name"] = "Times New Roman"

rootdir(args...) = joinpath(@__DIR__, "..", args...)
datadir(args...) = rootdir("data", args...)
figdir(args...) = rootdir("paper", "figures", args...)
mkpath(figdir())

files = glob("fpm-*_data.npz", datadir())
@info "loading images"
images = map(files) do filename
    data = npzread(filename)
    psf = data["psf"]
    occ_psf = data["occulted_psf"]
    # rescale
    norm_amp = maximum(psf)
    psf
    occ_psf
    (psf, occ_psf)
end

# FWHM is 4 based on q-value in focal grid used
fwhm = 4
starphot = Metrics.estimate_starphot(images[1][1], fwhm)
@info "calculating noise maps"
noisemaps = @showprogress map(images) do (psf, occ_psf)
    psf_noise = detectionmap(noise, psf, fwhm)
    occ_psf_noise = detectionmap(noise, occ_psf, fwhm)
    (psf_noise, occ_psf_noise)
end

sigma = 5 # 5σ contrast curves
@info "calculating contrast curves"
curves = @showprogress map(noisemaps) do (psf_noise, occ_psf_noise)
    radii, psf_curve = radial_profile(psf_noise)
    _, occ_psf_curve = radial_profile(occ_psf_noise)

    # correct for small sample statistics
    sigma_corr = Metrics.correction_factor.(radii, fwhm, sigma)

    psf_contrast = @. Metrics.calculate_contrast(sigma_corr, psf_curve / starphot)
    occ_psf_contrast = @. Metrics.calculate_contrast(sigma_corr, occ_psf_curve / starphot)

    (radii, psf_contrast, occ_psf_contrast)
end
radius = curves[1][1] ./ 4 .* rad2deg(750e-9 / 7.79) * 3600;
iwas = [37, 55, 91, 128]
names = ["CLC-$i ($iwa mas)" for (i, iwa) in zip((2, 3, 5, 7), iwas)]
psfm = rad2deg(750e-9 / 7.79) * 3600 .< radius .< 1.5
m = @. ((iwas') * 1e-3) < radius < 1.5

cycle = pro.Cycle("magma", 4, left=0.3, right=0.8)
colors = collect(cycle)[end:-1:begin]

@info "plotting"
fig, axs = pro.subplots(refwidth="6.5in", refheight="3.5in")
axs.plot(radius[psfm], curves[1][2][psfm], c="k", label="PSF")

for (i, datum) in enumerate(curves)
    axs.plot(radius[m[:, i]], datum[3][m[:, i]], c=colors[i]["color"], label=names[i])
    axs.axvline(iwas[i:i] * 1e-3, c=colors[i]["color"], ls=":")
end

axs.legend(loc="best", ncols=1)
axs.format(
    xlabel="separation [\"]",
    ylabel="raw 5\$\\sigma\$ contrast",
    yscale="log",
    yformatter="log",
    xlim=(0, 1.5),
    ylim=(nothing, 1)
)
fname = figdir("simulated_curves.pdf")
fig.savefig(fname)
@info "done" filename=fname