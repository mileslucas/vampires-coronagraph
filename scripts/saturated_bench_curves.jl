using ADI
using BiweightStats
using FITSIO
using Glob
using NPZ
using Photometry
using ProgressMeter
using PythonCall
using SatelliteSpots: get_cutout_inds, center_of_mass
using Statistics
using Unitful, UnitfulAngles

pro = pyimport("proplot")
pro.rc["style"] = "ggplot"
pro.rc["font.size"] = 10
pro.rc["legend.facecolor"] = "0.90"
pro.rc["font.name"] = "Times New Roman"

rootdir(args...) = joinpath(@__DIR__, "..", args...)
function datadir(args...)
    joinpath("/Volumes/mlucas SSD1/vampires-coronagraph-data/bench_20220629", args...)
end
procdir(args...) = datadir("processed", args...)
mkpath(procdir())
figdir(args...) = rootdir("paper", "figures", args...)
mkpath(figdir())

dark_file = datadir("darks_20ms_em0_750-50_Mirror_0_cam1.fits")

function produce_or_load(f, filename; force = get(ENV, "FORCE", "false") == "true")
    # produce data and save it, returning data
    if force || !isfile(filename)
        data = f()
        FITS(fh -> write(fh, data), filename, "w")
        return data
    end
    # load data
    data = read(FITS(filename)[1])
    return data
end

@info "Making master dark"
master_dark = begin
    outname = procdir("master_dark_em0_20ms.fits")
    dark_frame = produce_or_load(outname) do
        hdu = FITS(dark_file)[1]
        dark_cube = Float32.(read(hdu, :, :, 3:size(hdu, 3)))
        return median(dark_cube, dims = 3)
    end
end

sci_files = glob("CLC-*_cam1.fits", datadir())
push!(sci_files, datadir("Open_ag50nm-b2_750-50_EmptySlot2_0_cam1.fits"))
calib_frames = @showprogress "Calibrating and collapsing" map(sci_files) do filename
    hdu = FITS(filename)[1]
    base = basename(filename)
    outname = procdir(replace(base, ".fits" => "_calib_collapsed.fits"))
    calib_frame = produce_or_load(outname) do
        sci_cube = Float32.(read(hdu, :, :, 3:size(hdu, 3)))
        cal_cube = sci_cube .- master_dark
        cal_frame = median(cal_cube, dims = 3)[:, :, 1]
        return reverse(cal_frame, dims = 2)
    end
    return calib_frame
end

# FWHM is ~6 based on fitting
fwhm = 6.36
aparea = ?? * (1.5 * fwhm)^2
bkgarea = ?? * ((3 * fwhm)^2 - (2 * fwhm)^2)
center = (259.4, 249.4)
sep = 49.4 # px
angles = range(0, 270, length = 4) .- 4
inds = map(ang -> get_cutout_inds(axes(calib_frames[1]), sep, ang; width = 30, center),
           angles)

satresults = @showprogress "calculating satspot photometry and center" map(calib_frames) do frame
    com = map(ind -> center_of_mass(frame, CartesianIndices(ind)), inds)
    aps = map(com) do ctr
        CircularAperture(ctr, 1.5 * fwhm)
    end
    bkgaps = map(com) do ctr
        CircularAnnulus(ctr, 2 * fwhm, 3 * fwhm)
    end
    mx = mean(c -> c[1], com)
    my = mean(c -> c[2], com)
    phots = photometry(aps, frame).aperture_sum
    bkgphots = photometry(bkgaps, frame).aperture_sum
    calphots = @. phots - bkgphots * aparea / bkgarea
    return median(calphots), (mx, my)
end
centered_frames = map(calib_frames, satresults) do frame, (_, ctr)
    return shift_frame(frame, 256.5 .- ctr)
end

# mask out satellite spots
masked_frames = map(centered_frames) do frame
    out = copy(frame)
    for ang in angles
        # principal sat spots
        inds = get_cutout_inds(axes(calib_frames[1]), sep + 5, ang; width = 40)
        out[inds...] .= NaN
        # # secondary sat spots
        inds = get_cutout_inds(axes(calib_frames[1]), sep * sqrt(2), ang + 45; width = 20)
        out[inds...] .= NaN
    end
    return out
end

# mask out saturation
mask = masked_frames[5] .> 2^15
masked_frames[5][mask] .= NaN

function robustnoise(image, position, fwhm)
    pos = Tuple(position)
    sep = Metrics.radial_distance(pos, Metrics.center(image))
    fluxes = Metrics.get_aperture_fluxes(image, pos, sep, fwhm)
    return BiweightStats.scale(fluxes; c = 6)
end

noisemaps = @showprogress "calculating noise maps" map(img -> detectionmap(robustnoise, img, fwhm), masked_frames)

sigma = 5 # 5?? contrast curves

dark_frame_fix = @views master_dark[:, :, 1] .- median(master_dark[:, :, 1], dims=2)
dark_frame_fix .-= median(dark_frame_fix, dims=1)

@info "calculating noise floors"
noisemap = detectionmap(robustnoise, dark_frame_fix, fwhm)
radii, curve = radial_profile(noisemap)
m = @. fwhm < radii < 256 - fwhm/2
r = radii[m]
# correct for small sample statistics
starphot = median(map(first, satresults[1:4])) * 10^(1.52)
contrast = @. Metrics.calculate_contrast(sigma, curve[m] / starphot)
noise_floor = median(contrast)

curves = @showprogress "calculating contrast curves" map(noisemaps, satresults) do frame, (phot, _)
    radii, curve = radial_profile(frame)
    m = @. fwhm < radii < 256
    # correct for small sample statistics
    sigma_corr = Metrics.correction_factor.(radii[m], fwhm, sigma)
    starphot = phot * 10^(1.52)
    contrast = @. Metrics.calculate_contrast(sigma_corr, curve[m] / starphot)
    (radii[m], contrast)
end
radius = curves[1][1] .* 6.24e-3;
iwas = [36, 55, 92, 129]
names = ["CLC-$i ($iwa mas)" for (i, iwa) in zip((2, 3, 5, 7), iwas)]
psfm = 5.5 .* ustrip(u"arcsecond", 750u"nm" / 7.79u"m") .< radius .< 1.5
m = @. ((iwas') * 1e-3) < radius < 1.5

cycle = pro.Cycle("magma", 4, left = 0.3, right = 0.8)
colors = collect(cycle)[end:-1:begin]

@info "plotting"
fig, axs = pro.subplots(refwidth = "6.5in", refheight = "3.66in")
axs.plot(radius[psfm], curves[5][2][psfm], c = "k", label = "PSF")

for (i, datum) in enumerate(curves[1:4])
    axs.plot(radius[m[:, i]], datum[2][m[:, i]], c = colors[i]["color"], label = names[i])
    axs.axvline(iwas[i:i] * 1e-3, c = colors[i]["color"], ls = ":")
end
axs.axhline(noise_floor, c="k", alpha=0.4, ls="--", label="noise floor")


axs.legend(loc = "best", ncols = 1)
axs.format(xlabel = "separation [\"]",
           ylabel = "raw 5\$\\sigma\$ noise per annulus [contrast]",
           yscale = "log",
           yformatter = "log",
           xlim = (0, 1.5),
           ylim = (nothing, 1))
fname = figdir("saturated_bench_20220628_curves.pdf")
fig.savefig(fname)
@info "done" filename=fname
