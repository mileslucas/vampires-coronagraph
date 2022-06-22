using PythonCall
using ADI
using NPZ
using Glob
using ProgressMeter
using Unitful, UnitfulAngles
using SatelliteSpots: get_cutout_inds, center_of_mass
using Statistics
using Photometry

pro = pyimport("proplot")
pro.rc["style"] = "ggplot"
pro.rc["font.size"] = 10
pro.rc["legend.facecolor"] = "0.90"
pro.rc["font.name"] = "Times New Roman"

rootdir(args...) = joinpath(@__DIR__, "..", args...)
function datadir(args...)
    joinpath("/Volumes/mlucas SSD1/vampires-coronagraph-data/20220512", args...)
end
procdir(args...) = datadir("processed", args...)
mkpath(procdir())
figdir(args...) = rootdir("paper", "figures", args...)
mkpath(figdir())

dark_files = glob("dark*_cam1.fits", datadir())

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

dark_frames = @showprogress "Making dark frames" map(dark_files) do filename
    hdu = FITS(filename)[1]
    key = (;
           gain = round(Int, read_key(hdu, "U_EMGAIN")[1]),
           t = round(Int, read_key(hdu, "U_AQTINT")[1] / 1000))
    outname = procdir("master_dark_em$(key.gain)_$(key.t)ms.fits")
    dark_frame = produce_or_load(outname) do
        dark_cube = Float32.(read(hdu, :, :, 3:size(hdu, 3)))
        return median(dark_cube, dims = 3)
    end
    if any(==(512), size(dark_frame))
        dark_frame = dark_frame[128:383, 128:383]
    end
    return key => dark_frame
end
dark_frame_dict = Dict(dark_frames)

sci_files = glob("HIP56083*_cam1.fits", datadir())
# push!(sci_files, datadir("Open_ag50nm-b2_750-50_EmptySlot2_0_cam1.fits"))
calib_frames = @showprogress "Calibrating and collapsing" map(sci_files) do filename
    hdu = FITS(filename)[1]
    key = (;
           gain = round(Int, read_key(hdu, "U_EMGAIN")[1]),
           t = round(Int, read_key(hdu, "U_AQTINT")[1] / 1000))
    base = basename(filename)
    outname = procdir(replace(base, ".fits" => "_calib_collapsed.fits"))
    calib_frame = produce_or_load(outname) do
        sci_cube = Float32.(read(hdu, :, :, 3:size(hdu, 3)))
        cal_cube = sci_cube .- dark_frame_dict[key]
        cal_frame = median(cal_cube, dims = 3)[:, :, 1]
        return reverse(cal_frame, dims = 2)
    end
    return calib_frame
end

# FWHM is ~6 based on fitting
fwhm = 6.36
aparea = π * (1.5 * fwhm)^2
bkgarea = π * ((3 * fwhm)^2 - (2 * fwhm)^2)
center = (124, 125)
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
    return shift_frame(frame, 128.5 .- ctr)
end

# mask out satellite spots
masked_frames = map(centered_frames) do frame
    out = copy(frame)
    for ang in angles
        # principal sat spots
        inds = get_cutout_inds(axes(frame), sep, ang; width = 20)
        out[inds...] .= minimum(frame[inds...])
    end
    return out
end

noisemaps = @showprogress "calculating noise maps" map(img -> detectionmap(noise, img, fwhm),
                                                       masked_frames)

sigma = 5 # 5σ contrast curves
curves = @showprogress "calculating contrast curves" map(noisemaps,
                                                         satresults) do noisemap, (phot, _)
    radii, curve = radial_profile(noisemap)
    m = radii .> fwhm
    # correct for small sample statistics
    sigma_corr = Metrics.correction_factor.(radii[m], fwhm, sigma)
    contrast = @. Metrics.calculate_contrast(sigma_corr, curve[m] / (phot * 10^(1.52)))

    (radii[m], contrast)
end

groups = [
    1:13,
    14:27,
    28:38,
    39:51,
]
selected_curves = map(g -> argmin(c -> median(c[2]), curves[g]), groups)

radius = selected_curves[1][1] .* 6.24e-3;
iwas = [37, 55, 91, 128]
names = ["CLC-$i ($iwa mas)" for (i, iwa) in zip((2, 3, 5, 7), iwas)]
psfm = ustrip(u"arcsecond", 750u"nm" / 7.79u"m") .< radius .< 1.5
m = @. ((iwas') * 1e-3) < radius < 1.5

cycle = pro.Cycle("magma", 4, left = 0.3, right = 0.8)
colors = collect(cycle)[end:-1:begin]

@info "plotting"
fig, axs = pro.subplots(refwidth = "6.5in", refheight = "3.66in")
# axs.plot(radius[psfm], selected_curves[5][2][psfm], c="k", label="PSF")

for (i, datum) in enumerate(selected_curves[1:4])
    axs.plot(radius[m[:, i]], datum[2][m[:, i]], c = colors[i]["color"], label = names[i])
    axs.axvline(iwas[i:i] * 1e-3, c = colors[i]["color"], ls = ":")
end

# axs.fill_betweenx(0.2, 0.4, c="k", alpha=0.1)
axs.axvline(49.4 * 6.24e-3, c = "k", alpha = 0.3, lw = 1, label = "Satellite spots")

axs.legend(loc = "best", ncols = 1)
axs.format(xlabel = "separation [\"]",
           ylabel = "raw 5\$\\sigma\$ contrast",
           yscale = "log",
           yformatter = "log",
           xlim = (0, 0.75),
           ylim = (nothing, 1))
fname = figdir("HIP56083_20220512_curves.pdf")
fig.savefig(fname)
@info "done" filename=fname