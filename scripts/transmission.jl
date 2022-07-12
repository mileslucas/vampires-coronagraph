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
    joinpath("/Volumes/mlucas SSD1/vampires-coronagraph-data/transmission/", args...)
end
procdir(args...) = datadir("processed", args...)
mkpath(procdir())
figdir(args...) = rootdir("paper", "figures", args...)
mkpath(figdir())

dark_file = datadir("darks_em0_5ms_750-50_Mirror_0_cam1.fits")

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
    dark_frame = produce_or_load(outname, force=true) do
        hdu = FITS(dark_file)[1]
        dark_cube = Float32.(read(hdu, :, :, 3:size(hdu, 3)))
        return median(dark_cube, dims = 3)
    end
end

sci_files = glob("*750-50_EmptySlot_*_cam1.fits", datadir())
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

centers = [
    (61.7, 72.8),
    (61.9, 73.1),
    (61.7, 73.7),
    (61.7, 74.5),
    (61.7, 73.1),
]

centered_frames = @showprogress "centering frames" map(sci_files, calib_frames, centers) do filename, frame, ctr
    base = basename(filename)
    outname = procdir(replace(base, ".fits" => "_calib_collapsed_centered.fits"))
    produce_or_load(() -> shift_frame(frame, 64.5 .- ctr), outname, force=true)
end

# determine mask size
fwhm = 6.36
ap = CircularAperture(64.5, 64.5, 3)
ann = CircularAnnulus(64.5, 64.5, 3, 4)
apsums = @showprogress "measuring photometry" map(centered_frames) do frame
    apsum = photometry(ap, frame).aperture_sum
    annsum = photometry(ann, frame).aperture_sum
    return apsum - annsum * sum(ap) / sum(ann)
end

# relative flux of unocculted data
apsums[end] *= 10^(2.6243286403990362 - 0.7355975939297882)

transmissions = log10.(apsums[1:4]) .- log10(apsums[end])

med_trans = median(transmissions)