using FITSIO
using Statistics
using Photometry
using PSFModels
using LuckyImaging

datadir(args...) = joinpath(@__DIR__, "..", "data", "throughput", args...)

filenames = [
    datadir("CLC_throughput_750-50_EmptySlot2_1_cam1.fits"), # Open
    datadir("CLC_throughput_750-50_EmptySlot2_0_cam1.fits"), # offset FPM
    datadir("CLC_throughput_750-50_LyotStop_0_cam1.fits") # Lyot
]

cubes = map(filenames) do filename
    hdu = only(FITS(filename))
    Float32.(read(hdu))
end

collapsed = map(cubes) do cube
    lucky_image(cube; dims=3, q=0, register=:dft, upsample_factor=10, shift_reference=true)
end

for (filename, frame) in zip(filenames, collapsed)
    outname = replace(filename, ".fits" => "_collapsed.fits")
    FITS(fh -> write(fh, frame), outname, "w")
end

results = map(collapsed) do frame
    init = (;x=256.5, y=256.5, amp=maximum(frame), bkg=200, fwhm=4, ratio=0.4)
    window = (246:266, 246:266)
    fit = PSFModels.fit(airydisk, init, frame, window)
    return fit[1]
end

avefwhm = mean(r -> r.fwhm, results)

aps = map(results) do res
    CircularAperture(res.x, res.y, 0.667f0 * 5 * avefwhm)
end

apsums = map(collapsed, aps, results) do frame, ap, res
    src_flux = photometry(ap, frame).aperture_sum
    bkg_flux = res.bkg * sum(ap)
    return src_flux - bkg_flux
end