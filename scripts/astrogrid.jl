using FITSIO
using PSFModels
using Photometry
using Unitful
using UnitfulAngles
using Optim
using Statistics
using SatelliteSpots: get_cutout_inds

function datadir(args...)
    joinpath("/Volumes/mlucas SSD1/vampires-coronagraph-data/bench_20220526", args...)
end
figdir(args...) = joinpath(@__DIR__, "..", "paper", "figures", args...)
procdir(args...) = datadir("processed", args...)
mkpath(procdir())

ENV["FORCE"] = true

function produce_or_load(f, filename; force = get(ENV, "FORCE", "true") == "true")
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

@info "Making dark frame"
dark_filename = datadir("darks_em0_20ms_750-50_Mirror_0_cam1.fits")
dark_frame = produce_or_load(procdir("master_dark_em0_20ms.fits")) do
    hdu = FITS(dark_filename)[1]
    dark_cube = Float32.(read(hdu, :, :, 3:size(hdu, 3)))
    return median(dark_cube, dims = 3)
end

@info "Calibrating and collapsing data"
sci_filename = datadir("Open_ag50nm-b2_750-50_EmptySlot2_0_cam1.fits")
calib_frame = produce_or_load(procdir("Open_ag50nm-b2_750-50_EmptySlot2_0_cam1_calib_collapsed.fits")) do
    hdus = FITS(sci_filename)
    sci_cube = Float32.(read(hdus[1], :, :, 3:size(hdus[1], 3)))
    # dark subtract
    calib_cube = sci_cube .- dark_frame
    # median combine
    calib_frame = median(calib_cube, dims = 3)[:, :, 1]
    # reverse y-axis
    return reverse(calib_frame, dims = 2)
end

@info "Setting up satspot model"
ctr = CartesianIndex(argmax(calib_frame))
plate_scale = 6.24e-3u"arcsecond" # 6.24 mas / px
sep = 15.5 * uconvert(u"arcsecond", 750u"nm" / 7.79u"m") / plate_scale
cross_angle = -4u"°"
base_angles = Tuple(range(0u"°", 270u"°", length = 4))
angles = base_angles .+ cross_angle
# get 30x30 window cutouts
width = 30
window_inds = map(ang -> get_cutout_inds(axes(calib_frame), sep, ang; width, center = ctr.I),
                  angles)
# add central psf
push!(window_inds, get_cutout_inds(axes(calib_frame), 0, 0; width, center = ctr.I))
cart_inds = map(CartesianIndices, window_inds)

produce_or_load(procdir("Open_ag50nm-b2_750-50_EmptySlot2_0_cam1_calib_collapsed_cutout.fits")) do
    base = zero(calib_frame)
    for inds in cart_inds
        base[inds] += calib_frame[inds]
    end
    return base
end

# create hierarchical moffat model
function model(X::AbstractVector{T}) where {T}
    x0, y0, sep, t0, amp, fwhm, alpha, contrast = X
    angles = ustrip.(u"rad", base_angles) .+ t0
    cy = @. sep * sin(angles) + y0
    cx = @. sep * cos(angles) + x0
    satamp = amp * 10^contrast
    base = zeros(T, size(calib_frame))
    inds = CartesianIndices(base)
    base .+= moffat.(inds; amp = satamp, x = cx[1], y = cy[1], fwhm, alpha)
    base .+= moffat.(inds; amp = satamp, x = cx[2], y = cy[2], fwhm, alpha)
    base .+= moffat.(inds; amp = satamp, x = cx[3], y = cy[3], fwhm, alpha)
    base .+= moffat.(inds; amp = satamp, x = cx[4], y = cy[4], fwhm, alpha)
    base .+= moffat.(inds; amp = amp, x = x0, y = y0, fwhm, alpha)
    return base
end

# create loss function
psf_loss(inds, image, model) = mean(idx -> (image[idx] - model(idx))^2, inds)

function loss(X::AbstractVector{T}) where {T}
    x0, y0, sep, t0, amp, fwhm, alpha, contrast = X
    (fwhm < 0 || fwhm > 10) && return T(Inf)
    alpha < 0 && return T(Inf)
    angles = ustrip.(u"rad", base_angles) .+ t0
    cy = @. sep * sin(angles) + y0
    cx = @. sep * cos(angles) + x0
    satamp = amp * 10^contrast
    negloglike = psf_loss(cart_inds[1], calib_frame,
                          moffat(T; amp = satamp, x = cx[1], y = cy[1], fwhm, alpha))
    negloglike += psf_loss(cart_inds[2], calib_frame,
                           moffat(T; amp = satamp, x = cx[2], y = cy[2], fwhm, alpha))
    negloglike += psf_loss(cart_inds[3], calib_frame,
                           moffat(T; amp = satamp, x = cx[3], y = cy[3], fwhm, alpha))
    negloglike += psf_loss(cart_inds[4], calib_frame,
                           moffat(T; amp = satamp, x = cx[4], y = cy[4], fwhm, alpha))
    negloglike += psf_loss(cart_inds[5], calib_frame,
                           moffat(T; amp = amp, x = x0, y = y0, fwhm, alpha))

    return negloglike
end

P0 = Float32[ctr[1], ctr[2], sep, cross_angle, calib_frame[ctr], 5, 3, -2]
@info "fitting satellite spots" initial_params=Tuple(P0)

# fit
optoptions = Optim.Options(iterations = 10000)
res = optimize(loss, P0, NewtonTrustRegion(), optoptions; autodiff = :forward)

@info "optimization $(Optim.converged(res) ? "succeeded" : "failed")" P=Tuple(Optim.minimizer(res))

# # now calculat contrast using aperture photometry
# aprad = 3 * popt[5]
# angles = np.linspace(0, 2*np.pi, 4, endpoint=False) + popt[3]
# cy = popt[2] * np.sin(angles) + popt[1]
# cx = popt[2] * np.cos(angles) + popt[0]
# ctrs = list(zip(cx, cy))
# ctrs.append((popt[0], popt[1]))
# aps = CircularAperture(ctrs, aprad)
# phot = aperture_photometry(calib_frame, aps)
# spot_phot = phot["aperture_sum"][0:4]
# cent_phot = phot["aperture_sum"][-1]
# contrasts = np.log10(spot_phot) - np.log10(cent_phot)
# contrast = np.median(contrasts)

# report = f"""
# center (x, y): ({popt[0]:.1f}, {popt[1]:.1f})
# separation [px]: {popt[2]:.1f}
#           [mas]: {popt[2] * 6.24:.0f}
#           [l/d]: {popt[2] * 6.24 / (np.rad2deg(750e-9 / 7.79) * 3.6e6):.1f}
# angle offset (°): {np.rad2deg(popt[3]):.0f})
# contrast: 10^{contrast:.2f}
# """
# print(report)
# with open(procdir / "satspot_fit.txt", "w") as fh:
#     fh.write(report)

# fit = model(X, *popt)
# fit_frame = np.reshape(fit, (512, 512))
# fits.writeto(procdir / "Open_ag50nm-b2_model.fits", fit_frame, overwrite=True)
