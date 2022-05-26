using FITSIO
using Glob
using LinearAlgebra
using LuckyImaging
using Optim
using Photometry
using Plots
using PSFModels: Gaussian
using SAOImageDS9
using Statistics
using LossFunctions

DS9.connect()

datadir(args...) = joinpath("/Volumes/HCI_DATA/vampires-coro-benchdata", args...)

## load dark frames

function makedark(fname)
    data = read(FITS(fname)[1])[:, :, begin+1:end]
    return mean(data, dims=3)
end

# this set is for the coronagraphic observations
dark_frame_01_1 = makedark(datadir("bench_coro_test_dark_0.01_Open_Mirror_0_cam1.fits"))
dark_frame_01_2 = makedark(datadir("bench_coro_test_dark_0.01_Open_Mirror_0_cam2.fits"))
# this set is for the non-coronagraphiic observations
dark_frame_0001_1 = makedark(datadir("bench_coro_test_dark_0.0001_Open_Mirror_0_cam1.fits"))
dark_frame_0001_2 = makedark(datadir("bench_coro_test_dark_0.0001_Open_Mirror_0_cam2.fits"))

## load test cubes
all_cubes = readdir(glob"bench_coro_test_open_agon*.fits", datadir())
# load each cube and apply dark subtraction
cubes = map(all_cubes) do fname
    data = read(FITS(fname)[1])[:, :, begin+1:end]
    # flip cam2
    if occursin("cam2", fname)
        datat = reverse(data, dims=1)
    end
    if occursin("open", fname)
        if occursin("cam1", fname)
            return data .- dark_frame_0001_1
        else
            return data .- dark_frame_0001_2
        end
    else
        if occursin("cam1", fname)
            return data .- dark_frame_01_1
        else
            return data .- dark_frame_01_2
        end
    end
end

# do shift and add
cube_coadded = map(cubes) do cube
    classic_lucky_image(cube, dims=3; q=0, window=50, upsample_factor=4)
end

# remove line-by-line readout issues from short exposure frame transfer crap
cube_improved = map(cube_coadded) do cube
    rows = median(cube, dims=1)
    cube .- rows
end

## get aperture photometry of each central PSF
# average FWHM is ~3.2
centers = map(argmax, cube_improved)
aps = [CircularAperture(c.I, 1.6) for c in centers]

photsum = map(aps, cube_improved) do ap, cube
    phot = photometry(ap, cube)
    return phot.aperture_sum
end

res = Dict(zip(all_cubes, photsum))


photsum2 = map([CircularAperture(c.I, 3.2) for c in centers], cube_improved) do ap, cube
    phot = photometry(ap, cube)
    return phot.aperture_sum
end

res2 = Dict(zip(all_cubes, photsum2))

r1 = norm([293, 252] .- centers[1].I)
r2 = norm([209, 258] .- centers[1].I)

r3 = norm([303, 263] .- centers[2].I)
r4 = norm([219, 257] .- centers[2].I)

r5 = norm([297, 251] .- centers[3].I)
r6 = norm([207, 257] .- centers[3].I)

r = mean([r1, r2, r3, r4, r5, r6])
(r * 0.006) / (rad2deg(750e-9/7.92) * 3600)
