include("spherical_helper.jl")

"""
    power_spectrum_1d(::AtmosGCMConfigType, var_grid, z, lat, lon, weight)

Calculates the 1D (zonal) power spectra using the fourier transform at each latitude and level from a 3D velocity field.
The snapshots of these spectra should be averaged to obtain a time-average.
The input velocities must be interpolated to a Gaussian grid.

# References
- A. Wiin-Nielsen (1967) On the annual variation and spectral distribution of atmospheric energy, Tellus, 19:4, 540-559, DOI: 10.3402/tellusa.v19i4.9822
- Koshyk, J. N., and K. Hamilton, 2001: The Horizontal Kinetic Energy Spectrum and Spectral Budget Simulated by a High-Resolution Troposphere–Stratosphere–Mesosphere GCM. J. Atmos. Sci., 58, 329–348, https://doi.org/10.1175/1520-0469(2001)058<0329:THKESA>2.0.CO;2.

# Arguments
- var_grid: variable (typically u or v) on a Gausian (lon, lat, z) grid to be transformed
- z: vertical coordinate (height or pressure)
- lat: latitude
- lon: longitude
- mass_weight: for mass-weighted calculations
"""
function power_spectrum_1d(::AtmosGCMConfigType, var_grid, z, lat, lon, weight)
    num_lev = length(z)
    num_lat = length(lat)
    num_lon = length(lon)
    num_fourier = Int64(num_lon)

    # get number of positive Fourier coefficients incl. 0
    if mod(num_lon, 2) == 0 # even
        num_pfourier = div(num_lon, 2)
    else # odd
        num_pfourier = div(num_lon, 2) + 1
    end

    zon_spectrum = zeros(Float64, num_pfourier, num_lat, num_lev)
    freqs = zeros(Float64, num_pfourier, num_lat, num_lev)

    for k in 1:num_lev
        for j in 1:num_lat
            # compute fft frequencies for each latitude
            x = lon ./ 180.0 .* π
            dx = (lon[2] - lon[1]) ./ 180.0 .* π

            freqs_ = fftfreq(num_fourier, 1.0 / dx) # 0,+ve freq,-ve freqs (lowest to highest)
            freqs[:, j, k] = freqs_[1:num_pfourier] .* 2.0 .* π

            # compute the fourier coefficients for all latitudes
            fourier = fft(var_grid[:, j, k]) # e.g. vcos_grid, ucos_grid
            fourier = (fourier / num_fourier)

            # convert to energy spectra
            zon_spectrum[1, j, k] =
                zon_spectrum[1, j, k] +
                0.5 * weight[k] * fourier[1] .* conj(fourier[1])

            for m in 2:num_pfourier
                zon_spectrum[m, j, k] =
                    zon_spectrum[m, j, k] +
                    2.0 * weight[k] * fourier[m] * conj(fourier[m]) # factor 2 for neg freq contribution
            end
        end
    end
    return zon_spectrum, freqs
end

"""
    power_spectrum_2d(::AtmosGCMConfigType, var_grid, mass_weight)

- transform variable on grid to the 2d spectral space using fft on latitude circles
(as for the 1D spectrum) and Legendre polynomials for meridians, and calculate spectra

# Arguments
- var_grid: variable (typically u or v) on a Gausian (lon, lat, z) grid to be transformed
- mass_weight: weight for mass-weighted calculations

# References
Baer, F., 1972: An Alternate Scale Representation of Atmospheric Energy Spectra. J. Atmos. Sci., 29, 649–664, https://doi.org/10.1175/1520-0469(1972)029<0649:AASROA>2.0.CO;2
"""
function power_spectrum_2d(::AtmosGCMConfigType, var_grid, mass_weight)
    #  initialize spherical mesh variables
    nθ, nd = (size(var_grid, 2), size(var_grid, 3))
    mesh = SpectralSphericalMesh(nθ, nd)
    var_spectrum = mesh.var_spectrum
    var_spherical = mesh.var_spherical

    sinθ, wts = compute_gaussian!(mesh.nθ) # latitude weights using Gaussian quadrature, to orthogonalize Legendre polynomials upon summation
    mesh.qnm =
        compute_legendre!(mesh.num_fourier, mesh.num_spherical, sinθ, mesh.nθ) #  normalized associated Legendre polynomials

    for k in 1:(mesh.nd)
        # apply Gaussian quadrature weights
        for i in 1:(mesh.nθ)
            mesh.qwg[:, :, i] .= mesh.qnm[:, :, i] * wts[i] * mass_weight[k]
        end

        # Transform variable using spherical harmonics
        var_spherical[:, :, k, :] =
            trans_grid_to_spherical!(mesh, var_grid[:, :, k]) # var_spherical[m,n,k,sinθ]

        # Calculate energy spectra
        var_spectrum[:, :, k] =
            sum(var_spherical[:, :, k, :], dims = 3) .*
            conj(sum(var_spherical[:, :, k, :], dims = 3))  # var_spectrum[m,n,k]
    end
    return var_spectrum[2:end, 2:end, :] .* nθ .^ 2,
    mesh.wave_numbers,
    var_spherical,
    mesh
end

# TODO: enable mass weighted and vertically integrated calculations
