# Standalone test file that tests spectra visually
#using Plots # uncomment when using the plotting code below

using ClimateMachine.Spectra:
    compute_gaussian!,
    compute_legendre!,
    SpectralSphericalMesh,
    trans_grid_to_spherical!,
    compute_wave_numbers
using FFTW


include("spherical_helper_test.jl")

FT = Float64
# -- TEST 1: power_spectrum_gcm_1d(AtmosGCMConfigType(), var_grid, z, lat, lon, weight)
nlats = 32

# Setup grid
sinθ, wts = compute_gaussian!(nlats)
yarray = asin.(sinθ) .* 180 / π
xarray = 180.0 ./ nlats * collect(FT, 1:1:(2nlats))[:] .- 180.0
z = 1

# Setup variable
mass_weight = ones(Float64, length(z));
var_grid =
    1.0 * reshape(
        sin.(xarray / xarray[end] * 5.0 * 2π) .* (yarray .* 0.0 .+ 1.0)',
        length(xarray),
        length(yarray),
        1,
    ) +
    1.0 * reshape(
        sin.(xarray / xarray[end] * 10.0 * 2π) .* (yarray .* 0.0 .+ 1.0)',
        length(xarray),
        length(yarray),
        1,
    )
nm_spectrum, wave_numbers = power_spectrum_gcm_1d(
    AtmosGCMConfigType(),
    var_grid,
    z,
    yarray,
    xarray,
    mass_weight,
);


# Check visually
plot(wave_numbers[:, 16, 1], nm_spectrum[:, 16, 1], xlims = (0, 20))
contourf(var_grid[:, :, 1])
contourf(nm_spectrum[2:20, :, 1])

# -- TEST 2: power_spectrum_gcm_2d
# Setup grid
sinθ, wts = compute_gaussian!(nlats)
yarray = asin.(sinθ) .* 180 / π
xarray = 180.0 ./ nlats * collect(FT, 1:1:(2nlats))[:] .- 180.0
z = 1

# Setup variable: use an example analytical P_nm function
P_32 = sqrt(105 / 8) * (sinθ .- sinθ .^ 3)
var_grid =
    1.0 * reshape(
        sin.(xarray / xarray[end] * 3.0 * π) .* P_32',
        length(xarray),
        length(yarray),
        1,
    )

mass_weight = ones(Float64, z);
spectrum, wave_numbers, spherical, mesh =
    power_spectrum_gcm_2d(AtmosGCMConfigType(), var_grid, mass_weight)

# Grid to spherical to grid reconstruction
reconstruction = trans_spherical_to_grid!(mesh, spherical)

# Check visually
contourf(var_grid[:, :, 1])
contourf(reconstruction[:, :, 1])
contourf(var_grid[:, :, 1] .- reconstruction[:, :, 1])

# Spectrum
contourf(
    collect(0:1:(mesh.num_fourier - 1))[:],
    collect(0:1:(mesh.num_spherical - 1))[:],
    (spectrum[:, :, 1])',
    xlabel = "m",
    ylabel = "n",
)

# Check magnitude
println(sum(spectrum))
println(sum(
    0.5 .* var_grid[:, :, 1] .^ 2 *
    reshape(sqrt.(1 .- sinθ .^ 2), (length(sinθ), 1)),
))
