using Test
using FFTW

using ClimateMachine.ConfigTypes
using ClimateMachine.Spectra
using ClimateMachine.Spectra:
    compute_gaussian!,
    compute_legendre!,
    SpectralSphericalMesh,
    trans_grid_to_spherical!,
    compute_wave_numbers

include("spherical_helper_test.jl")

@testset "power_spectrum_1d (GCM)" begin
    FT = Float64
    # -- TEST 1: power_spectrum_1d(AtmosGCMConfigType(), var_grid, z, lat, lon, weight)
    nlats = 32

    # Setup grid
    sinθ, wts = compute_gaussian!(nlats)
    yarray = asin.(sinθ) .* 180 / π
    xarray = 180.0 ./ nlats * collect(FT, 1:1:(2nlats))[:] .- 180.0
    z = 1

    # Setup variable
    mass_weight = ones(Float64, length(z))
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
    nm_spectrum, wave_numbers = power_spectrum_1d(
        AtmosGCMConfigType(),
        var_grid,
        z,
        yarray,
        xarray,
        mass_weight,
    )

    nm_spectrum_ = nm_spectrum[:, 10, 1]
    var_grid_ = var_grid[:, 10, 1]
    sum_spec = sum((nm_spectrum_ .* conj(nm_spectrum_)))
    sum_grid = sum(0.5 .* (var_grid_ .^ 2)) / length(var_grid_)

    sum_res = (sum_spec - sum_grid) / sum_grid

    @test sum_res < 0.1
end

@testset "power_spectrum_2d (GCM)" begin
    # -- TEST 2: power_spectrum_2d
    # Setup grid
    FT = Float64
    nlats = 32
    sinθ, wts = compute_gaussian!(nlats)
    cosθ = sqrt.(1 .- sinθ .^ 2)
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

    mass_weight = ones(Float64, z)
    spectrum, wave_numbers, spherical, mesh =
        power_spectrum_2d(AtmosGCMConfigType(), var_grid, mass_weight)

    # Grid to spherical to grid reconstruction
    reconstruction = trans_spherical_to_grid!(mesh, spherical)

    sum_spec = sum((spectrum))
    sum_grid =
        sum(0.5 .* var_grid[:, :, 1] .^ 2 * reshape(cosθ, (length(cosθ), 1)))
    sum_reco = sum(
        0.5 .* reconstruction[:, :, 1] .^ 2 * reshape(cosθ, (length(cosθ), 1)),
    )

    sum_res_1 = (sum_spec - sum_grid) / sum_grid
    sum_res_2 = (sum_reco - sum_grid) / sum_grid

    @test abs(sum_res_1) < 0.5
    @test abs(sum_res_2) < 0.5
end
