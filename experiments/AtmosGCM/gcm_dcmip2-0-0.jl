#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.TurbulenceClosures
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Interpolation
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics: total_energy, air_density
using ClimateMachine.VariableTemplates

using Distributions: Uniform
using LinearAlgebra
using StaticArrays

using CLIMAParameters
using CLIMAParameters.Planet:
    R_d, day, grav, cp_d, planet_radius, Omega, kappa_d, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMAParameters
CLIMAParameters.Planet.Omega(::EarthParameterSet) = 0.0
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 6.371e6 / 125.0
CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 1e5

"""
    cubedshelltopowarp(a, b, c, R = max(abs(a), abs(b), abs(c)))

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the equiangular gnomonic grid proposed by
[Ronchi1996](@cite). Then, warp surface points to "lift" radial coordinates
to represent some topographical profile.
"""
function cubedshelltopowarp(a, b, c, R = max(abs(a), abs(b), abs(c)); 
                            r_inner = _planet_radius, 
                            r_outer = _planet_radius + domain_height)
    
    function f(sR, ξ, η, faceid)
        R_m = π * 3/4
        h0 = 2000
        ζ_m = π/16
        φ_m = 0
        λ_m = π*3/2

        X, Y = tan(π * ξ / 4), tan(π * η / 4)

        # Linear Decay Profile
        # Δ = (r_outer - abs(sR))/(r_outer-r_inner)
        Δ = 1.0 

        # Angles
        mR = sR
        if faceid == 1
            λ = atan(X)                     # longitude 
            φ = atan(cos(λ)/Y)              # latitude

        elseif faceid == 2
            λ = atan(-1/X) 
            φ = atan(sin(λ)/Y)
        elseif faceid == 3
            λ = atan(X) 
            φ = atan(-cos(λ)/Y)
        elseif faceid == 4
            λ = atan(-1/X) 
            φ = atan(-sin(λ)/Y)
        elseif faceid == 5
            λ = atan(-X/Y)
            φ = atan(X/sin(λ))
        elseif faceid == 6
            λ = atan(X/Y)
            φ = atan(-X/sin(λ)) 
        end

        r_m = acos(sin(φ_m)*sin(φ)+cos(φ_m)*cos(φ)*cos(λ-λ_m))
        if r_m < R_m
            zs = 0.5*h0*(1+cos(π*r_m/R_m)) * cos(π*r_m/ζ_m) * cos(π*r_m/ζ_m)
        else
            zs = 0.0
        end

        mR = sign(sR)*( abs(sR) + zs*Δ )
        # mR = sign(sR)*( abs(sR) + zs )

        δ = 1 + X^2 + Y^2
        x1 = mR / sqrt(δ)
        x2, x3 = X * x1, Y * x1
        x1, x2, x3
    end
    fdim = argmax(abs.((a, b, c)))
    
    if fdim == 1 && a < 0
        faceid = 1 
        # (-R, *, *) : Face I from Ronchi, Iacono, Paolucci (1996)
        x1, x2, x3 = f(-R, b / a, c / a, faceid)
    elseif fdim == 2 && b < 0
        faceid = 2
        # ( *,-R, *) : Face II from Ronchi, Iacono, Paolucci (1996)
        x2, x1, x3 = f(-R, a / b, c / b, faceid)
    elseif fdim == 1 && a > 0
        faceid = 3
        # ( R, *, *) : Face III from Ronchi, Iacono, Paolucci (1996)
        x1, x2, x3 = f(R, b / a, c / a, faceid)
    elseif fdim == 2 && b > 0
        faceid = 4 
        # ( *, R, *) : Face IV from Ronchi, Iacono, Paolucci (1996)
        x2, x1, x3 = f(R, a / b, c / b, faceid)
    elseif fdim == 3 && c > 0
        faceid = 5 
        # ( *, *, R) : Face V from Ronchi, Iacono, Paolucci (1996)
        x3, x2, x1 = f(R, b / c, a / c, faceid)
    elseif fdim == 3 && c < 0
        faceid = 6
        # ( *, *,-R) : Face VI from Ronchi, Iacono, Paolucci (1996)
        x3, x2, x1 = f(-R, b / c, a / c, faceid)
    else
        error("invalid case for cubedshellwarp: $a, $b, $c")
    end

    return x1, x2, x3
end

function init_nonhydrostatic_gravity_wave!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)

    # grid
    φ = latitude(bl, aux)
    λ = longitude(bl, aux)
    z = altitude(bl, aux)

    # parameters
    _grav::FT = grav(bl.param_set)
    _cp::FT = cp_d(bl.param_set)
    _Ω::FT = Omega(bl.param_set)
    _a::FT = planet_radius(bl.param_set)
    _R_d::FT = R_d(bl.param_set)
    _kappa::FT = kappa_d(bl.param_set)
    _p_eq::FT = MSLP(bl.param_set)

    N::FT = 0.01
    u_0::FT = 0.0
    G::FT = _grav^2 / N^2 / _cp
    T_eq::FT = 300
    Δθ::FT = 0.0
    d::FT = 5e3
    λ_c::FT = 2 * π / 3
    φ_c::FT = 0
    L_z::FT = 20e3

    # initial velocity profile (we need to transform the vector into the Cartesian
    # coordinate system)
    u_sphere = SVector{3, FT}(u_0 * cos(φ), 0, 0)
    u_init = sphr_to_cart_vec(bl.orientation, u_sphere, aux)

    # background temperature
    T_s::FT =
        G +
        (T_eq - G) *
        exp(-u_0 * N^2 / 4 / _grav^2 * (u_0 + 2 * _Ω * _a) * (cos(2 * φ) - 1))
    T_b::FT = G * (1 - exp(N^2 / _grav * z)) + T_s * exp(N^2 / _grav * z)

    # pressure
    p_s::FT =
        _p_eq *
        exp(u_0 / 4 / G / _R_d * (u_0 + 2 * _Ω * _a) * (cos(2 * φ) - 1)) *
        (T_s / T_eq)^(1 / _kappa)
    p::FT = p_s * (G / T_s * exp(-N^2 / _grav * z) + 1 - G / T_s)^(1 / _kappa)

    # background potential temperature
    θ_b::FT = T_b * (_p_eq / p)^_kappa

    # potential temperature perturbation
    r::FT = _a * acos(sin(φ_c) * sin(φ) + cos(φ_c) * cos(φ) * cos(λ - λ_c))
    s::FT = d^2 / (d^2 + r^2)
    θ′::FT = Δθ * s * sin(2 * π * z / L_z)

    # temperature perturbation
    T′::FT = θ′ * (p / _p_eq)^_kappa

    # temperature
    T::FT = T_b + T′

    # density
    ρ = air_density(bl.param_set, T_b, p)

    # potential & kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)
    e_kin::FT = 0.5 * sum(abs2.(u_init))

    state.ρ = ρ
    state.ρu = ρ * u_init
    state.ρe = ρ * total_energy(bl.param_set, e_kin, e_pot, T)

    nothing
end

function config_nonhydrostatic_gravity_wave(FT, poly_order, resolution)
    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(300), FT(100), FT(27.5e3))
    ref_state = HydrostaticState(temp_profile_ref)

    _planet_radius = FT(planet_radius(param_set))
    domain_height::FT = 10e3               # distance between surface and top of atmosphere (m)

    # Set up the atmosphere model
    exp_name = "NonhydrostaticGravityWave"

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_prognostic = init_nonhydrostatic_gravity_wave!,
        ref_state = ref_state,
        turbulence = ConstantKinematicViscosity(FT(0)),
        moisture = DryModel(),
        source = (Gravity(),),
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_nonhydrostatic_gravity_wave!;
        model = model,
        meshwarp = setmax(cubedshelltopowarp,                   ## Function
                          _planet_radius,                       ## Domain inner radius
                          _planet_radius + domain_height)       ## Domain outer radius
    )

    return config
end

function setmax(f, r_inner, r_outer)
    function setmaxima(a,b,c)
        return f(a, b, c, max(abs(a), abs(b), abs(c));
                 r_inner = r_inner,
                 r_outer = r_outer)
    end
    return setmaxima
end

function config_diagnostics(FT, driver_config)
    interval = "40000steps" # chosen to allow a single diagnostics collection

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info
    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(10), FT(10), FT(100)) # in (deg, deg, m)
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )

    dgngrp = setup_atmos_default_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )

    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = 4                           # discontinuous Galerkin polynomial order
    n_horz = 12                               # horizontal element number
    n_vert = 5                               # vertical element number
    timestart = FT(0)                        # start time (s)
    timeend = FT(0.5)                       # end time (s)

    # Set up driver configuration
    driver_config =
        config_nonhydrostatic_gravity_wave(FT, poly_order, (n_horz, n_vert))

    # Set up experiment
    CFL = FT(0.4)
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        CFL_direction = HorizontalDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    filterorder = 64
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            solver_config.dg.grid,
            filter,
            state_auxiliary = solver_config.dg.state_auxiliary,
        )
        nothing
    end

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )
end

main()
