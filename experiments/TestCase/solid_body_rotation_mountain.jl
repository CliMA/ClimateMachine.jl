#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.TurbulenceClosures
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Interpolation
using ClimateMachine.TemperatureProfiles
using ClimateMachine.VariableTemplates
using ClimateMachine.Thermodynamics: air_density, total_energy

using LinearAlgebra
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet: day, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function setmax(f, r_inner, r_outer)
    function setmaxima(a, b, c)
        return f(
            a,
            b,
            c,
            max(abs(a), abs(b), abs(c));
            r_inner = r_inner,
            r_outer = r_outer,
        )
    end
    return setmaxima
end

function cubedshelltopowarp(
    a,
    b,
    c,
    R = max(abs(a), abs(b), abs(c));
    r_inner = _planet_radius,
    r_outer = _planet_radius + domain_height,
)

    function f(sR, ξ, η, faceid)
        R_m = π * 3 / 4
        h0 = 2000
        ζ_m = π / 16
        φ_m = 0
        λ_m = π * 3 / 2

        X, Y = tan(π * ξ / 4), tan(π * η / 4)

        # Linear Decay Profile
        Δ = (r_outer - abs(sR)) / (r_outer - r_inner)
        δ = 1 + X^2 + Y^2

        # Angles
        # mR == modified radius 
        mR = sR
        if faceid == 1
            λ = atan(X)                     # longitude 
            φ = atan(cos(λ) * Y)              # latitude
        elseif faceid == 2
            λ = atan(X) + π / 2
            φ = atan(Y * cos(atan(X)))
        elseif faceid == 3
            λ = atan(X) + π
            φ = atan(Y * cos(atan(X)))
        elseif faceid == 4
            λ = atan(X) + (3 / 2) * π
            φ = atan(Y * cos(atan(X)))
        elseif faceid == 5
            λ = atan(X, -Y) + π
            φ = atan(1 / sqrt(δ - 1))
        elseif faceid == 6
            λ = atan(X, Y)
            φ = -atan(1 / sqrt(δ - 1))
        end

        r_m = acos(sin(φ_m) * sin(φ) + cos(φ_m) * cos(φ) * cos(λ - λ_m))
        if r_m < R_m
            zs =
                0.5 *
                h0 *
                (1 + cos(π * r_m / R_m)) *
                cos(π * r_m / ζ_m) *
                cos(π * r_m / ζ_m)
        else
            zs = 0.0
        end

        mR = sign(sR) * (abs(sR) + zs * Δ)

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


function init_solid_body_rotation!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)

    # initial velocity profile (we need to transform the vector into the Cartesian
    # coordinate system)
    u_0::FT = 0
    u_sphere = SVector{3, FT}(u_0, 0, 0)
    u_init = sphr_to_cart_vec(bl.orientation, u_sphere, aux)
    e_kin::FT = 0.5 * sum(abs2.(u_init))

    # Assign state variables
    state.ρ = aux.ref_state.ρ
    state.ρu = u_init
    state.ρe = aux.ref_state.ρe + state.ρ * e_kin

    nothing
end

function config_solid_body_rotation(FT, poly_order, resolution, ref_state)

    # Set up the atmosphere model
    exp_name = "SolidBodyRotation"
    domain_height::FT = 30e3 # distance between surface and top of atmosphere (m)

    _planet_radius = FT(planet_radius(param_set))

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_prognostic = init_solid_body_rotation!,
        ref_state = ref_state,
        turbulence = ConstantKinematicViscosity(FT(0)),
        #hyperdiffusion = DryBiharmonic(FT(8 * 3600)),
        moisture = DryModel(),
        source = (Gravity(), Coriolis()),
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_solid_body_rotation!;
        model = model,
        numerical_flux_first_order = RoeNumericalFlux(),
        meshwarp = setmax(
            cubedshelltopowarp,                   ## Function
            _planet_radius,                       ## Domain inner radius
            _planet_radius + domain_height,
        ),       ## Domain outer radius
    )

    return config
end

function main()
    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = 5                           # discontinuous Galerkin polynomial order
    n_horz = 8                               # horizontal element number
    n_vert = 4                               # vertical element number
    timestart::FT = 0                        # start time (s)
    timeend::FT = 7200

    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up driver configuration
    driver_config =
        config_solid_body_rotation(FT, poly_order, (n_horz, n_vert), ref_state)

    # Set up experiment
    ode_solver_type = ClimateMachine.IMEXSolverType(
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
        split_explicit_implicit = false,
        discrete_splitting = true,
    )

    CFL = FT(0.2) # target acoustic CFL number

    # time step is computed such that the horizontal acoustic Courant number is CFL
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        ode_solver_type = ode_solver_type,
        CFL_direction = HorizontalDirection(),
        diffdir = HorizontalDirection(),
    )

    # initialize using a different ref state (mega-hack)
    temp_profile_init =
        DecayingTemperatureProfile{FT}(param_set, FT(280), FT(230), FT(9e3))
    init_ref_state = HydrostaticState(temp_profile_init)

    init_driver_config = config_solid_body_rotation(
        FT,
        poly_order,
        (n_horz, n_vert),
        init_ref_state,
    )
    init_solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        init_driver_config,
        Courant_number = CFL,
        ode_solver_type = ode_solver_type,
        CFL_direction = HorizontalDirection(),
        diffdir = HorizontalDirection(),
    )

    # initialization
    solver_config.Q .= init_solver_config.Q

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    filterorder = 20
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            solver_config.dg.grid,
            filter,
            # filter perturbations from the initial state
            state_auxiliary = init_solver_config.dg.state_auxiliary,
        )
        nothing
    end

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbfilter,),
        check_euclidean_distance = false,
    )

    relative_error =
        norm(solver_config.Q .- init_solver_config.Q) /
        norm(init_solver_config.Q)
    @info "Relative error = $relative_error"
    @test relative_error < 1e-9
end

function config_diagnostics(FT, driver_config)
    interval = "0.5shours" # chosen to allow diagnostics every 30 simulated minutes

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info
    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(2), FT(2), FT(1000)) # in (deg, deg, m)
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

main()
