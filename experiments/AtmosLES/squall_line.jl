using ArgParse
using Random
using StaticArrays
using NCDatasets
using Test
using DocStringExtensions
using LinearAlgebra
using DelimitedFiles
using Dierckx
using Printf

using ClimateMachine
using ClimateMachine.Atmos
using ArtifactWrappers
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using Thermodynamics.TemperatureProfiles
using Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws: vars_state, Prognostic

using CLIMAParameters
using CLIMAParameters.Planet
using CLIMAParameters.Atmos.Microphysics

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

"""
  Define initial conditions based on sounding data
"""
function init_squall_line!(problem, bl, state, aux, localgeo, t, args...)

    FT = eltype(state)

    spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit = args[1]
    # interpolate data
    (x, y, z) = localgeo.coord
    data_t = FT(spl_tinit(z))
    data_q = FT(spl_qinit(z))
    data_u = FT(spl_uinit(z))
    data_v = FT(spl_vinit(z))
    data_p = FT(spl_pinit(z))
    u = data_u
    v = data_v
    w = FT(0)
    if z >= 14000
        data_q = FT(0)
    end
    θ_c = 3.0
    rx = 10000.0
    ry = 1500.0
    rz = 1500.0
    xc = 0.5 * (FT(0) + FT(0))
    yc = 0.5 * (FT(0) + FT(5000))
    zc = 2000.0
    cylinder_flg = 0.0
    r = sqrt(
        (x - xc)^2 / rx^2 +
        cylinder_flg * (y - yc)^2 / ry^2 +
        (z - zc)^2 / rz^2,
    )
    Δθ = 0.0
    if r <= 1.0
        Δθ = θ_c * (cospi(0.5 * r))^2
    end
    θ_liq = data_t + Δθ
    ts = PhaseNonEquil_pθq(param_set, data_p, θ_liq, PhasePartition(FT(data_q)))
    T = air_temperature(ts)
    ρ = air_density(ts)
    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux)
    E = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, FT(0))
    state.energy.ρe = E

    state.moisture.ρq_tot = ρ * data_q
    if moisture_model(bl) isa NonEquilMoist
        state.moisture.ρq_liq = FT(0)
        state.moisture.ρq_ice = FT(0)
    end

    if precipitation_model(bl) isa RainModel
        state.precipitation.ρq_rai = FT(0)
    end
    if precipitation_model(bl) isa RainSnowModel
        state.precipitation.ρq_rai = FT(0)
        state.precipitation.ρq_sno = FT(0)
    end

    return nothing
end

"""
  Read the original squall sounding
"""
function read_sounding()

    # Artifact creation is not thread-safe
    #      https://github.com/JuliaLang/Pkg.jl/issues/1219
    # To avoid race conditions from multiple jobs running this
    # driver at the same time, we must store artifacts in a
    # separate folder.

    soundings_dataset = ArtifactWrapper(
        @__DIR__,
        isempty(get(ENV, "CI", "")),
        "soundings",
        ArtifactFile[ArtifactFile(
            url = "https://caltech.box.com/shared/static/rjnvt2dlw7etm1c7mmdfrkw5gnfds5lx.nc",
            filename = "sounding_gabersek.nc",
        ),],
    )
    data_folder = get_data_folder(soundings_dataset)
    fsounding = joinpath(data_folder, "sounding_gabersek.nc")
    sounding = Dataset(fsounding, "r")
    height = sounding["height [m]"][:]
    θ = sounding["theta [K]"][:]
    q_vap = sounding["qv [g kg⁻¹]"][:]
    u = sounding["u [m s⁻¹]"][:]
    v = sounding["v [m s⁻¹]"][:]
    p = sounding["pressure [Pa]"][:]
    close(sounding)
    return (height = height, θ = θ, q_vap = q_vap, u = u, v = v, p = p)
end

"""
  Get data from interpolated array onto vectors.
  Accepts data in 6 column format
"""
function spline_int()

    nt = read_sounding()

    # WARNING: Not all sounding data is formatted/scaled the same.
    spl_tinit = Spline1D(nt.height, nt.θ; k = 1)
    spl_qinit = Spline1D(nt.height, nt.q_vap .* 0.001; k = 1)
    spl_uinit = Spline1D(nt.height, nt.u; k = 1)
    spl_vinit = Spline1D(nt.height, nt.v; k = 1)
    spl_pinit = Spline1D(nt.height, nt.p; k = 1)
    return (
        t = spl_tinit,
        q = spl_qinit,
        u = spl_uinit,
        v = spl_vinit,
        p = spl_pinit,
    )
end

function config_squall_line(
    FT,
    N,
    resolution,
    xmax,
    ymax,
    zmax,
    xmin,
    ymin,
    moisture_model = "nonequilibrium",
    precipitation_model = "rainsnow",
)
    # Reference state
    nt = read_sounding()

    zinit = nt.height
    tinit = nt.θ
    qinit = nt.q_vap .* 0.001
    u_init = nt.u
    v_init = nt.v
    p_init = nt.p

    maxz = length(zinit)
    thinit = zeros(maxz)
    piinit = zeros(maxz)
    thinit[1] = tinit[1] / (1 + 0.61 * qinit[1])
    piinit[1] = 1
    for k in 2:maxz
        thinit[k] = tinit[k] / (1 + 0.61 * qinit[k])
        piinit[k] =
            piinit[k - 1] -
            9.81 / (1004 * 0.5 * (tinit[k] + tinit[k - 1])) *
            (zinit[k] - zinit[k - 1])
    end
    T_min = FT(thinit[maxz] * piinit[maxz])
    T_s = FT(thinit[1] * piinit[1])
    Γ_lapse = FT(9.81 / 1004)
    tvmax = T_s * (1 + 0.61 * qinit[1])
    deltatv = -(T_min - tvmax)
    tvmin = T_min * (1 + 0.61 * qinit[maxz])
    htv = 8000.0
    #T = DecayingTemperatureProfile(T_min, T_s, Γ_lapse)
    Tv = DecayingTemperatureProfile{FT}(param_set, tvmax, tvmin, htv)
    rel_hum = FT(0)
    ref_state = HydrostaticState(Tv, rel_hum)

    # Sponge
    c_sponge = FT(0.5)
    # Rayleigh damping
    u_relaxation = SVector(FT(0), FT(0), FT(0))
    zsponge = FT(15000.0)
    rayleigh_sponge =
        RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)

    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.18) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)
    LHF = FT(50)
    SHF = FT(10)
    ics = init_squall_line!

    source = (Gravity(), rayleigh_sponge)

    # moisture model and its sources
    if moisture_model == "equilibrium"
        moisture = EquilMoist(; maxiter = 20, tolerance = FT(1))
    elseif moisture_model == "nonequilibrium"
        source = (source..., CreateClouds())
        moisture = NonEquilMoist()
    else
        @warn @sprintf(
            """
%s: unrecognized moisture_model in source terms, using the defaults""",
            moisture_model,
        )
        source = (source..., CreateClouds())
        moisture = NonEquilMoist()
    end

    # precipitation model and its sources
    if precipitation_model == "noprecipitation"
        precipitation = NoPrecipitation()
        source = (source..., RemovePrecipitation(true))
    elseif precipitation_model == "rain"
        source = (source..., WarmRain_1M())
        precipitation = RainModel()
    elseif precipitation_model == "rainsnow"
        source = (source..., RainSnow_1M())
        precipitation = RainSnowModel()
    else
        @warn @sprintf(
            """
%s: unrecognized precipitation_model in source terms, using the defaults""",
            precipitation_model,
        )
        source = (source..., RainSnow_1M())
        precipitation = RainSnowModel()
    end

    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = ref_state,
        moisture = moisture,
        precipitation = precipitation,
        turbulence = SmagorinskyLilly{FT}(C_smag),
    )

    problem = AtmosProblem(
        boundaryconditions = (AtmosBC(physics), AtmosBC(physics)),
        init_state_prognostic = ics,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        physics;
        problem = problem,
        source = source,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "Squall_line",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_squall_line!,
        xmin = xmin,
        ymin = ymin,
        model = model,
        periodicity = (true, true, false),
        boundary = ((2, 2), (2, 2), (1, 2)),
    )
    return config
end

function config_diagnostics(driver_config, boundaries, resolution)

    interval = "3600ssecs"

    dgngrp_profiles = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
    )

    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )
    dgngrp_state = setup_dump_state_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )
    dgngrp_aux = setup_dump_aux_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )

    return ClimateMachine.DiagnosticsConfiguration([
        dgngrp_profiles,
        dgngrp_state,
        dgngrp_aux,
    ])
end

function main()
    # TODO: this will move to the future namelist functionality
    squall_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(squall_args, "SQUALL_LINE")
    @add_arg_table! squall_args begin
        "--moisture-model"
        help = "specify cloud condensate model"
        metavar = "equilibrium|nonequilibrium"
        arg_type = String
        default = "nonequilibrium"
        "--precipitation-model"
        help = "specify precipitation model"
        metavar = "noprecipitation|rain|rainsnow"
        arg_type = String
        default = "rainsnow"
        "--check-asserts"
        help = "should asserts be checked at the end of the simulation"
        metavar = "yes|no"
        arg_type = String
        default = "no"
    end

    cl_args =
        ClimateMachine.init(parse_clargs = true, custom_clargs = squall_args)
    moisture_model = cl_args["moisture_model"]
    precipitation_model = cl_args["precipitation_model"]
    check_asserts = cl_args["check_asserts"]

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δx = FT(250)
    Δy = FT(1000)
    Δv = FT(200)
    resolution = (Δx, Δy, Δv)

    xmax = FT(30000)
    ymax = FT(5000)
    zmax = FT(24000)
    xmin = FT(-30000)
    ymin = FT(0)
    # for diagnostics
    boundaries = [
        xmin ymin FT(0)
        xmax ymax zmax
    ]

    t0 = FT(0)
    timeend = FT(9000)
    spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit = spline_int()
    Cmax = FT(0.4)

    # driver, solver and diagnostics configs
    driver_config = config_squall_line(
        FT,
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        xmin,
        ymin,
        moisture_model,
        precipitation_model,
    )
    ode_solver_type = ClimateMachine.IMEXSolverType()
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        (spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit);
        ode_solver_type = ode_solver_type,
        init_on_cpu = true,
        Courant_number = Cmax,
    )
    dgn_config = config_diagnostics(driver_config, boundaries, resolution)

    filter_vars = ("moisture.ρq_tot",)
    if moisture_model == "nonequilibrium"
        filter_vars = (filter_vars..., "moisture.ρq_liq", "moisture.ρq_ice")
    end
    if precipitation_model == "rain"
        filter_vars = (filter_vars..., "precipitation.ρq_rai")
    end
    if precipitation_model == "rainsnow"
        filter_vars =
            (filter_vars..., "precipitation.ρq_rai", "precipitation.ρq_rai")
    end

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            filter_vars,
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )

    if check_asserts == "yes"

        m = driver_config.bl
        Q = solver_config.Q
        ρ_ind = varsindex(vars_state(m, Prognostic(), FT), :ρ)

        if moisture_model == "equilibrium"
            ρq_tot_ind =
                varsindex(vars_state(m, Prognostic(), FT), :moisture, :ρq_tot)

            min_q_tot = minimum(abs.(
                Array(Q[:, ρq_tot_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_tot = maximum(abs.(
                Array(Q[:, ρq_tot_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))

            @info(min_q_tot, max_q_tot)

            # test that moisture exists and is not NaN
            @test !isnan(max_q_tot)

            # test that there is reasonable amount of moisture
            @test abs(max_q_tot) > FT(1e-2)
        end
        if moisture_model == "nonequilibrium"
            ρq_tot_ind =
                varsindex(vars_state(m, Prognostic(), FT), :moisture, :ρq_tot)
            ρq_liq_ind =
                varsindex(vars_state(m, Prognostic(), FT), :moisture, :ρq_liq)
            ρq_ice_ind =
                varsindex(vars_state(m, Prognostic(), FT), :moisture, :ρq_ice)

            min_q_tot = minimum(abs.(
                Array(Q[:, ρq_tot_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_tot = maximum(abs.(
                Array(Q[:, ρq_tot_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            min_q_liq = minimum(abs.(
                Array(Q[:, ρq_liq_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_liq = maximum(abs.(
                Array(Q[:, ρq_liq_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            min_q_ice = minimum(abs.(
                Array(Q[:, ρq_ice_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_ice = maximum(abs.(
                Array(Q[:, ρq_ice_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            @info(min_q_tot, max_q_tot)
            @info(min_q_liq, max_q_liq)
            @info(min_q_ice, max_q_ice)

            # test that moisture exists and is not NaN
            @test !isnan(max_q_tot)
            @test !isnan(max_q_liq)
            @test !isnan(max_q_ice)

            # test that there is reasonable amount of moisture
            @test abs(max_q_tot) > FT(1e-2)
            @test abs(max_q_liq) > FT(1e-3)
            @test abs(max_q_ice) > FT(1e-3)
        end

        if precipitation_model == "rain"
            ρq_rai_ind = varsindex(
                vars_state(m, Prognostic(), FT),
                :precipitation,
                :ρq_rai,
            )

            min_q_rai = minimum(abs.(
                Array(Q[:, ρq_rai_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_rai = maximum(abs.(
                Array(Q[:, ρq_rai_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))

            @info(min_q_rai, max_q_rai)

            # test that rain variable exists and is not NaN
            @test !isnan(max_q_rai)

            # test that there is reasonable amount of rain water...
            @test abs(max_q_rai) > FT(1e-4)
        end
        if precipitation_model == "rainsnow"
            ρq_rai_ind = varsindex(
                vars_state(m, Prognostic(), FT),
                :precipitation,
                :ρq_rai,
            )
            ρq_sno_ind = varsindex(
                vars_state(m, Prognostic(), FT),
                :precipitation,
                :ρq_sno,
            )
            min_q_rai = minimum(abs.(
                Array(Q[:, ρq_rai_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_rai = maximum(abs.(
                Array(Q[:, ρq_rai_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            min_q_sno = minimum(abs.(
                Array(Q[:, ρq_sno_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_sno = maximum(abs.(
                Array(Q[:, ρq_sno_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))

            @info(min_q_rai, max_q_rai)
            @info(min_q_sno, max_q_sno)

            # test that rain and snow variables exists and are not NaN
            @test !isnan(max_q_rai)
            @test !isnan(max_q_sno)

            # test that there is reasonable amount of precipitation
            @test abs(max_q_rai) > FT(1e-4)
            @test abs(max_q_sno) > FT(1e-6)
        end
    end
end

main()
