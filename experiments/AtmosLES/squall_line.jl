using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra
using DelimitedFiles
using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using CLIMAParameters
using CLIMAParameters.Planet
using CLIMAParameters.Atmos.Microphysics

struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet <: AbstractIceParameterSet end
struct RainParameterSet <: AbstractRainParameterSet end
struct SnowParameterSet <: AbstractSnowParameterSet end
struct MicropysicsParameterSet{L, I, R, S} <: AbstractMicrophysicsParameterSet
    liq::L
    ice::I
    rain::R
    snow::S
end
struct EarthParameterSet{M} <: AbstractEarthParameterSet
    microphys::M
end

const microphys = MicropysicsParameterSet(
    LiquidParameterSet(),
    IceParameterSet(),
    RainParameterSet(),
    SnowParameterSet(),
)
const param_set = EarthParameterSet(microphys)

using Dierckx
ClimateMachine.init(parse_clargs = true)

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
    E =
        ρ *
        total_energy(bl.param_set, e_kin, e_pot, T, PhasePartition(FT(data_q)))
    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, FT(0))
    state.ρe = E
    state.moisture.ρq_tot = ρ * data_q
    state.moisture.ρq_liq = FT(0)
    state.moisture.ρq_ice = FT(0)
    return nothing
end

function read_sounding()
    #read in the original squal sounding
    fsounding = open(joinpath(@__DIR__, "../sounding_gabersek.dat"))
    #fsounding  = open(joinpath(@__DIR__, "../sounding_gabersek_3deg_warmer.dat"))
    sounding = readdlm(fsounding)
    close(fsounding)
    (nzmax, ncols) = size(sounding)
    if nzmax == 0
        error("SOUNDING ERROR: The Sounding file is empty!")
    end
    return (sounding, nzmax, ncols)
end

function spline_int()

    # ----------------------------------------------------
    # GET DATA FROM INTERPOLATED ARRAY ONTO VECTORS
    # This driver accepts data in 6 column format
    # ----------------------------------------------------
    (sounding, _, ncols) = read_sounding()

    # WARNING: Not all sounding data is formatted/scaled
    # the same. Care required in assigning array values
    # height theta qv    u     v     pressure
    zinit, tinit, qinit, u_init, v_init, pinit = sounding[:, 1],
    sounding[:, 2],
    0.001 .* sounding[:, 3],
    sounding[:, 4],
    sounding[:, 5],
    sounding[:, 6]
    spl_tinit = Spline1D(zinit, tinit; k = 1)
    spl_qinit = Spline1D(zinit, qinit; k = 1)
    spl_uinit = Spline1D(zinit, u_init; k = 1)
    spl_vinit = Spline1D(zinit, v_init; k = 1)
    spl_pinit = Spline1D(zinit, pinit; k = 1)
    return spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit#, spl_rhoinit, spl_ppiinit, spl_thetainit
end

function config_squall_line(FT, N, resolution, xmax, ymax, zmax, xmin, ymin)
    # Reference state
    (sounding, _, ncols) = read_sounding()

    zinit, tinit, qinit, u_init, v_init, pinit = sounding[:, 1],
    sounding[:, 2],
    0.001 .* sounding[:, 3],
    sounding[:, 4],
    sounding[:, 5],
    sounding[:, 6]
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
    @info T_min, T_s
    Γ_lapse = FT(9.81 / 1004)
    tvmax = T_s * (1 + 0.61 * qinit[1])
    deltatv = -(T_min - tvmax)
    tvmin = T_min * (1 + 0.61 * qinit[maxz])
    @info deltatv
    htv = 8000.0
    #T = DecayingTemperatureProfile(T_min, T_s, Γ_lapse)
    Tv = DecayingTemperatureProfile{FT}(param_set, tvmax, tvmin, htv)
    rel_hum = FT(0)
    ref_state = HydrostaticState(Tv, rel_hum)
    # Sponge
    c_sponge = FT(1.0)
    # Rayleigh damping
    u_relaxation = SVector(FT(0), FT(0), FT(0))
    zsponge = FT(20000.0)
    rayleigh_sponge =
        RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)
    viscous_sponge =  UpperAtmosSponge{FT}(zmax, zsponge, FT(1.0), 2, FT(5))
    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)
    LHF = FT(50)
    SHF = FT(10)
    ics = init_squall_line!

    source = (Gravity(), CreateClouds(), rayleigh_sponge)


    problem = AtmosProblem(
        boundarycondition = (AtmosBC(), AtmosBC()),
        init_state_prognostic = ics,
    )


    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        ref_state = ref_state,
        moisture = NonEquilMoist(),
	#viscoussponge = viscous_sponge,
        turbulence = SmagorinskyLilly{FT}(C_smag),#ConstantDynamicViscosity(FT(200),WithDivergence()),
        source = source,
    )

    ode_solver = ClimateMachine.IMEXSolverType()

    config = ClimateMachine.AtmosLESConfiguration(
        "Squall_line_test_constvisc",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_squall_line!,
        xmin = xmin,
        ymin = ymin,
        solver_type = ode_solver,
        model = model,
        periodicity = (true, true, false),
        boundary = ((2, 2), (2, 2), (1, 2)),
        #numerical_flux_first_order = RoeNumericalFlux(),
    )
    return config
end
function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δx = FT(1000)
    Δy = FT(1000)
    Δv = FT(300)
    resolution = (Δx, Δy, Δv)

    xmax = FT(30000)
    ymax = FT(5000)
    zmax = FT(24000)
    xmin = FT(-30000)
    ymin = FT(0)

    t0 = FT(0)
    timeend = FT(2* 84600)
    spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit = spline_int()
    Cmax = FT(0.4)
    driver_config =
        config_squall_line(FT, N, resolution, xmax, ymax, zmax, xmin, ymin)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        (spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit);
        init_on_cpu = true,
        Courant_number = Cmax,
    )
    #dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
            Filters.apply!(
	        solver_config.Q,
		("moisture.ρq_tot","moisture.ρq_liq","moisture.ρq_ice",)
		,solver_config.dg.grid,
		TMARFilter()
		)
	        nothing
	    end

    filterorder = 30
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(solver_config.Q, (2, 3, 4), solver_config.dg.grid, filter)
        nothing
    end
    cutoff = CutoffFilter(solver_config.dg.grid)
    cbcutoff = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(solver_config.Q, (2, 3, 4), solver_config.dg.grid, cutoff)
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        #diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )
end

main()
