using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra
using Interpolations
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

function init_tc!(problem, bl, state, aux, (x, y, z), t, args...)
    FT = eltype(state)
    spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_pres = args[1]
    # interpolate data
    data_t = FT(spl_tinit(z))
    data_q = FT(spl_qinit(z))
    data_u = FT(spl_uinit(x, y, z))
    data_v = FT(spl_vinit(x, y, z))
    data_p = FT(spl_pinit(z))
    pres = FT(spl_pres(x, y, z))
    #data_rho = FT(spl_rhoinit(x,y,z))
    #data_pi = FT(spl_ppiinit(x,y,z))
    #data_theta = FT(spl_thetainit(x,y,z))
    u = data_u
    v = data_v
    w = FT(0)
    t_anom = 1
    RMW = 50000
    anom = t_anom * exp(-(data_p - 40000)^2 / 2 / (11000^2))
    Δθ = anom * exp(-(x^2 + y^2) / (2 * RMW^2))
    θ_liq = data_t / (1 + 0.61 * data_q) + Δθ
    T = air_temperature_from_liquid_ice_pottemp_given_pressure(
        bl.param_set,
        θ_liq,
        pres,
        PhasePartition(FT(data_q)),
    )
    ρ = air_density(bl.param_set,T, pres)
    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux)
    E = ρ * total_energy(bl.param_set,e_kin, e_pot, T, PhasePartition(FT(data_q)))
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
    t_anom = 1
    RMW = 100000
    X = -810000:5000:810000
    Y = -810000:5000:810000
    theta = zeros(length(X), length(X), length(zinit))
    thetav = zeros(length(X), length(X), length(zinit))
    pressure = zeros(length(X), length(X), length(zinit))
    ppi = zeros(length(X), length(X), length(zinit))
    temp = zeros(length(X), length(X), length(zinit))
    rho = zeros(length(X), length(X), length(zinit))
    anom = zeros(length(zinit))
    for i in 1:length(X)
        for j in 1:length(X)
            for k in 1:length(zinit)
                anom[k] = t_anom * exp(-(pinit[k] - 40000)^2 / 2 / (11000^2))
                thetav[i, j, k] =
                    tinit[k]  + anom[k] * exp(-(X[i]^2 + Y[j]^2) / (2 * RMW^2))
                theta[i, j, k] = thetav[i, j, k] / (1 + 0.61 * qinit[k])
            end
        end
    end
    f = 5e-5
    RMW = 50000
    maxz = length(zinit)
    tvinit = zeros(maxz)
    piinit = zeros(maxz)
    tvinit[1] = tinit[1]
    piinit[1] = 1
    for k in 2:maxz
        tvinit[k] = tinit[k] 
        piinit[k] =
            piinit[k - 1] -
            9.81 / (1004 * 0.5 * (tvinit[k] + tvinit[k - 1])) *
            (zinit[k] - zinit[k - 1])
    end
    for i in 1:length(X)
        for j in 1:length(X)
            pressure[i, j, maxz] = pinit[maxz]
            ppi[i, j, maxz] = piinit[maxz]
            temp[i, j, maxz] = tinit[maxz] * piinit[maxz]
            rho[i, j, maxz] =
              pinit[maxz] / (287.04 * tvinit[maxz] * piinit[maxz])
        end
    end
    for i in 1:length(X)
        for j in 1:length(X)
            for k in (maxz - 1):-1:1
                ppi[i, j, k] =
                    ppi[i, j, k + 1] +
                    9.81 /
                    (1004 * 0.5 * (thetav[i, j, k] + thetav[i, j, k + 1])) *
                    (zinit[k + 1] - zinit[k])
                pressure[i, j, k] = 101325 * ppi[i, j, k]^(1004 / 287.04)
                temp[i, j, k] = theta[i, j, k] * ppi[i, j, k]
                rho[i, j, k] =
                    pressure[i, j, k] / (287.04 * thetav[i, j, k] * ppi[i, j, k])
            end
        end
    end
    Z = collect(zinit)
    knots = (X, Y, Z)
    gradpx = zeros(length(X), length(X), length(zinit))
    gradpy = zeros(length(X), length(X), length(zinit))


    uinit = zeros(length(X), length(X), length(zinit))
    vinit = zeros(length(X), length(X), length(zinit))
    for i in 1:length(X)
        for j in 1:length(X)
            for k in 1:length(zinit)
                if (i == 1) || (j == 1) || (i == length(X)) || (j == length(X))
                    uinit[i, j, k] = 0
                    vinit[i, j, k] = 0

                else
                    gradpx[i, j, k] =
                        (pressure[i + 1, j, k] - pressure[i, j, k]) / 5000
                    gradpy[i, j, k] =
                        (pressure[i, j + 1, k] - pressure[i, j, k]) / 5000
                    if (gradpx[i, j, k] == 0)
                        uinit[i, j, k] = 0
                    else
                        uinit[i, j, k] =
                            -1 / (f * rho[i, j, k]) * gradpy[i, j, k]
                    end
                    if (gradpy[i, j, k] == 0)
                        vinit[i, j, k] = 0
                    else
                        vinit[i, j, k] =
                            1 / (f * rho[i, j, k]) * gradpx[i, j, k]
                    end
                end
            end
        end
    end
    #------------------------------------------------------
    # GET SPLINE FUNCTION
    #------------------------------------------------------
    itp = interpolate(knots, pressure, Gridded(Linear()))
    spl_pres = itp
    #itp = interpolate(knots, temp, Gridded(Linear()))
    #spl_tinit = itp
    #itp = interpolate(knots, rho, Gridded(Linear()))
    #spl_rhoinit = itp
    #itp = interpolate(knots, ppi, Gridded(Linear()))
    #spl_ppiinit = itp
    #itp = interpolate(knots, theta, Gridded(Linear()))
    #spl_thetainit = itp
    #itp = interpolate(knots, thetav, Gridded(Linear()))
    #spl_thetavinit = itp
    itp = interpolate(knots, uinit, Gridded(Linear()))
    spl_uinit = itp
    itp = interpolate(knots, vinit, Gridded(Linear()))
    spl_vinit = itp


    spl_tinit = Spline1D(zinit, tinit; k = 1)
    spl_qinit = Spline1D(zinit, qinit; k = 1)
    #spl_uinit    = Spline1D(zinit, uinit; k=1)
    #spl_vinit    = Spline1D(zinit, vinit; k=1)
    spl_pinit = Spline1D(zinit, pinit; k = 1)
    return spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_pres#, spl_rhoinit, spl_ppiinit, spl_thetainit
end

function config_tc(FT, N, resolution, xmax, ymax, zmax, xmin, ymin)
    # Reference state
    (sounding, _, ncols) = read_sounding()

  zinit, tinit, qinit, u_init, v_init, pinit  =
      sounding[:, 1], sounding[:, 2], 0.001 .* sounding[:, 3], sounding[:, 4], sounding[:, 5], sounding[:, 6]
  maxz = length(zinit)
  thinit = zeros(maxz)
  piinit = zeros(maxz)
  thinit[1] = tinit[1]/(1+0.61*qinit[1])
  piinit[1] = 1
  for k in 2:maxz
    thinit[k] = tinit[k]/(1+0.61*qinit[k])
    piinit[k] = piinit[k-1] - 9.81 / (1004 * 0.5 *(tinit[k] + tinit[k-1])) * (zinit[k] - zinit[k-1])
  end
    T_min = FT(thinit[maxz] * piinit[maxz])
    T_s = FT(thinit[1] * piinit[1])
    @info T_min, T_s
    Γ_lapse = FT(9.81 / 1004)
    tvmax = T_s*(1+0.61*qinit[1])
    deltatv = -( T_min   - tvmax)
    tvmin = T_min*(1+0.61*qinit[maxz])
    @info deltatv
    htv = 8000.0
    #T = DecayingTemperatureProfile(T_min, T_s, Γ_lapse)
    Tv = DecayingTemperatureProfile{FT}(param_set,tvmax,tvmin,htv)
    rel_hum = FT(0)
    ref_state = HydrostaticState(Tv, rel_hum)
    # Sponge
    c_sponge = 0.000833
    # Rayleigh damping
    u_relaxation = SVector(FT(0), FT(0), FT(0))
    zsponge = FT(16000.0)
    rayleigh_sponge =
        RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)

    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)
    LHF = FT(50)
    SHF = FT(10)
    ics = init_tc!

    source = (Gravity(),rayleigh_sponge, Coriolis(), CreateClouds())
    
    
    problem = AtmosProblem(
        boundarycondition = (
		    AtmosBC(
                momentum = (Impenetrable(DragLaw(
                    (state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu,
                ))),
                energy = BulkFormulaEnergy((state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu, (state, aux, t) ->((T_s), state.moisture.ρq_tot)),
                moisture = BulkFormulaMoisture(
                    (state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu, (state, aux, t) -> state.moisture.ρq_tot,
                ),
            ),
            AtmosBC(),
            #AtmosBC(energy = PrescribedTemperature((state⁻, aux⁻, t)-> 218),)
        ),
        init_state_prognostic = ics,
    )

    
    model = AtmosModel{FT}(
        AtmosLESConfigType,
	param_set;
	problem = problem,
	ref_state = ref_state,
        moisture = NonEquilMoist(),
        turbulence = SmagorinskyLilly{FT}(C_smag),#ConstantViscosityWithDivergence{FT}(200),
        source = source,
    )

    ode_solver = ClimateMachine.IMEXSolverType()

    config = ClimateMachine.AtmosLESConfiguration(
        "CYCLONE_WALLS_30_moist",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
	param_set,
        init_tc!,
        xmin = xmin,
        ymin = ymin,
        solver_type = ode_solver,
        model = model,
	periodicity =(false,false,false),
	boundary = ((2,2),(2,2),(1,2)),
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
    Δh = FT(10000)
    Δv = FT(200)
    resolution = (Δh, Δh, Δv)

    xmax = FT(400000)
    ymax = FT(400000)
    zmax = FT(20000)
    xmin = FT(-400000)
    ymin = FT(-400000)

    t0 = FT(0)
    timeend = FT(86400)
    spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_pres =
        spline_int()
    Cmax = FT(0.2)
    driver_config = config_tc(FT, N, resolution, xmax, ymax, zmax, xmin, ymin)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        (spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_pres);
	init_on_cpu = true,
	Courant_number = Cmax
    )
    #dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(
            solver_config.Q,
            (6,7,8),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end
    filterorder = 30
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            (2,3,4),
            solver_config.dg.grid,
            filter,
        )
        nothing
    end
    cutoff = CutoffFilter(solver_config.dg.grid)
    cbcutoff = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            (2,3,4),
            solver_config.dg.grid,
            cutoff,
        )
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        #diagnostics_config = dgn_config,
        #user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )
end

main()
