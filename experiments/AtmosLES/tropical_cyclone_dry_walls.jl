using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra
using Interpolations
using DelimitedFiles
using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.Diagnostics
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates
using Dierckx
using CLIMA.Parameters
const clima_dir = dirname(pathof(CLIMA))
include(joinpath(clima_dir, "..", "Parameters", "Parameters.jl"))

import CLIMA.DGmethods:
    vars_state,
    vars_aux,
    vars_integrals,
    vars_reverse_integrals,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_aux!,
    integral_set_aux!,
    reverse_integral_load_aux!,
    reverse_integral_set_aux!

import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: flux_diffusive!






function init_tc!(bl, state, aux, (x, y, z), args...)
    FT = eltype(state)
    spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_pres = args[2]
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
        θ_liq,
        pres,
        PhasePartition(FT(0)),
    )
    ρ = air_density(T, pres)
    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux)
    E = ρ * total_energy(e_kin, e_pot, T, PhasePartition(FT(0)))
    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, FT(0))
    state.ρe = E
    #state.moisture.ρq_tot = ρ * data_q
    #state.moisture.ρq_liq = FT(0)
    #state.moisture.ρq_ice = FT(0)
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
                    tinit[k] / (1 + 0.61 * qinit[k]) + anom[k] * exp(-(X[i]^2 + Y[j]^2) / (2 * RMW^2))
                theta[i, j, k] = thetav[i, j, k] #/ (1 + 0.61 * qinit[k])
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
        tvinit[k] = tinit[k] / (1 + 0.61 * qinit[k])
        piinit[k] =
            piinit[k - 1] -
            grav / (1004 * 0.5 * (tvinit[k] + tvinit[k - 1])) *
            (zinit[k] - zinit[k - 1])
    end
    for i in 1:length(X)
        for j in 1:length(X)
            pressure[i, j, maxz] = pinit[maxz]
            ppi[i, j, maxz] = piinit[maxz]
            temp[i, j, maxz] = tinit[maxz] * piinit[maxz]
            rho[i, j, maxz] =
                pinit[maxz] / (R_d * tvinit[maxz] * piinit[maxz])
        end
    end
    for i in 1:length(X)
        for j in 1:length(X)
            for k in (maxz - 1):-1:1
                ppi[i, j, k] =
                    ppi[i, j, k + 1] +
                    grav /
                    (1004 * 0.5 * (thetav[i, j, k] + thetav[i, j, k + 1])) *
                    (zinit[k + 1] - zinit[k])
                pressure[i, j, k] = MSLP * ppi[i, j, k]^(1004 / 287.04)
                temp[i, j, k] = theta[i, j, k] * ppi[i, j, k]
                rho[i, j, k] =
                    pressure[i, j, k] / (R_d * thetav[i, j, k] * ppi[i, j, k])
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
    piinit[k] = piinit[k-1] - grav / (1004 * 0.5 *(tinit[k] + tinit[k-1])) * (zinit[k] - zinit[k-1])
  end
    T_min = FT(thinit[maxz] * piinit[maxz])
    T_s = FT(thinit[1] * piinit[1])
    @info T_min, T_s
    Γ_lapse = FT(grav / cp_d)
    T = LinearTemperatureProfile(T_min, T_s, Γ_lapse)
    rel_hum = FT(0)
    ref_state = HydrostaticState(T, rel_hum)
    # Sponge
    c_sponge = FT(1)#0.00833
    # Rayleigh damping
    u_relaxation = SVector(FT(0), FT(0), FT(0))
    zsponge = FT(15000.0)
    rayleigh_sponge =
        RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)

    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)
    LHF = FT(50)
    SHF = FT(10)
    ics = init_tc!

    source = (Gravity(), rayleigh_sponge, Coriolis())
    model = AtmosModel{FT}(
        AtmosLESConfigType;
        ref_state = ref_state,
        moisture = DryModel{FT}(),
        turbulence = SmagorinskyLilly{FT}(C_smag),#ConstantViscosityWithDivergence{FT}(200),
        source = source,
        boundarycondition = (
            AtmosBC(
                momentum = (Impenetrable(DragLaw(
                    (state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu,
                ))),
                energy = BulkFormulationEnergy((state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu),
                #moisture = BulkFormulationMoisture(
                #    (state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu,
                #),
            ),
            AtmosBC(),
	    AtmosBC(energy = PrescribedTemperature((state⁻, aux⁻, t)-> 218),)
        ),
        init_state = ics,
        param_set = ParameterSet{FT}(),
    )

    ode_solver =
        CLIMA.IMEXSolverType()

    config = CLIMA.AtmosLESConfiguration(
        "CYCLONE_WALLS",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        init_tc!,
        xmin = xmin,
        ymin = ymin,
        solver_type = ode_solver,
        model = model,
	periodicity =(false,false,false),
	boundary = ((2,2),(2,2),(1,3)),
    )
    return config
end

function config_diagnostics(driver_config)
    interval = 10000 # in time steps
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return CLIMA.setup_diagnostics([dgngrp])
end

function main()
    CLIMA.init()

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
    driver_config = config_tc(FT, N, resolution, xmax, ymax, zmax, xmin, ymin)
    solver_config = CLIMA.setup_solver(
        t0,
        timeend,
        driver_config,
        (spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit, spl_pres),
        Courant_number = 0.35,
	init_on_cpu = true,
    )
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(
            solver_config.Q,
            (6, 7, 8),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    result = CLIMA.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        #user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )
end

main()
