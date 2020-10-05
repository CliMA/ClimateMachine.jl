using Random
using StaticArrays
using Test
using Printf
using MPI

using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.Diagnostics
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.VariableTemplates
using CLIMA.VTK
using CLIMA.Atmos: vars_state, vars_aux

using CLIMA.Parameters
using CLIMA.UniversalConstants
const clima_dir = dirname(pathof(CLIMA))
include(joinpath(clima_dir, "..", "Parameters", "Parameters.jl"))
using CLIMA.Parameters.Planet
param_set = ParameterSet()

# ------------------------ Description ------------------------- #
# 1) Dry Rising Bubble (circular potential temperature perturbation)
# 2) Boundaries - `All Walls` : Impenetrable(FreeSlip())
#                               Laterally periodic
# 3) Domain - 20000m[horizontal] x 10000m[vertical] (2-dimensional)
# 4) Timeend - 1000s
# 5) Mesh Aspect Ratio (Effective resolution) 2:1
# 7) Overrides defaults for
#               `init_on_cpu`
#               `solver_type`
#               `sources`
#               `C_smag`
# 8) Default settings can be found in `src/Driver/Configurations.jl`
# ------------------------ Description ------------------------- #
function init_risingbubble!(bl, state, aux, (x, y, z), t)
    FT = eltype(state)
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    γ::FT = c_p / c_v
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)

    xc::FT = 10000
    zc::FT = 3000
    r = sqrt((x - xc)^2 + (z - zc)^2)
    rc::FT = 2000
    θ_ref::FT = 300
    Δθ::FT = 0

    if r <= rc
        Δθ = FT(2) * cospi(0.5*r/rc)^2
    end

    #Perturbed state:
    θ = θ_ref + Δθ # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas) # density
    q_tot = FT(0)
    ts = LiquidIcePotTempSHumEquil(θ, ρ, q_tot, bl.param_set)
    q_pt = PhasePartition(ts)

    ρu = SVector(FT(0), FT(0), FT(0))

    #State (prognostic) variable assignment
    e_kin = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot
    state.moisture.ρq_tot = ρ * q_pt.tot
end

function config_risingbubble(FT, N, resolution, xmin, xmax, ymax, zmax, hm, a)

    # Choose explicit solver
    solver_type = MultirateInfinitesimalStep
    fast_solver_type = StormerVerlet

    if solver_type==MultirateInfinitesimalStep
        if fast_solver_type==StormerVerlet
            ode_solver = CLIMA.MISSolverType(
                fast_model = AtmosAcousticGravityLinearModelSplit,
                mis_method = MIS2,
                fast_method = (dg,Q) -> StormerVerlet(dg, [1,5], 2:4, Q),
                nsubsteps = (15,),
            )
        elseif fast_solver_type==StrongStabilityPreservingRungeKutta
            ode_solver = CLIMA.MISSolverType(
                fast_model = AtmosAcousticGravityLinearModel,
                mis_method = MIS2,
                fast_method = SSPRK33ShuOsher,
                nsubsteps = (45,),
            )
        elseif fast_solver_type==MultirateInfinitesimalStep
            ode_solver = CLIMA.MISSolverType(
                fast_model = AtmosAcousticGravityLinearModel,
                mis_method = MIS2,
                fast_method = (dg, Q, nsubsteps) -> MultirateInfinitesimalStep(
                    :MISKWRK43,
                    dg,
                    (dgi,Qi) -> StormerVerlet(dgi, [1,5], 2:4, Qi),
                    Q,
                    nsubsteps = nsubsteps,
                ),
                nsubsteps = (12,2),
                hivi_splitting = true
            )
        elseif fast_solver_type==MultirateRungeKutta
            ode_solver = CLIMA.MISSolverType(
                fast_model = AtmosAcousticGravityLinearModel,
                mis_method = MIS2,
                fast_method = (dg,Q,nsubsteps) -> MultirateRungeKutta(
                    :LSRK144NiegemannDiehlBusch,
                    dg,
                    Q, nsubsteps=nsubsteps,
                ),
                steps = (12,4),
                hivi_splitting = true,
            )
        end
        Δt = FT(0.4)
    elseif solver_type==StrongStabilityPreservingRungeKutta
        ode_solver = CLIMA.ExplicitSolverType(solver_method = SSPRK33ShuOsher)
        #ode_solver = CLIMA.ExplicitSolverType(solver_method = SSPRK34SpiteriRuuth)
        Δt = FT(0.1)
    end

    # Set up the model
    C_smag = FT(0.23)
    ref_state = HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0))
        #HydrostaticState(IsothermalProfile(FT(T_0)),FT(0))
        #HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0))
    model = AtmosModel{FT}(
        AtmosLESConfigType;
        turbulence = SmagorinskyLilly{FT}(C_smag), #AnisoMinDiss{FT}(1),
        source = (Gravity(),),
        ref_state = ref_state,
        init_state = init_risingbubble!,
        param_set = param_set,
    )

    # Problem configuration
    function agnesiWarp(x,y,z)
        h=(hm*a^2)/((x-0.5*(xmin+xmax))^2+a^2)
        return x,y,zmax*(z+h)/(zmax+h)
    end
    config = CLIMA.AtmosLESConfiguration(
        "DryRisingBubble",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        init_risingbubble!,
        solver_type = ode_solver,
        model = model,
        meshwarp = agnesiWarp,
        #boundary = ((0, 0), (0, 0), (0, 0)),
        #periodicity = (false, true, false),
    )
    return config, Δt
end

function config_diagnostics(driver_config)
    interval = 10000 # in time steps
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return CLIMA.DiagnosticsConfiguration([dgngrp])
end

function main()
    CLIMA.init()

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 2
    # Domain resolution and size
    Δx = FT(125)
    Δy = FT(125)
    Δz = FT(125)
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmin = FT(0)
    xmax = FT(20000)
    ymax = FT(1000)
    zmax = FT(10000)
    # Mountain parameters
    hm = FT(1000)
    a = FT(10000)
    # Simulation time
    t0 = FT(0)
    timeend = FT(1000)

    # Courant number
    CFL = FT(20)

    driver_config, Δt = config_risingbubble(FT, N, resolution, xmin, xmax, ymax, zmax, hm, a)
    solver_config = CLIMA.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_dt = Δt,
        Courant_number = CFL,
    )
    dgn_config = config_diagnostics(driver_config)

    # User defined filter (TMAR positivity preserving filter)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    nvtk=timeend/(Δt*100)
    vtk_step = 0
    cbvtk = GenericCallbacks.EveryXSimulationSteps(nvtk)  do (init=false)
        mkpath("./vtk-rtb/")
        outprefix = @sprintf("./vtk-rtb/risingBubbleBryanSplitMountain_mpirank%04d_step%04d",
                         MPI.Comm_rank(driver_config.mpicomm), vtk_step)
        writevtk(outprefix, solver_config.Q, solver_config.dg,
            flattenednames(vars_state(driver_config.bl,FT)),
            solver_config.dg.auxstate,
            flattenednames(vars_aux(driver_config.bl,FT)))
        vtk_step += 1
        nothing
     end

    # Invoke solver (calls solve! function for time-integrator)
    result = CLIMA.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbvtk,),
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()
