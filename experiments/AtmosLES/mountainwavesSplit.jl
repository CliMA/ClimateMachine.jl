using ClimateMachine
ClimateMachine.init()

using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.MoistThermodynamics
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK
using ClimateMachine.Atmos: vars_state_conservative, vars_state_auxiliary

using StaticArrays
using Test
using Printf
using MPI

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

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
function init_gravitywave!(bl, state, aux, (x, y, z), t)
    FT = eltype(state)
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    γ::FT = c_p / c_v
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    N::FT = 0.01
    kappa::FT = R_gas/c_p

    UMax = FT(10)
    θ_ref::FT = 300

    θ = θ_ref * exp(z*N^2/_grav) # potential temperature
    p = p0 * (1 - _grav / (c_p * θ_ref * N^2 / _grav) * (1 - exp(-N^2 / _grav * z))) ^ (c_p / R_gas) # density
    ρ = p / ((p / p0) ^ kappa * R_gas * θ); # density
    q_tot = FT(0)
    ts = LiquidIcePotTempSHumEquil(bl.param_set, θ, ρ, q_tot)
    q_pt = PhasePartition(ts)

    ρu = SVector(UMax * ρ, FT(0), FT(0))

    #State (prognostic) variable assignment
    e_kin = 0.5 * UMax^2
    e_pot = gravitational_potential(bl.orientation, aux)
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot
    state.moisture.ρq_tot = ρ * q_pt.tot
end

function config_gravitywave(FT, N, resolution, xmin, xmax, ymax, zmax, hm, a)

    # Choose explicit solver
    solver_type = MultirateInfinitesimalStep
    fast_solver_type = StormerVerlet

    if solver_type==MultirateInfinitesimalStep
        if fast_solver_type==StormerVerlet
            ode_solver = ClimateMachine.MISSolverType(
                fast_model = AtmosAcousticGravityLinearModelSplit,
                mis_method = MIS2,
                fast_method = (dg,Q) -> StormerVerlet(dg, [1,5], 2:4, Q),
                nsubsteps = (70,),
            )
        elseif fast_solver_type==StrongStabilityPreservingRungeKutta
            ode_solver = ClimateMachine.MISSolverType(
                fast_model = AtmosAcousticGravityLinearModel,
                mis_method = MIS2,
                fast_method = SSPRK33ShuOsher,
                nsubsteps = (45,),
            )
        elseif fast_solver_type==MultirateInfinitesimalStep
            ode_solver = ClimateMachine.MISSolverType(
                fast_model = AtmosAcousticGravityLinearModel,
                mis_method = MIS2,
                fast_method = (dg, Q, nsteps) -> MultirateInfinitesimalStep(
                    :MISKWRK43,
                    dg,
                    (dgi,Qi) -> StormerVerlet(dgi, [1,5], 2:4, Qi),
                    Q,
                    nsteps = nsteps,
                ),
                nsubsteps = (45,7),
                hivi_splitting = true
            )
        elseif fast_solver_type==MultirateRungeKutta
            ode_solver = ClimateMachine.MISSolverType(
                fast_model = AtmosAcousticGravityLinearModel,
                mis_method = MIS2,
                fast_method = (dg,Q,nsteps) -> MultirateRungeKutta(
                    :LSRK144NiegemannDiehlBusch,
                    dg,
                    Q, nsteps=nsteps,
                ),
                nsubsteps = (45,15),
                hivi_splitting = true,
            )
        end
        Δt = FT(5.0)
    elseif solver_type==StrongStabilityPreservingRungeKutta
        ode_solver = ClimateMachine.ExplicitSolverType(solver_method = SSPRK33ShuOsher)
        #ode_solver = ClimateMachine.ExplicitSolverType(solver_method = SSPRK34SpiteriRuuth)
        Δt = FT(0.1)
    end

    # Set up the model
    C_smag = FT(0.23)
    ref_state = HydrostaticState(StableTemperatureProfile(FT(300),FT(1.e-2)), FT(0))
    #ref_state = HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0))
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = SmagorinskyLilly{FT}(C_smag), #AnisoMinDiss{FT}(1),
        source = (Gravity(),),
        ref_state = ref_state,
        init_state_conservative = init_gravitywave!,
    )

    # Problem configuration
    function agnesiWarp(x,y,z)
        h=(hm*a^2)/((x-0.5*(xmin+xmax))^2+a^2)
        return x,y,zmax*(z+h)/(zmax+h)
    end
    config = ClimateMachine.AtmosLESConfiguration(
        "GravityWave",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_gravitywave!,
        xmin = xmin,
        solver_type = ode_solver,
        model = model,
        meshwarp = agnesiWarp,
        #boundary = ((0, 0), (0, 0), (0, 0)),
        periodicity = (true, true, false),
    )
    return config, Δt
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    ClimateMachine.init()

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 2
    # Number of elements in each direction
    Ne = (100, 0, 78)
    # Domain extents
    xmin = FT(-20000)
    xmax = FT(20000)
    zmax = FT(15600)
    # Domain resolution is the average node spacing in the elements
    Δx = (xmax - xmin) / (N * Ne[1])
    Δz = FT(zmax) / (N * Ne[3])
    # Force the y resolution to be the same as the x with a single element
    Δy = Δx
    ymax = Δy * N
    resolution = (Δx, Δy, Δz)
    # Mountain parameters
    hm = FT(400)
    a = FT(1000) #FT(10000)
    # Simulation time
    t0 = FT(0)
    timeend = FT(2160.0)

    # Courant number
    CFL = FT(20)

    driver_config, Δt = config_gravitywave(FT, N, resolution, xmin, xmax, ymax, zmax, hm, a)
    solver_config = ClimateMachine.SolverConfiguration(
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

    nvtk=timeend/(Δt*72)
    vtk_step = 0
    cbvtk = GenericCallbacks.EveryXSimulationSteps(nvtk)  do (init=false)
        mkpath("./vtk-rtb/")
        outprefix = @sprintf("./vtk-rtb/mountainwavesSplit_mpirank%04d_step%04d",
                         MPI.Comm_rank(driver_config.mpicomm), vtk_step)
        writevtk(outprefix, solver_config.Q, solver_config.dg,
            flattenednames(vars_state_conservative(driver_config.bl,FT)),
            solver_config.dg.state_auxiliary,
            flattenednames(vars_state_auxiliary(driver_config.bl,FT)))
        vtk_step += 1
        nothing
     end

    # Invoke solver (calls solve! function for time-integrator)
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbvtk,),
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()
