using Test
using ClimateMachine
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.BalanceLaws
using ClimateMachine.Ocean
using ClimateMachine.Ocean.SplitExplicit01
using ClimateMachine.Ocean.SplitExplicit01: AbstractOceanProblem
using ClimateMachine.Ocean.OceanProblems


using StaticArrays

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.Ocean.SplitExplicit01:
    ocean_init_aux!, ocean_init_state!, ocean_boundary_state!

struct SimpleBox{T} <: AbstractOceanProblem
    Lˣ::T
    Lʸ::T
    H::T
    τₒ::T
    λʳ::T
    θᴱ::T
end

function ocean_init_state!(p::SimpleBox, Q, A, coords, t)
    @inbounds y = coords[2]
    @inbounds z = coords[3]
    @inbounds H = p.H

    Q.u = @SVector [-0, -0]
    Q.η = -0
    Q.θ = (5 + 4 * cos(y * π / p.Lʸ)) * (1 + z / H)

    return nothing
end

function ocean_init_aux!(m::OceanModel, p::SimpleBox, A, geom)
    FT = eltype(A)
    @inbounds A.y = geom.coord[2]

    # not sure if this is needed but getting weird intialization stuff
    A.w = -0
    A.pkin = -0
    A.wz0 = -0
    A.u_d = @SVector [-0, -0]
    A.ΔGu = @SVector [-0, -0]

    return nothing
end

# A is Filled afer the state
function ocean_init_aux!(m::BarotropicModel, P::SimpleBox, A, geom)
    @inbounds A.y = geom.coord[2]

    A.Gᵁ = @SVector [-0, -0]
    A.U_c = @SVector [-0, -0]
    A.η_c = -0
    A.U_s = @SVector [-0, -0]
    A.η_s = -0
    A.Δu = @SVector [-0, -0]
    A.η_diag = -0
    A.Δη = -0

    return nothing
end

@inline function ocean_boundary_state!(
    m::Continuity3dModel,
    p::SimpleBox,
    bctype,
    x...,
)
    return ocean_boundary_state!(
        m,
        ClimateMachine.Ocean.SplitExplicit01.CoastlineNoSlip(),
        x...,
    )
end

@inline function ocean_boundary_state!(
    m::OceanModel,
    p::SimpleBox,
    bctype,
    x...,
)
    if bctype == 1
        ocean_boundary_state!(
            m,
            ClimateMachine.Ocean.SplitExplicit01.CoastlineNoSlip(),
            x...,
        )
    elseif bctype == 2
        ocean_boundary_state!(
            m,
            ClimateMachine.Ocean.SplitExplicit01.OceanFloorNoSlip(),
            x...,
        )
    elseif bctype == 3
        ocean_boundary_state!(
            m,
            ClimateMachine.Ocean.SplitExplicit01.OceanSurfaceStressForcing(),
            x...,
        )
    end
end

@inline function ocean_boundary_state!(
    m::BarotropicModel,
    p::SimpleBox,
    bctype,
    x...,
)
    return ocean_boundary_state!(
        m,
        ClimateMachine.Ocean.SplitExplicit01.CoastlineNoSlip(),
        x...,
    )
end

function config_simple_box(
    name,
    resolution,
    dimensions;
    dt_slow = 90.0 * 60.0,
    dt_fast = 240.0,
)
    problem = SimpleBox{FT}(dimensions..., 0.1, 10 // 86400, 10)

    add_fast_substeps = 2
    numImplSteps = 5
    numImplSteps > 0 ? ivdc_dt = dt_slow / FT(numImplSteps) : ivdc_dt = dt_slow
    model_3D = OceanModel{FT}(
        param_set,
        problem;
        cʰ = 1,
        κᶜ = FT(0.1),
        add_fast_substeps = add_fast_substeps,
        numImplSteps = numImplSteps,
        ivdc_dt = ivdc_dt,
    )

    N, Nˣ, Nʸ, Nᶻ = resolution
    resolution = (Nˣ, Nʸ, Nᶻ)

    config = ClimateMachine.OceanSplitExplicitConfiguration(
        name,
        N,
        resolution,
        param_set,
        model_3D;
        solver_type = SplitExplicitSolverType{FT}(dt_slow, dt_fast),
    )

    return config
end

function run_simple_box(driver_config, timespan; refDat = ())

    timestart, timeend = timespan
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_dt = driver_config.solver_type.dt_slow,
    )

    ## Create a callback to report state statistics for main MPIStateArrays
    ## every ntFreq timesteps.
    nt_freq = 1 # floor(Int, 1 // 10 * solver_config.timeend / solver_config.dt)
    cb = ClimateMachine.StateCheck.sccreate(
        [
            (solver_config.Q, "oce Q_3D"),
            (solver_config.dg.state_auxiliary, "oce aux"),
            (solver_config.dg.modeldata.Q_2D, "baro Q_2D"),
            (solver_config.dg.modeldata.dg_2D.state_auxiliary, "baro aux"),
        ],
        nt_freq;
        prec = 12,
    )

    result = ClimateMachine.invoke!(solver_config; user_callbacks = [cb])

    ## Check results against reference if present
    ClimateMachine.StateCheck.scprintref(cb)
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cb, refDat)
    end
end
