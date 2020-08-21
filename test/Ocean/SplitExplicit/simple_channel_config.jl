include("split_explicit.jl")

import ClimateMachine.Ocean.SplitExplicit01:
    ocean_init_aux!,
    ocean_init_state!,
    ocean_boundary_state!,
    CoastlineFreeSlip,
    CoastlineNoSlip,
    OceanFloorFreeSlip,
    OceanFloorLinearDrag,
    OceanFloorNoSlip,
    OceanSurfaceNoStressNoForcing,
    OceanSurfaceStressNoForcing,
    OceanSurfaceNoStressForcing,
    OceanSurfaceStressForcing

function SplitConfig(
    name,
    resolution,
    dimensions,
    modelparams,
    problemparams,
    dt_slow;
    solver = SplitExplicitLSRK2nSolver,
)
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()

    N, Nˣ, Nʸ, Nᶻ = resolution
    Lˣ, Lʸ, H = dimensions

    xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
    yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
    zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

    brickrange_2D = (xrange, yrange)
    topl_2D = BrickTopology(
        mpicomm,
        brickrange_2D,
        periodicity = (true, false),
        boundary = ((0, 0), (1, 1)),
    )
    grid_2D = DiscontinuousSpectralElementGrid(
        topl_2D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    brickrange_3D = (xrange, yrange, zrange)
    topl_3D = StackedBrickTopology(
        mpicomm,
        brickrange_3D;
        periodicity = (true, false, false),
        boundary = ((0, 0), (1, 1), (2, 3)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    problem = EddyingChannel{FT}(Lˣ, Lʸ, H, problemparams...)

    cʰ, νʰ, νᶻ, κʰ, κᶻ, κᶜ, fₒ = modelparams
    add_fast_substeps = 2
    numImplSteps = 5
    numImplSteps > 0 ? ivdc_dt = dt_slow / FT(numImplSteps) : ivdc_dt = dt_slow
    model_3D = OceanModel{FT}(
        problem,
        grav = grav(param_set),
        cʰ = FT(cʰ),
        νʰ = FT(νʰ),
        νᶻ = FT(νᶻ),
        κʰ = FT(κʰ),
        κᶻ = FT(κᶻ),
        κᶜ = FT(κᶜ),
        fₒ = FT(fₒ),
        add_fast_substeps = add_fast_substeps,
        numImplSteps = numImplSteps,
        ivdc_dt = ivdc_dt,
    )

    model_2D = BarotropicModel(model_3D)

    vert_filter = CutoffFilter(grid_3D, polynomialorder(grid_3D) - 1)
    exp_filter = ExponentialFilter(grid_3D, 1, 8)

    flowintegral_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.FlowIntegralModel(model_3D),
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    tendency_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.TendencyIntegralModel(model_3D),
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    conti3d_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.Continuity3dModel(model_3D),
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )
    conti3d_Q = init_ode_state(conti3d_dg, FT(0); init_on_cpu = true)

    ivdc_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.IVDCModel(model_3D),
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        direction = VerticalDirection(),
    )
    # Not sure this is needed since we set values later,
    # but we'll do it just in case!
    ivdc_Q = init_ode_state(ivdc_dg, FT(0); init_on_cpu = true)
    ivdc_RHS = init_ode_state(ivdc_dg, FT(0); init_on_cpu = true)

    ivdc_bgm_solver = BatchedGeneralizedMinimalResidual(
        ivdc_dg,
        ivdc_Q;
        max_subspace_size = 10,
    )

    modeldata = (
        vert_filter = vert_filter,
        exp_filter = exp_filter,
        flowintegral_dg = flowintegral_dg,
        tendency_dg = tendency_dg,
        conti3d_dg = conti3d_dg,
        conti3d_Q = conti3d_Q,
        ivdc_dg = ivdc_dg,
        ivdc_Q = ivdc_Q,
        ivdc_RHS = ivdc_RHS,
        ivdc_bgm_solver = ivdc_bgm_solver,
    )

    return SplitConfig(
        name,
        model_3D,
        model_2D,
        grid_3D,
        grid_2D,
        modeldata,
        solver,
        mpicomm,
        ArrayType,
    )
end

struct EddyingChannel{T} <: AbstractOceanProblem
    Lˣ::T
    Lʸ::T
    H::T
    h::T # relaxation e-folding length
    efl::T # for sponge relaxation
    σʳ::T # relaxation time for sponge relaxation
    τₒ::T
    λʳ::T
    θᴱ::T
    λᴰ::T # bottom drag coefficient
end

@inline function ocean_boundary_state!(
    m::OceanModel,
    p::EddyingChannel,
    bctype,
    x...,
)
    if bctype == 1
        ocean_boundary_state!(m, CoastlineFreeSlip(), x...)
    elseif bctype == 2
        ocean_boundary_state!(m, OceanFloorLinearDrag(), x...)
    elseif bctype == 3
        ocean_boundary_state!(m, OceanSurfaceNoStressNoForcing(), x...)
    end
end

@inline function ocean_boundary_state!(
    m::Continuity3dModel,
    p::EddyingChannel,
    bctype,
    x...,
)
    #if bctype == 1
    ocean_boundary_state!(m, CoastlineFreeSlip(), x...)
    #end
end

@inline function ocean_boundary_state!(
    m::BarotropicModel,
    p::EddyingChannel,
    bctype,
    x...,
)
    return ocean_boundary_state!(m, CoastlineFreeSlip(), x...)
end

function ocean_init_aux!(m::OceanModel, p::EddyingChannel, A, geom)
    FT = eltype(A)
    @inbounds A.y = geom.coord[2]
    @inbounds A.z = geom.coord[3]

    # not sure if this is needed but getting weird intialization stuff
    A.w = -0
    A.pkin = -0
    A.wz0 = -0
    A.u_d = @SVector [-0, -0]
    A.ΔGu = @SVector [-0, -0]

    return nothing
end

# A is Filled afer the state
function ocean_init_aux!(m::BarotropicModel, P::EddyingChannel, A, geom)
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

init_temp(θᴱ, Lʸ, h, H, y, z) =
    0.5 * θᴱ * (1 - cos(π * y / Lʸ)) * (exp(z / h) - exp(-H / h)) /
    (1 - exp(-H / h))
init_temp(θᴱ, h, H, z) = θᴱ * (exp(z / h) - exp(-H / h)) / (1 - exp(-H / h))

function ocean_init_state!(p::EddyingChannel, Q, A, coords, t)
    Q.u = @SVector [-0, -0]
    Q.η = -0
    # Q.θ = init_temp(p.θᴱ, p.Lʸ, p.h, p.H, A.y, A.z)
    Q.θ = init_temp(p.θᴱ, p.h, p.H, A.z)

    return nothing
end

@inline function velocity_flux(p::EddyingChannel, y, ρ)
    Σ = (p.Lʸ^2 / 32)
    term1 = exp(-(y - p.Lʸ / 2)^2 / Σ)
    term2 = exp(-(p.Lʸ / 2)^2 / Σ)
    # return -(p.τₒ / ρ) * (term1 - term2)
    return eltype(ρ)(0)
end

@inline function temperature_flux(p::EddyingChannel, y, θ)
    θʳ = p.θᴱ * y / p.Lʸ

    # return p.λʳ * (θʳ - θ)
    return eltype(θ)(0)
end

@inline function sponge_relaxation(m::OceanModel, p::EddyingChannel, S, Q, A)
    θʳ = init_temp(p.θᴱ, p.Lʸ, p.h, p.H, A.y, A.z)
    # θʳ = init_temp(p.θᴱ, p.h, p.H, A.z)
    # S.θ = p.σʳ * (θʳ - Q.θ) * exp((A.y - p.Lʸ) / p.efl)
    S.θ = 0

    return nothing
end
