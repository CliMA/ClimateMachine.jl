using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.ODESolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.Mesh.Grids: min_node_distance

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

include("hyperdiffusion_model.jl")

struct ConstantHyperDiffusion{dim, dir, FT} <: HyperDiffusionProblem
    D::SMatrix{3, 3, FT, 9}
end

function nodal_init_state_auxiliary!(
    balance_law::HyperDiffusion,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.D = balance_law.problem.D
end

"""
    initial condition is given by ρ0 = sin(kx+ly+mz)
"""

function initial_condition!(
    problem::ConstantHyperDiffusion{dim, dir},
    state,
    aux,
    x,
    t,
) where {dim, dir}
    @inbounds begin
        k = SVector(1, 2, 3)
        kD = k * k' .* problem.D
        if dir === EveryDirection()
            c = sum(abs2, k[SOneTo(dim)]) * sum(kD[SOneTo(dim), SOneTo(dim)])
        elseif dir === HorizontalDirection()
            c =
                sum(abs2, k[SOneTo(dim - 1)]) *
                sum(kD[SOneTo(dim - 1), SOneTo(dim - 1)])
        elseif dir === VerticalDirection()
            c = k[dim]^2 * kD[dim, dim]
        end
        state.ρ = sin(dot(k[SOneTo(dim)], x[SOneTo(dim)])) * exp(-c * t)
    end
end

function run(
    mpicomm,
    ArrayType,
    dim,
    topl,
    N,
    FT,
    direction,
    τ
)

    grid = DiscontinuousSpectralElementGrid(
            topl,
            FloatType = FT,
            DeviceArray = ArrayType,
            polynomialorder = N,
        )
    dx = min_node_distance(grid)

    D = (dx/2)^4/2/τ * SMatrix{3, 3, FT}(1,0,0,0,1,0,0,0,1,) 

    model = HyperDiffusion{dim}(ConstantHyperDiffusion{dim, direction(), FT}(D))
    dg = DGModel(
            model,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            direction = direction(),
        )

    Q0 = init_ode_state(dg, FT(0))
    dt = dx^4 / 25 / sum(D)
    @info "time step" dt
    @info "Δ(horz)" dx
    # dt = outputtime / ceil(Int64, outputtime / dt) 

    rhs_diag = similar(Q0)
    dg(rhs_diag, Q0, nothing, 0)

    k = SVector(1, 2, 3)
    kD = k * k' .* D
        if direction === EveryDirection
            c = sum(abs2, k[SOneTo(dim)]) * sum(kD[SOneTo(dim), SOneTo(dim)])
        elseif direction === HorizontalDirection
            c =
                sum(abs2, k[SOneTo(dim - 1)]) *
                sum(kD[SOneTo(dim - 1), SOneTo(dim - 1)])
        elseif direction === VerticalDirection
            c = k[dim]^2 * kD[dim, dim]
        end
    rhs_anal = -c*Q0  

    # Q1_lsrk_diag = euclidean_distance(Q1_diag, Q1_lsrk)
    # Q1_form_diag = euclidean_distance(Q1_diag, Q1_form)
    # Q1_form_lsrk = euclidean_distance(Q1_lsrk, Q1_form)
    rhs_diag_ana = euclidean_distance(rhs_diag,rhs_anal)
    # rhs_lsrk_ana = euclidean_distance(rhs_lsrk,rhs_anal)
    # rhs_lsrk_diag = euclidean_distance(rhs_lsrk,rhs_diag)
    
    @info @sprintf """Finished
    c ⋅ Δt                             = %.16e
    norm(rhs_diag_ana)                 = %.16e
    """ c*dt rhs_diag_ana  

    return rhs_diag_ana
end




using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    numlevels =
        integration_testing || ClimateMachine.Settings.integration_testing ? 3 :
        1

    # polynomialorder = 4
    # base_num_elem = 4
    direction = HorizontalDirection
    dim = 3

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32)
            for base_num_elem in (4,5,6)
                for polynomialorder in (3,4,5,6)

                    # D = 5e-8 * SMatrix{3, 3, FT}(1,0,0,0,1,0,0,0,1,) 
                    # # estimated as D = (dx/2)^4/2/τ where τ is the hyperdiff time scale 3600sec 
                    
                    τ = 3600 # time scale for hyperdiffusion

                    xrange = range(FT(0); length = base_num_elem + 1, stop = FT(2pi))
                    brickrange = ntuple(j -> xrange, dim)
                    periodicity = ntuple(j -> true, dim)
                    topl = StackedBrickTopology(
                        mpicomm,
                        brickrange;
                        periodicity = periodicity,
                    )
                    # timeend = 1
                    # outputtime = 1

                    @info (ArrayType, FT, base_num_elem, polynomialorder)
                    result = run(mpicomm, ArrayType, dim, topl, 
                                polynomialorder, FT, direction, τ)
                        
                    @test result < 0.01
            
                end
            end
        end
    end
end
nothing
