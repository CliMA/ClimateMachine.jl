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
    D::FT
    k::SVector{3, FT}
end

function nodal_init_state_auxiliary!(
    balance_law::HyperDiffusion,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    aux.D = balance_law.problem.D * Matrix{FT}(I,3,3)
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
        # k = SVector(1, 2, 3)
        k = problem.k
        kD = k * k' .* problem.D
        c = get_c(k, kD, dir, dim)
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
    τ,
    k
)

    grid = DiscontinuousSpectralElementGrid(
            topl,
            FloatType = FT,
            DeviceArray = ArrayType,
            polynomialorder = N,
        )
    dx = min_node_distance(grid)

    D = (dx/2)^4/2/τ 

    # model = HyperDiffusion{dim}(ConstantHyperDiffusion{dim, direction(), FT}(D))
    model = HyperDiffusion{dim}(ConstantHyperDiffusion{dim, direction(), FT}(D, k))
    dg = DGModel(
            model,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            direction = direction(),
        )

    Q0 = init_ode_state(dg, FT(0))
    @info "Δ(horz)" dx

    rhs_diag = similar(Q0)
    dg(rhs_diag, Q0, nothing, 0)

    # k = SVector(1, 2, 3)
    kD = k * k' .* D
    c = get_c(k, kD, direction(), dim)
    rhs_anal = -c*Q0  

    rhs_diag_ana = euclidean_distance(rhs_diag,rhs_anal)
    
    @info @sprintf """Finished
    norm(rhs_diag_ana)                 = %.16e
    """ rhs_diag_ana  

    return rhs_diag_ana
end

get_c(k, kD, dir::VerticalDirection, dim) =
    k[dim]^2 * kD[dim, dim]
get_c(k, kD, dir::EveryDirection, dim) =
    sum(abs2, k[SOneTo(dim)]) * sum(kD[SOneTo(dim), SOneTo(dim)])
get_c(k, kD, dir::HorizontalDirection, dim) =
    sum(abs2, k[SOneTo(dim - 1)]) * sum(kD[SOneTo(dim - 1), SOneTo(dim - 1)])

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    numlevels =
        integration_testing || ClimateMachine.Settings.integration_testing ? 3 :
        1

    direction = HorizontalDirection
    dim = 3

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32,)
            for base_num_elem in (4,5,)
                for polynomialorder in (3,4,5,6,)

                    for τ in (1,4,8,) # time scale for hyperdiffusion
                        xrange = range(FT(0); length = base_num_elem + 1, stop = FT(2pi))
                        brickrange = ntuple(j -> xrange, dim)
                        periodicity = ntuple(j -> true, dim)
                        topl = StackedBrickTopology(
                            mpicomm,
                            brickrange;
                            periodicity = periodicity,
                        )

                        @info (ArrayType, FT, base_num_elem, polynomialorder)
                        result = run(mpicomm, ArrayType, dim, topl, 
                                    polynomialorder, FT, direction, τ*3600, SVector(1, 2, 3) )
                            
                        @test result < 1e-4
                    
                    end
                end
            end
        end
    end
end
nothing
