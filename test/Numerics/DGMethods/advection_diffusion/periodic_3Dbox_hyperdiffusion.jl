"""
the hyperdiffusion model in a 3D periodic box
test the spatical discretization vs the analytical solution
init cond:  ρ = sin(kx+ly+mz)
analytical solution:  ∇^4 ρ = (k^2+l^2+m^2)^2 ρ
"""

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
    initial condition is given by ρ = sim(kx+ly+mz)
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
        state.ρ = sin(dot(k[SOneTo(dim)], x[SOneTo(dim)])) 
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
    D
)
"""
balance_law = DivergenceDampingLaw()
model = DGModel(
        balance_law,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient
    )
Q_0 = cool_state()
Q_1 = similar(Q_0)
model(Q_1, Q_0,  Q_0, nothing, 0)
"""
    grid = DiscontinuousSpectralElementGrid(
            topl,
            FloatType = FT,
            DeviceArray = ArrayType,
            polynomialorder = N,
        )
    model = HyperDiffusion{dim}(ConstantHyperDiffusion{dim, direction(), FT}(D))
    dg = DGModel(
            model,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
        )

    Q_0 = init_ode_state(dg, FT(0))
    Q_1 = similar(Q_0)
    dg(Q_1, Q_0, nothing, 0)

    norm1 = norm(Q_1)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    """ norm1

    return norm1
end

ClimateMachine.init()
ArrayType = ClimateMachine.array_type()
mpicomm = MPI.COMM_WORLD
FT = Float64
dim = 2
Ne = 4
xrange = range(FT(0); length = Ne + 1, stop = FT(2pi))
brickrange = ntuple(j -> xrange, dim)
periodicity = ntuple(j -> true, dim)
topl = StackedBrickTopology(
    mpicomm,
    brickrange;
    periodicity = periodicity,
)
polynomialorder = 4
D = 1 // 100 * SMatrix{3, 3, FT}(
        9 // 50,
        3 // 50,
        5 // 50,
        3 // 50,
        7 // 50,
        4 // 50,
        5 // 50,
        4 // 50,
        10 // 50,
    )

outnorm = run(mpicomm, ArrayType, dim, topl, polynomialorder, Float64, HorizontalDirection, D)
@info @sprintf """Finished
outnorm(Q)                 = %.16e
""" outnorm