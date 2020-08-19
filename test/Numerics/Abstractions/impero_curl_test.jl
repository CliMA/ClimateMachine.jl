using MPI
using Test
using LinearAlgebra
using ClimateMachine
using ClimateMachine.MPIStateArrays
using ClimateMachine.Diagnostics
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Abstractions

import ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD

let
    dim = 3
    FT = Float64
    Nh = 3
    Nv = 3

    # Defining grid structure
    periodicity = ntuple(j -> false, dim)
    brickrange = (
        ntuple(
            j -> range(FT(-1); length = Nh + 1,
                stop = 1),
            dim - 1,
        )...,
        range(FT(-5); length = Nv + 1, stop = 5),
    )

    topl = StackedBrickTopology(
                                mpicomm,
                                brickrange;
                                periodicity = periodicity,
                                boundary = (
                                    ntuple(j -> (1, 2), dim - 1)...,
                                    (3, 4),
                                )
    )

    N = 1
    Np = N+1
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )
    numelem = length(grid.topology.realelems)

    u = MPIStateArray{Float64}(mpicomm, ArrayType, Np^dim, 1, numelem)
    v = MPIStateArray{Float64}(mpicomm, ArrayType, Np^dim, 1, numelem)
    w = MPIStateArray{Float64}(mpicomm, ArrayType, Np^dim, 1, numelem)
    T = MPIStateArray{Float64}(mpicomm, ArrayType, Np^dim, 1, numelem)

    # the i stands for "impero"
    # TODO: move to Abstractions directory

    @wrapper iu=u iv=v iw=w ivelocity=(u,v,w)

    div, ∇, curl, ×, ⋅ = impero_operators(grid)

    ∇u = Diagnostics.VectorGradient(grid, u, 1)
    ∇v = Diagnostics.VectorGradient(grid, v, 1)
    ∇w = Diagnostics.VectorGradient(grid, w, 1)
    ∇T = Diagnostics.VectorGradient(grid, T, 1)
    Q = [u,v,w]

    d1 = Diagnostics.VectorGradient(grid, Q[1], 1)
    d2 = Diagnostics.VectorGradient(grid, Q[2], 1)
    d3 = Diagnostics.VectorGradient(grid, Q[3], 1)

    vgrad = Diagnostics.VectorGradients(d1, d2, d3)
    vort = Diagnostics.Vorticity(grid, vgrad)

    @testset "curl check" begin
        @test isapprox(norm(vort.data),
                       norm(Abstractions.curl(grid, Q)))
    end

    norm(compute(∇(iu)) - ∇u)
    curl(ivelocity)
    norm(compute(curl(ivelocity)).data - vort.data)

    # div grad curl
    ∇⋅ivelocity
    ∇(ivelocity)
    ∇×ivelocity

    # check that norms are zero in this case
    norm(compute(∇⋅ivelocity))
    norm(compute(∇×ivelocity).data - compute(curl(ivelocity)).data)
    computed_grad = compute(∇(ivelocity))

end
