using Test
using MPI
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.MPIStateArrays
using KernelAbstractions

DA = ClimateMachine.array_type()

function test_dss3d_stacked_3d()
    FT = Float64
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    @assert csize == 3

    N = (4, 4, 5)
    brickrange = (0:4, 5:9, 0:3)
    periodicity = (false, false, false)
    nvars = 3

    topl = StackedBrickTopology(
        comm,
        brickrange,
        periodicity = periodicity,
        boundary = ((1, 2), (3, 4), (5, 6)),
        connectivity = :full,
    )
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = DA,
        polynomialorder = N,
    )

    Nq = N .+ 1
    Np = prod(Nq)
    Q = MPIStateArray{FT}(
        comm,
        DA,
        Np,
        nvars,
        length(topl.elems),
        realelems = topl.realelems,
        ghostelems = topl.ghostelems,
        vmaprecv = grid.vmaprecv,
        vmapsend = grid.vmapsend,
        nabrtorank = topl.nabrtorank,
        nabrtovmaprecv = grid.nabrtovmaprecv,
        nabrtovmapsend = grid.nabrtovmapsend,
    )
    realelems = Q.realelems
    ghostelems = Q.ghostelems

    Q.data[:, :, realelems] .= FT(1)
    Q.data[:, :, ghostelems] .= FT(0)

    effect_dss3d!(Q, grid)

    #---------Tests-------------------------
    nodes = reshape(1:Np, Nq)
    interior = nodes[2:(Nq[1] - 1), 2:(Nq[2] - 1), 2:(Nq[3] - 1)][:]
    vertmap = Array(grid.vertmap)
    edgemap = Array(grid.edgemap)
    facemap = Array(grid.facemap)
    data = Array(Q.data)
    compare(data, ivar, iel, efmap) =
        unique(data[setdiff(efmap, [-1]), ivar, lel])
    lel = 1 # local element number for each process
    if crank == 0
        for ivar in 1:nvars
            # local element #1, global elememt # 1
            # interior dof should not be affected
            idof = unique(data[interior, ivar, lel])
            @test idof == [FT(1)]
            # vertex dof check
            @test data[vertmap, ivar, lel] == FT.([1, 2, 2, 4, 2, 4, 4, 8])
            # edge dof check
            @test compare(data, ivar, lel, edgemap[:, 1, 1]) == [FT(1)] # edge #  1
            @test compare(data, ivar, lel, edgemap[:, 2, 1]) == [FT(2)] # edge #  2
            @test compare(data, ivar, lel, edgemap[:, 3, 1]) == [FT(2)] # edge #  3
            @test compare(data, ivar, lel, edgemap[:, 4, 1]) == [FT(4)] # edge #  4
            @test compare(data, ivar, lel, edgemap[:, 5, 1]) == [FT(1)] # edge #  5
            @test compare(data, ivar, lel, edgemap[:, 6, 1]) == [FT(2)] # edge #  6
            @test compare(data, ivar, lel, edgemap[:, 7, 1]) == [FT(2)] # edge #  7
            @test compare(data, ivar, lel, edgemap[:, 8, 1]) == [FT(4)] # edge #  8
            @test compare(data, ivar, lel, edgemap[:, 9, 1]) == [FT(1)] # edge #  9
            @test compare(data, ivar, lel, edgemap[:, 10, 1]) == [FT(2)] # edge # 10
            @test compare(data, ivar, lel, edgemap[:, 11, 1]) == [FT(2)] # edge # 11
            @test compare(data, ivar, lel, edgemap[:, 12, 1]) == [FT(4)] # edge # 12

            # face dof check
            @test compare(data, ivar, lel, facemap[:, 1, 1]) == [FT(1)] # face # 1
            @test compare(data, ivar, lel, facemap[:, 2, 1]) == [FT(2)] # face # 2
            @test compare(data, ivar, lel, facemap[:, 3, 1]) == [FT(1)] # face # 3
            @test compare(data, ivar, lel, facemap[:, 4, 1]) == [FT(2)] # face # 4
            @test compare(data, ivar, lel, facemap[:, 5, 1]) == [FT(1)] # face # 5
            @test compare(data, ivar, lel, facemap[:, 6, 1]) == [FT(2)] # face # 6
        end

    elseif crank == 1
        for ivar in 1:nvars
            # local element #1, global elememt # 6 (in 2D)
            # interior dof should not be affected
            idof = unique(data[interior, ivar, lel])
            @test idof == [FT(1)]
            # vertex dof check
            @test data[vertmap, ivar, lel] == FT.([2, 4, 1, 2, 4, 8, 2, 4])
            # edge dof check
            @test compare(data, ivar, lel, edgemap[:, 1, 1]) == [FT(2)] # edge #  1
            @test compare(data, ivar, lel, edgemap[:, 2, 1]) == [FT(1)] # edge #  2
            @test compare(data, ivar, lel, edgemap[:, 3, 1]) == [FT(4)] # edge #  3
            @test compare(data, ivar, lel, edgemap[:, 4, 1]) == [FT(2)] # edge #  4
            @test compare(data, ivar, lel, edgemap[:, 5, 1]) == [FT(1)] # edge #  5
            @test compare(data, ivar, lel, edgemap[:, 6, 1]) == [FT(2)] # edge #  6
            @test compare(data, ivar, lel, edgemap[:, 7, 1]) == [FT(2)] # edge #  7
            @test compare(data, ivar, lel, edgemap[:, 8, 1]) == [FT(4)] # edge #  8
            @test compare(data, ivar, lel, edgemap[:, 9, 1]) == [FT(2)] # edge #  9
            @test compare(data, ivar, lel, edgemap[:, 10, 1]) == [FT(4)] # edge # 10
            @test compare(data, ivar, lel, edgemap[:, 11, 1]) == [FT(1)] # edge # 11
            @test compare(data, ivar, lel, edgemap[:, 12, 1]) == [FT(2)] # edge # 12


            # face dof check
            @test compare(data, ivar, lel, facemap[:, 1, 1]) == [FT(1)] # face # 1
            @test compare(data, ivar, lel, facemap[:, 2, 1]) == [FT(2)] # face # 2
            @test compare(data, ivar, lel, facemap[:, 3, 1]) == [FT(2)] # face # 3
            @test compare(data, ivar, lel, facemap[:, 4, 1]) == [FT(1)] # face # 4
            @test compare(data, ivar, lel, facemap[:, 5, 1]) == [FT(1)] # face # 5
            @test compare(data, ivar, lel, facemap[:, 6, 1]) == [FT(2)] # face # 6
        end
    else # crank == 2
        for ivar in 1:nvars
            # local element #1, global elememt # 11 (in 2D)
            # interior dof should not be affected
            idof = unique(data[interior, ivar, lel])
            @test idof == [FT(1)]
            # vertex dof check
            @test data[vertmap, ivar, lel] == FT.([4, 2, 2, 1, 8, 4, 4, 2])
            # edge dof check
            @test compare(data, ivar, lel, edgemap[:, 1, 1]) == [FT(2)] # edge #  1
            @test compare(data, ivar, lel, edgemap[:, 2, 1]) == [FT(1)] # edge #  2
            @test compare(data, ivar, lel, edgemap[:, 3, 1]) == [FT(4)] # edge #  3
            @test compare(data, ivar, lel, edgemap[:, 4, 1]) == [FT(2)] # edge #  4
            @test compare(data, ivar, lel, edgemap[:, 5, 1]) == [FT(2)] # edge #  5
            @test compare(data, ivar, lel, edgemap[:, 6, 1]) == [FT(1)] # edge #  6
            @test compare(data, ivar, lel, edgemap[:, 7, 1]) == [FT(4)] # edge #  7
            @test compare(data, ivar, lel, edgemap[:, 8, 1]) == [FT(2)] # edge #  8
            @test compare(data, ivar, lel, edgemap[:, 9, 1]) == [FT(4)] # edge #  9
            @test compare(data, ivar, lel, edgemap[:, 10, 1]) == [FT(2)] # edge # 10
            @test compare(data, ivar, lel, edgemap[:, 11, 1]) == [FT(2)] # edge # 11
            @test compare(data, ivar, lel, edgemap[:, 12, 1]) == [FT(1)] # edge # 12
            # face dof check
            @test compare(data, ivar, lel, facemap[:, 1, 1]) == [FT(2)] # face # 1
            @test compare(data, ivar, lel, facemap[:, 2, 1]) == [FT(1)] # face # 2
            @test compare(data, ivar, lel, facemap[:, 3, 1]) == [FT(2)] # face # 3
            @test compare(data, ivar, lel, facemap[:, 4, 1]) == [FT(1)] # face # 4
            @test compare(data, ivar, lel, facemap[:, 5, 1]) == [FT(1)] # face # 5
            @test compare(data, ivar, lel, facemap[:, 6, 1]) == [FT(2)] # face # 6
        end
    end
end

test_dss3d_stacked_3d()
