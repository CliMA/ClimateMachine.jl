using Test
using MPI
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.MPIStateArrays
using ClimateMachine.Mesh.DSS
using KernelAbstractions

DA = ClimateMachine.array_type()

function test_dss()
    FT = Float64
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    @assert csize == 1

    N = (4, 4, 5)
    brickrange = (0:2, 5:6, 0:1)
    periodicity = (false, false, false)
    nvars = 1

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

    ldof = DA{FT, 1}(reshape(1:Np, Np))

    for ivar in 1:nvars
        for ielem in realelems
            Q.data[:, ivar, ielem] .= ldof .+ FT((ielem - 1) * Np)
        end
        Q.data[:, ivar, ghostelems] .= FT(0)
    end

    pre_dss = Array(Q.data)

    dss!(Q, grid)
    #---------Tests-------------------------
    nodes = reshape(1:Np, Nq)
    interior = nodes[2:(Nq[1] - 1), 2:(Nq[2] - 1), 2:(Nq[3] - 1)][:]
    vertmap = Array(grid.vertmap)
    edgemap = Array(grid.edgemap)
    facemap = Array(grid.facemap)
    post_dss = Array(Q.data)

    ne1r = 1:max(Nq[1] - 2, 0)
    ne2r = 1:max(Nq[2] - 2, 0)
    ne3r = 1:max(Nq[3] - 2, 0)

    nf1r = 1:max((Nq[2] - 2) * (Nq[3] - 2), 0)
    nf2r = 1:max((Nq[1] - 2) * (Nq[3] - 2), 0)
    nf3r = 1:max((Nq[1] - 2) * (Nq[2] - 2), 0)

    compare(el1, i1, el2, i2, efmap, rng, post, pre) =
        post[efmap[rng, i1, 1], 1, el1] == pre[efmap[rng, i2, 1], 1, el2]

    compare(el1, i1, el2, i2, el3, i3, efmap, rng, post, pre) =
        post[efmap[rng, i1, 1], 1, el1] ==
        pre[efmap[rng, i2, 1], 1, el2] .+ pre[efmap[rng, i3, 1], 1, el3]

    el1, el2 = 1, 2
    # Element # 1 --------------------------------------------------------------------------
    # interior dof should not be affected
    @test pre_dss[interior, 1, 1] == post_dss[interior, 1, 1]
    # vertex check
    @test post_dss[vertmap, 1, 1] == [
        pre_dss[vertmap[1], 1, 1],
        pre_dss[vertmap[2], 1, 1] + pre_dss[vertmap[1], 1, 2],
        pre_dss[vertmap[3], 1, 1],
        pre_dss[vertmap[4], 1, 1] + pre_dss[vertmap[3], 1, 2],
        pre_dss[vertmap[5], 1, 1],
        pre_dss[vertmap[6], 1, 1] + pre_dss[vertmap[5], 1, 2],
        pre_dss[vertmap[7], 1, 1],
        pre_dss[vertmap[8], 1, 1] + pre_dss[vertmap[7], 1, 2],
    ]
    # edge check
    @test compare(el1, 1, el1, 1, edgemap, ne1r, post_dss, pre_dss)          # edge  1 (unaffected)
    @test compare(el1, 2, el1, 2, edgemap, ne1r, post_dss, pre_dss)          # edge  2 (unaffected)
    @test compare(el1, 3, el1, 3, edgemap, ne1r, post_dss, pre_dss)          # edge  3 (unaffected)
    @test compare(el1, 4, el1, 4, edgemap, ne1r, post_dss, pre_dss)          # edge  4 (unaffected)

    @test compare(el1, 5, el1, 5, edgemap, ne2r, post_dss, pre_dss)          # edge  5 (unaffected)
    @test compare(el1, 6, el1, 6, el2, 5, edgemap, ne2r, post_dss, pre_dss) # edge  6 (shared edge)
    @test compare(el1, 7, el1, 7, edgemap, ne2r, post_dss, pre_dss)          # edge  7 (unaffected)
    @test compare(el1, 8, el1, 8, el2, 7, edgemap, ne2r, post_dss, pre_dss) # edge  8 (shared edge)

    @test compare(el1, 9, el1, 9, edgemap, ne3r, post_dss, pre_dss)          # edge  9 (unaffected)
    @test compare(el1, 10, el1, 10, el2, 9, edgemap, ne3r, post_dss, pre_dss) # edge 10 (shared edge)
    @test compare(el1, 11, el1, 11, edgemap, ne3r, post_dss, pre_dss)          # edge 11 (unaffected)
    @test compare(el1, 12, el1, 12, el2, 11, edgemap, ne3r, post_dss, pre_dss) # edge 12 (shared edge)

    # face check
    @test compare(el1, 1, el1, 1, facemap, nf1r, post_dss, pre_dss)         # face 1 (unaffected)
    @test compare(el1, 2, el1, 2, el2, 1, facemap, nf1r, post_dss, pre_dss) # face 2 (shared face)

    @test compare(el1, 3, el1, 3, facemap, nf2r, post_dss, pre_dss)         # face 3 (unaffected)
    @test compare(el1, 4, el1, 4, facemap, nf2r, post_dss, pre_dss)         # face 4 (unaffected)

    @test compare(el1, 5, el1, 5, facemap, nf3r, post_dss, pre_dss)         # face 5 (unaffected)
    @test compare(el1, 6, el1, 6, facemap, nf3r, post_dss, pre_dss)         # face 6 (unaffected)

    # Element # 2 --------------------------------------------------------------------------
    # interior dof should not be affected
    @test pre_dss[interior, 1, 2] == post_dss[interior, 1, 2]
    # vertex check
    @test post_dss[vertmap, 1, 2] == [
        pre_dss[vertmap[1], 1, 2] + pre_dss[vertmap[2], 1, 1],
        pre_dss[vertmap[2], 1, 2],
        pre_dss[vertmap[3], 1, 2] + pre_dss[vertmap[4], 1, 1],
        pre_dss[vertmap[4], 1, 2],
        pre_dss[vertmap[5], 1, 2] + pre_dss[vertmap[6], 1, 1],
        pre_dss[vertmap[6], 1, 2],
        pre_dss[vertmap[7], 1, 2] + pre_dss[vertmap[8], 1, 1],
        pre_dss[vertmap[8], 1, 2],
    ]
    # edge check
    @test compare(el2, 1, el2, 1, edgemap, ne1r, post_dss, pre_dss)          # edge  1 (unaffected)
    @test compare(el2, 2, el2, 2, edgemap, ne1r, post_dss, pre_dss)          # edge  2 (unaffected)
    @test compare(el2, 3, el2, 3, edgemap, ne1r, post_dss, pre_dss)          # edge  3 (unaffected)
    @test compare(el2, 4, el2, 4, edgemap, ne1r, post_dss, pre_dss)          # edge  4 (unaffected)

    @test compare(el2, 5, el2, 5, el1, 6, edgemap, ne2r, post_dss, pre_dss) # edge  5 (shared edge)
    @test compare(el2, 6, el2, 6, edgemap, ne2r, post_dss, pre_dss)          # edge  6 (unaffected)
    @test compare(el2, 7, el2, 7, el1, 8, edgemap, ne2r, post_dss, pre_dss) # edge  7 (shared edge)
    @test compare(el2, 8, el2, 8, edgemap, ne2r, post_dss, pre_dss)          # edge  8 (unaffected)

    @test compare(el2, 9, el2, 9, el1, 10, edgemap, ne3r, post_dss, pre_dss) # edge  9 (shared edge)
    @test compare(el2, 10, el2, 10, edgemap, ne3r, post_dss, pre_dss)          # edge 10 (unaffected)
    @test compare(el2, 11, el2, 11, el1, 12, edgemap, ne3r, post_dss, pre_dss) # edge 11 (shared edge)
    @test compare(el2, 12, el2, 12, edgemap, ne3r, post_dss, pre_dss)          # edge 12 (unaffected)

    # face check
    @test compare(el2, 1, el2, 1, el1, 2, facemap, nf1r, post_dss, pre_dss) # face 1 (shared face)
    @test compare(el2, 2, el2, 2, facemap, nf1r, post_dss, pre_dss)         # face 2 (unaffected)

    @test compare(el2, 3, el2, 3, facemap, nf2r, post_dss, pre_dss)         # face 3 (unaffected)
    @test compare(el2, 4, el2, 4, facemap, nf2r, post_dss, pre_dss)         # face 4 (unaffected)

    @test compare(el2, 5, el2, 5, facemap, nf3r, post_dss, pre_dss)         # face 5 (unaffected)
    @test compare(el2, 6, el2, 6, facemap, nf3r, post_dss, pre_dss)         # face 6 (unaffected)

end

test_dss()
