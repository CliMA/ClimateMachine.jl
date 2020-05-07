using Test
using MPI
using ClimateMachine.Mesh.BrickMesh

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD

    (elemtovert, elemtocorner, facecode) = brickmesh(
        (2.0:5.0, 4.0:6.0),
        (false, true);
        part = MPI.Comm_rank(comm) + 1,
        numparts = MPI.Comm_size(comm),
    )

    code = centroidtocode(comm, elemtocorner)
    (d, nelem) = size(code)

    root = 0
    counts = MPI.Allgather(Cint(length(code)), comm)
    code_all = MPI.Gatherv(code, counts, root, comm)

    if MPI.Comm_rank(comm) == root
        code_all = reshape(code_all, d, div(sum(counts), d))

        code_expect = UInt64[
            0x0000000000000000 0x1555555555555555 0xffffffffffffffff 0x5555555555555555 0x6aaaaaaaaaaaaaaa 0xaaaaaaaaaaaaaaaa
            0x0000000000000000 0x5555555555555555 0xffffffffffffffff 0x5555555555555555 0xaaaaaaaaaaaaaaaa 0xaaaaaaaaaaaaaaaa
        ]
        @test code_all == code_expect
    end
end

main()
