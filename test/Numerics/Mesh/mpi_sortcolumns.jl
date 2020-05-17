using Test
using MPI
using ClimateMachine.Mesh.BrickMesh

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    d = 4
    A = rand(1:10, d, 3rank)
    B = BrickMesh.parallelsortcolumns(comm, A, rev = true)

    root = 0
    Acounts = MPI.Allgather(Cint(length(A)), comm)
    A_all = MPI.Gatherv(A, Acounts, root, comm)

    Bcounts = MPI.Allgather(Cint(length(B)), comm)
    B_all = MPI.Gatherv(B, Bcounts, root, comm)

    if MPI.Comm_rank(comm) == root
        A_all = reshape(A_all, d, div(length(A_all), d))
        B_all = reshape(B_all, d, div(length(B_all), d))

        A_all = sortslices(A_all, dims = 2, rev = true)

        @test A_all == B_all
    end
end

main()
