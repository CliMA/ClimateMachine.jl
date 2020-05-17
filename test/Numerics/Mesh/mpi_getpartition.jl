using Test
using MPI
using ClimateMachine.Mesh.BrickMesh
using Random

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    Nelemtotal = 113
    Random.seed!(1234)
    globalcode = randperm(Nelemtotal)

    @assert csize > 1

    bs = [
        (i == 1) ? (1:0) :
            BrickMesh.linearpartition(Nelemtotal, i - 1, csize - 1)
        for i in 1:csize
    ]
    as = [BrickMesh.linearpartition(Nelemtotal, i, csize) for i in 1:csize]

    codeb = globalcode[bs[crank + 1]]

    (so, ss, rs) = BrickMesh.getpartition(comm, codeb)

    codeb = codeb[so]
    codec = []
    for r in 0:(csize - 1)
        sendrange = ss[r + 1]:(ss[r + 2] - 1)
        rcounts = MPI.Allgather(Cint(length(sendrange)), comm)
        c = MPI.Gatherv(view(codeb, sendrange), rcounts, r, comm)
        if r == crank
            codec = c
        end
    end

    codea = (1:Nelemtotal)[as[crank + 1]]

    @test sort(codec) == codea
end

main()
