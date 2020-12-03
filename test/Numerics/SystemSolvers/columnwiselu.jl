using MPI
using Test
using LinearAlgebra
using Random
using KernelAbstractions, StaticArrays

using ClimateMachine
using ClimateMachine.SystemSolvers
using ClimateMachine.SystemSolvers:
    band_lu_kernel!,
    band_forward_kernel!,
    band_back_kernel!,
    DGColumnBandedMatrix

import ClimateMachine.MPIStateArrays: array_device

ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()

function band_to_full(B, p, q)
    _, n = size(B)

    A = similar(B, n, n) # assume square
    fill!(A, 0)
    for j in 1:n, i in max(1, j - q):min(j + p, n)
        A[i, j] = B[q + i - j + 1, j]
    end
    A
end

function run_columnwiselu_test(FT, N)

    Nq = N .+ 1
    Nq1 = Nq[1]
    Nq2 = Nq[2]
    Nqv = Nq[3]
    nstate = 3
    nvertelem = 5
    nhorzelem = 4
    eband = 2

    m = n = Nqv * nstate * nvertelem
    p = q = Nqv * nstate * eband - 1

    Random.seed!(1234)
    AB = rand(FT, Nq1, Nq2, p + q + 1, n, nhorzelem)

    AB[:, :, q + 1, :, :] .+= 10 # Make A's diagonally dominate

    Random.seed!(5678)
    b = rand(FT, Nq1, Nq2, Nqv, nstate, nvertelem, nhorzelem)
    x = similar(b)

    perm = (4, 3, 5, 1, 2, 6)
    bp = reshape(PermutedDimsArray(b, perm), n, Nq1, Nq2, nhorzelem)
    xp = reshape(PermutedDimsArray(x, perm), n, Nq1, Nq2, nhorzelem)

    d_F = ArrayType(AB)
    d_F = DGColumnBandedMatrix{
        3,
        N,
        nstate,
        nhorzelem,
        nvertelem,
        eband,
        false,
        typeof(d_F),
    }(
        d_F,
    )

    groupsize = (Nq1, Nq2)
    ndrange = (Nq1, Nq2, nhorzelem)

    event = Event(array_device(d_F.data))
    event = band_lu_kernel!(array_device(d_F.data), groupsize, ndrange)(
        d_F,
        dependencies = (event,),
    )
    wait(array_device(d_F.data), event)

    F = Array(d_F.data)

    for h in 1:nhorzelem, j in 1:Nq2, i in 1:Nq1
        B = AB[i, j, :, :, h]
        G = band_to_full(B, p, q)
        GLU = lu!(G, Val(false))

        H = band_to_full(F[i, j, :, :, h], p, q)

        @assert H ≈ G

        xp[:, i, j, h] .= GLU \ bp[:, i, j, h]
    end

    b = reshape(b, Nq1 * Nq2 * Nqv, nstate, nvertelem * nhorzelem)
    x = reshape(x, Nq1 * Nq2 * Nqv, nstate, nvertelem * nhorzelem)

    d_x = ArrayType(b)

    event = Event(array_device(d_x))
    event = band_forward_kernel!(array_device(d_x), groupsize, ndrange)(
        d_x,
        d_F,
        dependencies = (event,),
    )

    event = band_back_kernel!(array_device(d_x), groupsize, ndrange)(
        d_x,
        d_F,
        dependencies = (event,),
    )
    wait(array_device(d_x), event)

    result = x ≈ Array(d_x)
    return result
end

@testset "Columnwise LU test" begin
    for FT in (Float64, Float32)
        for N in ((1, 1, 1), (1, 1, 2), (2, 2, 1))
            result = run_columnwiselu_test(FT, N)
            @test result
        end
    end
end
