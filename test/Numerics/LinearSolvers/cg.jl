using MPI
using Test
using LinearAlgebra
using Random
using StaticArrays
using KernelAbstractions: CPU, CUDA
using ClimateMachine
using ClimateMachine.LinearSolvers
using ClimateMachine.ConjugateGradientSolver
using ClimateMachine.MPIStateArrays
using CUDAapi
using Random
Random.seed!(1235)

let
    ClimateMachine.init()
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()
    device = ArrayType == Array ? CPU() : CUDA()
    n = 100
    T = Float64
    A = rand(n, n)
    scale = 2.0
    ϵ = 0.1
    # Matrix 1
    A = A' * A .* ϵ + scale * I

    err_thresh = sqrt(eps(T))

    # Matrix 2
    # A = Diagonal(collect(1:n) * 1.0)
    positive_definite = minimum(eigvals(A)) > eps(1.0)
    @test positive_definite

    b = ones(n) * 1.0
    mulbyA!(y, x) = (y .= A * x)

    tol = sqrt(eps(T))
    method(b, tol) = ConjugateGradient(b, max_iter = n)
    linearsolver = method(b, tol)

    x = ones(n) * 1.0
    x0 = copy(x)
    iters = linearsolve!(mulbyA!, linearsolver, x, b; max_iters = Inf)
    exact = A \ b
    x0 = copy(x)

    @testset "Array test" begin
        @test norm(x - exact) / norm(exact) < err_thresh
        @test norm(A * x - b) / norm(b) < err_thresh
    end

    # Testing for CuArrays
    if CUDAapi.has_cuda_gpu()
        at_A = ArrayType(A)
        at_b = ArrayType(b)
        at_x = ArrayType(ones(n) * 1.0)
        mulbyat_A!(y, x) = (y .= at_A * x)
        at_method(b, tol) = ConjugateGradient(b, max_iter = n)
        linearsolver = at_method(at_b, tol)
        iters =
            linearsolve!(mulbyat_A!, linearsolver, at_x, at_b; max_iters = n)
        exact = at_A \ at_b

        @testset "CuArray test" begin
            @test norm(at_x - exact) / norm(exact) < err_thresh
            @test norm(at_A * at_x - at_b) / norm(at_b) < err_thresh
        end
    end

    mpi_b = MPIStateArray{T}(mpicomm, ArrayType, 4, 4, 4)
    mpi_x = MPIStateArray{T}(mpicomm, ArrayType, 4, 4, 4)
    mpi_A = ArrayType(randn(4^3, 4^3))
    mpi_A .= mpi_A' * mpi_A

    function mpi_mulby!(x, y)
        fy = y.data[:]
        fx = mpi_A * fy
        x.data[:] .= fx[:]
        return nothing
    end

    mpi_b.data[:] .= ArrayType(randn(4^3))
    mpi_x.data[:] .= ArrayType(randn(4^3))

    mpi_method(mpi_b, tol) = ConjugateGradient(mpi_b, max_iter = n)
    linearsolver = mpi_method(mpi_b, tol)
    iters = linearsolve!(mpi_mulby!, linearsolver, mpi_x, mpi_b; max_iters = n)

    exact = mpi_A \ mpi_b[:]

    @testset "MPIStateArray test" begin
        @test norm(mpi_x.data[:] - exact) / norm(exact) < err_thresh
        mpi_Ax = MPIStateArray{T}(mpicomm, ArrayType, 4, 4, 4)
        mpi_mulby!(mpi_Ax, mpi_x)
        @test norm(mpi_Ax - mpi_b) / norm(mpi_b) < err_thresh
    end

    # ## More Complex Example
    function closure_linear_operator!(A, tup)
        function linear_operator!(y, x)
            alias_x = reshape(x, tup)
            alias_y = reshape(y, tup)
            for i6 in 1:tup[6]
                for i4 in 1:tup[4]
                    for i2 in 1:tup[2]
                        for i1 in 1:tup[1]
                            tmp = alias_x[i1, i2, :, i4, :, i6][:]
                            tmp2 = A[i1, i2, i4, i6] * tmp
                            alias_y[i1, i2, :, i4, :, i6] .=
                                reshape(tmp2, (tup[3], tup[5]))
                        end
                    end
                end
            end
        end
    end

    tup = (3, 4, 7, 2, 20, 2)

    B = [
        randn(tup[3] * tup[5], tup[3] * tup[5])
        for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
    ]
    columnwise_A = [
        B[i1, i2, i4, i6] * B[i1, i2, i4, i6]' + 10I
        for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
    ]
    columnwise_inv_A = [
        inv(columnwise_A[i1, i2, i4, i6])
        for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
    ]
    columnwise_linear_operator! = closure_linear_operator!(columnwise_A, tup)
    columnwise_inverse_linear_operator! =
        closure_linear_operator!(columnwise_inv_A, tup)

    mpi_tup = (tup[1] * tup[2] * tup[3], tup[4], tup[5] * tup[6])
    b = randn(mpi_tup)
    x = randn(mpi_tup)

    linearsolver = ConjugateGradient(
        x,
        max_iter = tup[3] * tup[5],
        dims = (3, 5),
        reshape_tuple = tup,
    )

    iters = linearsolve!(columnwise_linear_operator!, linearsolver, x, b)
    x_exact = copy(x)
    columnwise_inverse_linear_operator!(x_exact, b)

    @testset "Columnwise test" begin
        @test norm(x - x_exact) / norm(x_exact) < err_thresh
    end
end

nothing
