using MPI
using Test
using LinearAlgebra
using Random
using StaticArrays
using ClimateMachine
using ClimateMachine.LinearSolvers
using ClimateMachine.BatchedGeneralizedMinimalResidualSolver
using ClimateMachine.MPIStateArrays
using CUDAapi
using Random
using KernelAbstractions
using CuArrays

let
    Random.seed!(1235)
    if CUDAapi.has_cuda_gpu()
        arrays = [Array, CuArray]
    else
        arrays = [Array]
    end
    # Initialize
    ClimateMachine.init()
    T = Float64
    mpicomm = MPI.COMM_WORLD
    # set the error threshold
    err_thresh = 1e-8

    @kernel function multiply_A_kernel!(x, A, y, n1, n2)
        I = @index(Global)
        for i in 1:n1
            tmp = zero(eltype(x))
            for j in 1:n2
                tmp += A[i, j, I] * y[j, I]
            end
            x[i, I] = tmp
        end
    end

    function multiply_by_A!(
        x,
        A,
        y,
        n1,
        n2;
        ndrange = size(x[1, :]),
        cpu_threads = Threads.nthreads(),
        gpu_threads = 256,
    )
        if isa(x, Array)
            kernel! = multiply_A_kernel!(CPU(), cpu_threads)
        else
            kernel! = multiply_A_kernel!(CUDA(), gpu_threads)
        end
        return kernel!(x, A, y, n1, n2, ndrange = ndrange)
    end
    # for defining linear_operator
    function closure_linear_operator_multi!(A, n1, n2, n3)
        function linear_operator!(x, y)
            event = multiply_by_A!(x, A, y, n1, n2, ndrange = n3)
            wait(event)
            return nothing
        end
    end

    # Test 1: Basic Functionality
    for array in arrays
        ArrayType = array
        n = 100  # size of vector space
        ni = 10 # number of independent linear solves
        b = ArrayType(randn(n, ni)) # rhs
        x = ArrayType(randn(n, ni)) # initial guess
        A = ArrayType(randn((n, n, ni)) ./ sqrt(n) .* 1.0)
        for i in 1:n
            A[i, i, :] .+= 1.0
        end
        ss = size(b)[1]
        gmres = BatchedGeneralizedMinimalResidual(
            b,
            ArrayType = ArrayType,
            subspace_size = ss,
        )
        for i in 1:ni
            x[:, i] = A[:, :, i] \ b[:, i]
        end
        sol = copy(x)
        x += ArrayType(randn(n, ni) * 0.01 * maximum(abs.(x)))
        y = copy(x)
        linear_operator! = closure_linear_operator_multi!(A, size(A)...)
        iters = linearsolve!(linear_operator!, gmres, x, b; max_iters = ss)
        linear_operator!(y, x)
        test_name = "(" * string(ArrayType) * ") Basic Test"
        @testset "$test_name" begin
            @test norm(y - b) / norm(b) < err_thresh
        end
    end
    ###
    # Test 2: MPI State Array
    for array in arrays
        ArrayType = array
        Random.seed!(1235)
        n1 = 8
        n2 = 3
        n3 = 10
        mpi_b = MPIStateArray{T}(mpicomm, ArrayType, n1, n2, n3)
        mpi_x = MPIStateArray{T}(mpicomm, ArrayType, n1, n2, n3)
        mpi_A = ArrayType(randn(n1 * n2, n1 * n2, n3))

        # need to make sure that mpi_b and mpi_x are reproducible
        mpi_b.data[:] .= ArrayType(randn(n1 * n2 * n3))
        mpi_x.data[:] .= ArrayType(randn(n1 * n2 * n3))
        mpi_y = copy(mpi_x)

        # for defining linear_operator
        function closure_linear_operator_mpi!(A, n1, n2, n3)
            function linear_operator!(x, y)
                alias_x = reshape(x.data, (n1, n3))
                alias_y = reshape(y.data, (n1, n3))
                event =
                    multiply_by_A!(alias_x, A, alias_y, n1, n2, ndrange = n3)
                wait(event)
                return nothing
            end
        end

        gmres = BatchedGeneralizedMinimalResidual(
            mpi_b,
            ArrayType = ArrayType,
            m = n1 * n2,
            n = n3,
        )

        # Now define the linear operator
        linear_operator! = closure_linear_operator_mpi!(mpi_A, size(mpi_A)...)
        iters = linearsolve!(
            linear_operator!,
            gmres,
            mpi_x,
            mpi_b;
            max_iters = n1 * n2,
        )
        linear_operator!(mpi_y, mpi_x)
        # check one in the batch
        sol = mpi_A[:, :, 1] \ mpi_b.data[:, :, 1][:]
        test_name = "(" * string(ArrayType) * ") MPIStaterray Test"
        @testset "$test_name" begin
            @test norm(mpi_y.data - mpi_b.data) / norm(mpi_b) < err_thresh
            @test norm(sol - mpi_x.data[:, :, 1][:]) < err_thresh
        end
    end

    ###
    # Test 3: Columnwise test (CPU ONLY)
    Random.seed!(1235)
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

    tup = (3, 4, 7, 6, 5, 2)

    B = [
        randn(tup[3] * tup[5], tup[3] * tup[5])
        for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
    ]
    columnwise_A = [
        B[i1, i2, i4, i6] + 10I
        for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
    ]
    # taking the inverse of A isn't great, but it is convenient
    columnwise_inv_A = [
        inv(columnwise_A[i1, i2, i4, i6])
        for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
    ]
    columnwise_linear_operator! = closure_linear_operator!(columnwise_A, tup)
    columnwise_inverse_linear_operator! =
        closure_linear_operator!(columnwise_inv_A, tup)

    mpi_tup = (tup[1] * tup[2] * tup[3], tup[4], tup[5] * tup[6])
    b = randn(mpi_tup)
    x = copy(b)
    columnwise_inverse_linear_operator!(x, b)
    x += randn((tup[1] * tup[2] * tup[3], tup[4], tup[5] * tup[6])) * 0.1

    reshape_tuple_f = tup
    permute_tuple_f = (5, 3, 1, 4, 2, 6)

    gmres = BatchedGeneralizedMinimalResidual(
        b,
        m = tup[3] * tup[5],
        n = tup[1] * tup[2] * tup[4] * tup[6],
        reshape_tuple_f = reshape_tuple_f,
        permute_tuple_f = permute_tuple_f,
        atol = eps(T),
        rtol = eps(T),
    )

    x_exact = copy(x)
    iters = linearsolve!(
        columnwise_linear_operator!,
        gmres,
        x,
        b,
        max_iters = tup[3] * tup[5],
    )

    @testset "(Array) Columnwise Test" begin
        columnwise_inverse_linear_operator!(x_exact, b)
        @test norm(x - x_exact) / norm(x_exact) < err_thresh
        columnwise_linear_operator!(x_exact, x)
        @test norm(x_exact - b) / norm(b) < err_thresh
    end
end
