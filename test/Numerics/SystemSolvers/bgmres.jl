using MPI
using Test
using LinearAlgebra
using Random
using StaticArrays
using ClimateMachine
using ClimateMachine.SystemSolvers
using ClimateMachine.MPIStateArrays
using CUDA
using Random
using KernelAbstractions
using CUDAKernels

import ClimateMachine.MPIStateArrays: array_device

ClimateMachine.init(; fix_rng_seed = true)

@kernel function multiply_by_A!(x, A, y, n1, n2)
    I = @index(Global)
    for i in 1:n1
        tmp = zero(eltype(x))
        for j in 1:n2
            tmp += A[i, j, I] * y[j, I]
        end
        x[i, I] = tmp
    end
end

let
    if CUDA.has_cuda_gpu()
        Arrays = [Array, CuArray]
    else
        Arrays = [Array]
    end

    for ArrayType in Arrays
        for T in [Float32, Float64]
            ϵ = eps(T)

            @testset "($ArrayType, $T) Basic Test" begin
                Random.seed!(42)

                # Test 1: Basic Functionality
                n = 100   # size of local (batch) matrix
                ni = 100  # batch size

                b = ArrayType(randn(n, ni))  # rhs
                x = ArrayType(randn(n, ni))  # initial guess
                x_ref = similar(x)

                A = ArrayType(randn((n, n, ni)) ./ sqrt(n))
                for i in 1:n
                    A[i, i, :] .+= 10i
                end

                ss = size(b)[1]
                bgmres = BatchedGeneralizedMinimalResidual(
                    b,
                    n,
                    ni,
                    M = ss,
                    atol = ϵ,
                    rtol = ϵ,
                )

                # Define the linear operator
                function closure_linear_operator_multi!(A, n1, n2, n3)
                    function linear_operator!(x, y)
                        device = array_device(x)
                        if isa(device, CPU)
                            groupsize = Threads.nthreads()
                        else # isa(device, CUDADevice)
                            groupsize = 256
                        end
                        event = Event(device)
                        event = multiply_by_A!(device, groupsize)(
                            x,
                            A,
                            y,
                            n1,
                            n2,
                            ndrange = n3,
                            dependencies = (event,),
                        )
                        wait(device, event)
                        nothing
                    end
                end
                linear_operator! = closure_linear_operator_multi!(A, size(A)...)

                # Now solve
                linearsolve!(
                    linear_operator!,
                    nothing,
                    bgmres,
                    x,
                    b;
                    max_iters = Inf,
                )

                # reference solution
                for i in 1:ni
                    x_ref[:, i] = A[:, :, i] \ b[:, i]
                end

                @test norm(x - x_ref) < 1000ϵ
            end

            ###
            # Test 2: MPI State Array
            ###
            @testset "($ArrayType, $T) MPIStateArray Test" begin
                Random.seed!(43)

                n1 = 8
                n2 = 3
                n3 = 10
                mpicomm = MPI.COMM_WORLD
                mpi_b = MPIStateArray{T}(mpicomm, ArrayType, n1, n2, n3)
                mpi_x = MPIStateArray{T}(mpicomm, ArrayType, n1, n2, n3)
                mpi_A = ArrayType(randn(n1 * n2, n1 * n2, n3))

                mpi_b.data[:] .= ArrayType(randn(n1 * n2 * n3))
                mpi_x.data[:] .= ArrayType(randn(n1 * n2 * n3))

                bgmres = BatchedGeneralizedMinimalResidual(
                    mpi_b,
                    n1 * n2,
                    n3,
                    M = n1 * n2,
                    atol = ϵ,
                    rtol = ϵ,
                )

                # Now define the linear operator
                function closure_linear_operator_mpi!(A, n1, n2, n3)
                    function linear_operator!(x, y)
                        alias_x = reshape(x.data, (n1, n3))
                        alias_y = reshape(y.data, (n1, n3))
                        device = array_device(x)
                        if isa(device, CPU)
                            groupsize = Threads.nthreads()
                        else # isa(device, CUDADevice)
                            groupsize = 256
                        end
                        event = Event(device)
                        event = multiply_by_A!(device, groupsize)(
                            alias_x,
                            A,
                            alias_y,
                            n1,
                            n2,
                            ndrange = n3,
                            dependencies = (event,),
                        )
                        wait(device, event)
                        nothing
                    end
                end
                linear_operator! =
                    closure_linear_operator_mpi!(mpi_A, size(mpi_A)...)

                # Now solve
                linearsolve!(
                    linear_operator!,
                    nothing,
                    bgmres,
                    mpi_x,
                    mpi_b;
                    max_iters = Inf,
                )

                # check all solutions
                norms = -zeros(n3)
                for cidx in 1:n3
                    sol =
                        Array(mpi_A[:, :, cidx]) \
                        Array(mpi_b.data[:, :, cidx])[:]
                    norms[cidx] = norm(sol - Array(mpi_x.data[:, :, cidx])[:])
                end
                @test maximum(norms) < 8000ϵ
            end

            ###
            # Test 3: Columnwise test
            ###
            @testset "(Array, $T) Columnwise Test" begin

                Random.seed!(2424)
                function closure_linear_operator_columwise!(A, tup)
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
                    for
                    i1 in 1:tup[1],
                    i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
                ]
                columnwise_A = [
                    B[i1, i2, i4, i6] + 10I
                    for
                    i1 in 1:tup[1],
                    i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
                ]
                # taking the inverse of A isn't great, but it is convenient
                columnwise_inv_A = [
                    inv(columnwise_A[i1, i2, i4, i6])
                    for
                    i1 in 1:tup[1],
                    i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
                ]
                columnwise_linear_operator! =
                    closure_linear_operator_columwise!(columnwise_A, tup)
                columnwise_inverse_linear_operator! =
                    closure_linear_operator_columwise!(columnwise_inv_A, tup)

                mpi_tup = (tup[1] * tup[2] * tup[3], tup[4], tup[5] * tup[6])
                b = randn(mpi_tup)
                x = copy(b)
                columnwise_inverse_linear_operator!(x, b)
                x +=
                    randn((tup[1] * tup[2] * tup[3], tup[4], tup[5] * tup[6])) *
                    0.1

                reshape_tuple_f = tup
                permute_tuple = (5, 3, 1, 4, 2, 6)

                bgmres = BatchedGeneralizedMinimalResidual(
                    b,
                    tup[3] * tup[5],
                    tup[1] * tup[2] * tup[4] * tup[6];
                    M = tup[3] * tup[5],
                    forward_reshape = tup,
                    forward_permute = permute_tuple,
                    atol = 10ϵ,
                    rtol = 10ϵ,
                )

                x_exact = copy(x)
                linearsolve!(
                    columnwise_linear_operator!,
                    nothing,
                    bgmres,
                    x,
                    b,
                    max_iters = tup[3] * tup[5],
                )

                columnwise_inverse_linear_operator!(x_exact, b)
                @test norm(x - x_exact) / norm(x_exact) < 1000ϵ
                columnwise_linear_operator!(x_exact, x)
                @test norm(x_exact - b) / norm(b) < 1000ϵ
            end
        end
    end
end
