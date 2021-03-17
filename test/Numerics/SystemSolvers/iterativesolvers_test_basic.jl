using Test
using ClimateMachine
using ClimateMachine.SystemSolvers
import ClimateMachine.MPIStateArrays: array_device
using CUDA, KernelAbstractions
using StaticArrays, SparseArrays, LinearAlgebra, Random

Random.seed!(1)

linear_algs = (
    GMRES = (atol, rtol, maxiters) -> GeneralizedMinimalResidualAlgorithm(;
        atol = atol,
        rtol = rtol,
        maxrestarts = maxiters,
    ),
    GCR = (atol, rtol, maxiters) -> GeneralizedConjugateResidualAlgorithm(;
        atol = atol,
        rtol = rtol,
        maxrestarts = maxiters,
    ),
    CG = (atol, rtol, maxiters) -> ConjugateGradientAlgorithm(;
        atol = atol,
        rtol = rtol,
        maxiters = maxiters,
    ),
)

standard_algs = (
    Newton_FD_GMRES = (atol, rtol, maxiters) -> JacobianFreeNewtonKrylovAlgorithm(
        GeneralizedMinimalResidualAlgorithm();
        atol = atol,
        rtol = rtol,
        maxiters = maxiters,
    ),
    Newton_AD_GMRES = (atol, rtol, maxiters) -> JacobianFreeNewtonKrylovAlgorithm(
        GeneralizedMinimalResidualAlgorithm(); 
        atol = atol,
        rtol = rtol,
        maxiters = maxiters,
        autodiff=true,
    ),
    Newton_FD_GCR = (atol, rtol, maxiters) -> JacobianFreeNewtonKrylovAlgorithm(
        GeneralizedConjugateResidualAlgorithm(); 
        atol = atol,
        rtol = rtol,
        maxiters = maxiters,
    ),
    Newton_AD_GCR = (atol, rtol, maxiters) -> JacobianFreeNewtonKrylovAlgorithm(
        GeneralizedConjugateResidualAlgorithm(); 
        atol = atol,
        rtol = rtol,
        maxiters = maxiters,
        autodiff=true,
    ),
)

fixed_pt_algs = (
    Picard = (atol, rtol, maxiters) -> StandardPicardAlgorithm(;
        atol = atol,
        rtol = rtol,
        maxiters = maxiters,
    ),
    AndersonPicard = (atol, rtol, maxiters) -> AndersonAccelerationAlgorithm(
        StandardPicardAlgorithm(;
            atol = atol,
            rtol = rtol,
            maxiters = maxiters
        );
        depth = 4,
    )
)

# @testset "Solver Params" begin
#     # test rtol, atol, maxiter functions return the correct type and value
#     # test errors throw for out of bounds values

#     FT = Float32
#     f(args...) = nothing
#     Q = [FT(1)]
#     rhs = [FT(1)]

#     atol, rtol, maxiters = 1.e-4, 1.e-5, 14

#     for alg in (linear_algs..., standard_algs..., fixed_pt_algs...)
#         solver = IterativeSolver(alg(atol, rtol, maxiters), Q, f, rhs)
#         a = ClimateMachine.SystemSolvers.atol(solver)
#         r = ClimateMachine.SystemSolvers.rtol(solver)
#         m = ClimateMachine.SystemSolvers.maxiters(solver)
#         @test a == FT(atol)
#         @test r == FT(rtol)
#         @test m == maxiters
#     end

#     params = (
#         (-1.e-4, 1.e-5, 14),
#         (1.e-4, -1.e-5, 14),
#         (1.e-4, 1.e-5, -14),
#     )
#     for alg in (linear_algs..., standard_algs..., fixed_pt_algs...)
#         for (atol, rtol, maxiters) in params
#             @test_throws DomainError alg(atol, rtol, maxiters)
#         end
#     end
# end

let 
    if CUDA.has_cuda_gpu()
        Arrays = [Array, CuArray]
    else
        Arrays = [Array]
    end

    for ArrayType in Arrays
        for FT in [Float32, Float64]
            @testset "Linear Problems, $ArrayType, $FT" begin
                # Creates a Laplacian matrix based on the code from: http://math.mit.edu/~stevenj/18.303/lecture-10.html
                # construct the (M+1)xM matrix D, not including the 1/dx factor
                sdiff1(M) = [ [1.0 spzeros(1, M-1)]; spdiagm(1=>ones(M-1)) - I ]

                # make the discrete -Laplacian in 2d, with Dirichlet boundaries
                function Laplacian(Nx, Ny, Lx, Ly)
                    dx = Lx / (Nx+1)
                    dy = Ly / (Ny+1)
                    Dx = sdiff1(Nx) / dx
                    Dy = sdiff1(Ny) / dy
                    Ax = Dx' * Dx
                    Ay = Dy' * Dy
                    return kron(spdiagm(0=>ones(Ny)), Ax) + kron(Ay, spdiagm(0=>ones(Nx)))
                end

                Lx = 1
                Ly = 1
                Nx = 10
                Ny = 10
                A = ArrayType(Laplacian(Nx, Ny, Lx, Ly))
                n, _ = size(A)
                @info "Size of matrix A: ($n, $n)"
                b = ArrayType(rand(FT, n))
                x = ArrayType(rand(FT, n))

                f!(y, x) = (y .= A * x)

                atol = eps(FT)
                rtol = sqrt(eps(FT))
                maxiters = 35

                for (key, alg) in pairs(linear_algs)
                    @testset "$key" begin
                        Q0 = copy(x)
                        Q = copy(x)
                        solver = IterativeSolver(alg(atol, rtol, maxiters), Q, f!, b)
                        iters = solver(Q, f!, b)
                        @test norm(A * Q - b) < rtol * norm(A * Q0 - b)
                    end
                end
            end

            @testset "Standard Problems" begin

                atol, rtol, maxiters = 1.e-6, 1.e-6, 10

                for T in [Float32, Float64]
                    f!(fx, x) = (fx[1] = x[1]^2 + T(2); fx[2] = (x[2] - T(1))^3 + T(3))
                    x0 = rand(T, 2)
                    b = T[11, 11]
                    x_true = T[3, 3]

                    for alg in standard_algs
                        Q = copy(x0)
                        solver = IterativeSolver(alg(atol, rtol, maxiters), Q, f!, b)
                        solver(Q, f!, b)
                        @test norm(Q - x_true) < 1.e-6
                    end
                end
            end

            @testset "Fixed Point Problems" begin  

                atol, rtol, maxiters = 1.e-6, 1.e-6, 40

                for T in [Float32, Float64]
                    function f!(y,x)
                        y[1] = .5(exp(-x[1]) + x[2])
                        y[2] = (exp(-x[2]) + x[1])/3
                    end
                    x0 = rand(T, 2)
                    x_true = T[0.499336, 0.391739]
                    
                    for alg in fixed_pt_algs
                        Q = copy(x0)
                        solver = IterativeSolver(alg(atol, rtol, maxiters), Q, f!)
                        iters = solver(Q, f!)

                        @test norm(Q - x_true) < atol
                    end
                end
            end
        end
    end
end