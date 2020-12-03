# Test for Anderson Acceleration + Picard

using LinearAlgebra
using Test
using ClimateMachine.SystemSolvers: PicardStandardSolver, AndersonAccelerator,
        GeneralizedMinimalResidual, JacobianFreeNewtonKrylovSolver, JacobianAction,
        NoPreconditioner, nonlinearsolve!

# Example of contractive function from https://math.stackexchange.com/questions/1837585/prove-that-a-function-is-contractive
@testset "Picard NL Solver - Contractive f" begin
    ## want to solve f(Q) = Q ==> f(Q) - Q = 0
    function f!(y,x)
        y[1] = .5(exp(-x[1]) + x[2]) - x[1]
        y[2] = (exp(-x[2]) + x[1])/3 - x[2]
    end
    ## want to solve f(Q) = Q
    function fpic!(y,x)
        y[1] = .5(exp(-x[1]) + x[2])
        y[2] = (exp(-x[2]) + x[1])/3
    end
    Qrhs = zeros(2) 
    Qnewt = zeros(2) # initial Newton guess
    Qpic = zeros(2) # initial Picard guess
    Qand = zeros(2) # initial Anderson guess

    # Newton Solve
    linsolver = GeneralizedMinimalResidual(Qnewt; M=20)
    nlsolver = JacobianFreeNewtonKrylovSolver(Qnewt, linsolver)
    nlsolver.jvp!.rhs! = f!
    nonlinearsolve!(
        nlsolver,
        f!,
        Qnewt,
        Qrhs;
        preconditioner=NoPreconditioner(),
    )

    # Picard Solve
    nlsolver = PicardStandardSolver(Qpic)
    nonlinearsolve!(nlsolver, fpic!, Qpic, Qrhs)
    
    #Anderson Solve
    nonlinearsolver = PicardStandardSolver(Qand)
    nlsolver = AndersonAccelerator(Qand, nonlinearsolver; depth = 3)
    nonlinearsolve!(nlsolver, fpic!, Qand, Qrhs)

    # @test norm(Qnewt - Qpic) < tol
    @info Qnewt, Qpic, Qand
    @info norm(Qnewt .- Qpic), norm(Qnewt .- Qand)
end