# Test for Anderson Acceleration + Picard

using LinearAlgebra
using Test
using ClimateMachine.SystemSolvers: PicardStandardSolver, AndersonAccelerator,
        GeneralizedMinimalResidual, JacobianFreeNewtonKrylovSolver, JacobianAction,
        NoPreconditioner
import ClimateMachine.SystemSolvers

function nonlinearsolve!(
    rhs!,
    solver::Union{PicardStandardSolver,AndersonAccelerator},
    Q::AT,
    Qrhs,
    args...;
    max_nl_iters = 20,
    cvg = Ref{Bool}(),
) where {AT}

    jvp! = nothing
    preconditioner = nothing

    FT = eltype(Q)
    tol = solver.tol
    converged = false
    iters = 0

    if preconditioner === nothing
        preconditioner = NoPreconditioner()
    end

    # Initialize NLSolver, compute initial residual
    initial_residual_norm = SystemSolvers.initialize!(rhs!, Q, Qrhs, solver, args...)
    if initial_residual_norm < tol
        converged = true
    end
    converged && return iters

    while !converged && iters < max_nl_iters

        # do nonlinear iteration
        residual_norm, linear_iterations = SystemSolvers.dononlineariteration!(
            rhs!,
            jvp!,
            preconditioner,
            Q,
            Qrhs,
            solver,
            iters,
            args...,
        )
        @info iters += 1
        @info Q

        if !isfinite(residual_norm)
            error("norm of residual is not finite after $iters iterations of `dononlineariteration!`")
        end

        # Check residual_norm / norm(R0)
        # Comment: Should we check "correction" magitude?
        # ||Delta Q|| / ||Q|| ?
        relresidual = residual_norm / initial_residual_norm
        if relresidual < tol || residual_norm < tol
            @info "Picard converged in $iters iterations!"
            converged = true
        end
    end

    converged ||
        @warn "Nonlinear solver did not converge after $iters iterations"
    cvg[] = converged

    iters
end

# Example of contractive function from https://math.stackexchange.com/questions/1837585/prove-that-a-function-is-contractive
@testset "Picard NL Solver - Contractive f" begin
    ## want to solve f(Q) = Q ==> f(Q) - Q = 0
    function f!(y,x)
        y[1] = .5(exp(-x[1]) + x[2]) - x[1]
        y[2] = (exp(-x[2]) + x[1])/3 - x[2]
    end
    Qrhs = zeros(2) 
    Qnewt = zeros(2) # initial Newton guess
    Qpic = zeros(2) # initial Picard guess
    Qand = zeros(2) # initial Anderson guess

    tol = 1.e-6

    # Newton Solve
    linsolver = GeneralizedMinimalResidual(Qnewt; M=20)
    nlsolver = JacobianFreeNewtonKrylovSolver(Qnewt, linsolver; tol=tol, M=30)
    jvp! = JacobianAction(f!, Qnewt, 10e-3)
    SystemSolvers.nonlinearsolve!(
        f!,
        jvp!,
        NoPreconditioner(),
        nlsolver,
        Qnewt,
        Qrhs;
        max_newton_iters = nlsolver.M
    )

    # Picard Solve
    nlsolver = PicardStandardSolver(Qpic; tol = tol, M = 35)
    nonlinearsolve!(f!, nlsolver, Qpic, Qrhs; max_nl_iters=nlsolver.M)
    
    #Anderson Solve
    nonlinearsolver = PicardStandardSolver(Qand; tol = tol, M = 35)
    nlsolver = AndersonAccelerator(Qand, nonlinearsolver; M = 3)
    nonlinearsolve!(f!, nlsolver, Qand, Qrhs; max_nl_iters=nonlinearsolver.M)

    @test norm(Qnewt - Qpic) < tol
    @info Qnewt, Qpic, Qand
end