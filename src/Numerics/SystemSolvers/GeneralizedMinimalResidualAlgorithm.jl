export GeneralizedMinimalResidualAlgorithm

struct GeneralizedMinimalResidualAlgorithm <: IterativeAlgorithm
    preconditioner
    atol
    rtol
    maxrestarts
    M
    sarrays
end

"""
    GeneralizedMinimalResidualAlgorithm(
        atol::Union{AbstractFloat, Nothing} = nothing,
        rtol::Union{AbstractFloat, Nothing} = nothing,
        maxrestarts::Union{Int, Nothing} = nothing,
        krylovdim::Union{Int, Nothing} = nothing,
    )

Constructor for the `GeneralizedMinimalResidualAlgorithm`, which solves a
`StandardProblem` that represents the equation `f(Q) = frhs`, where `f` must be
a linear function of `Q`. This algorithm uses the restarted Generalized Minimal
Residual method of Saad and Schultz (1986).

## References

 - [Saad1986](@cite)

...
# Arguments
- `preconditioner`: right preconditioner; defaults to NoPreconditioner
- `atol`: absolute tolerance; defaults to `eps(eltype(Q))`
- `rtol`: relative tolerance; defaults to `√eps(eltype(Q))`
- `maxrestarts`: maximum number of restarts; defaults to 10
- `M`: number of steps after which the algorithm restarts, and number of basis
    vectors in the Kyrlov subspace; defaults to `min(20, length(Q))`
- `sarrays`: whether to use statically sized arrays; defaults to `true`
...
"""
function GeneralizedMinimalResidualAlgorithm(;
    preconditioner::Union{AbstractPreconditioner, Nothing} = nothing,
    atol::Union{AbstractFloat, Nothing} = nothing,
    rtol::Union{AbstractFloat, Nothing} = nothing,
    maxrestarts::Union{Int, Nothing} = nothing,
    M::Union{Int, Nothing} = nothing,
    sarrays::Union{Bool, Nothing} = nothing,
)
    return GeneralizedMinimalResidualAlgorithm(
        preconditioner,
        atol,
        rtol,
        maxrestarts,
        M,
        sarrays,
    )
end

struct GeneralizedMinimalResidualSolver{PT, KT, HT, GT, FT} <: IterativeSolver
    preconditioner::PT # right preconditioner
    krylov_basis::KT   # containers for Krylov basis vectors
    H::HT              # container for Hessenberg matrix
    g0::GT             # container for right-hand side of least squares problem
    atol::FT           # absolute tolerance
    rtol::FT           # relative tolerance
    maxrestarts::Int   # maximum number of restarts
    M::Int             # number of steps after which the algorithm restarts
end

function IterativeSolver(
    algorithm::GeneralizedMinimalResidualAlgorithm,
    problem::StandardProblem
)
    Q = problem.Q
    FT = eltype(Q)
    
    @assert size(Q) == size(problem.frhs)

    preconditioner = isnothing(algorithm.preconditioner) ? NoPreconditioner() :
        algorithm.preconditioner
    atol = isnothing(algorithm.atol) ? eps(FT) : FT(algorithm.atol)
    rtol = isnothing(algorithm.rtol) ? √eps(FT) : FT(algorithm.rtol)
    maxrestarts = isnothing(algorithm.maxrestarts) ? 10 : algorithm.maxrestarts
    M = isnothing(algorithm.M) ? min(20, length(Q)) : algorithm.M
    sarrays = isnothing(algorithm.sarrays) ? true : algorithm.sarrays

    return GeneralizedMinimalResidualSolver(
        preconditioner,
        ntuple(i -> similar(Q), M + 1),
        sarrays ? (@MArray zeros(FT, M + 1, M)) : zeros(FT, M + 1, M),
        sarrays ? (@MArray zeros(FT, M + 1, 1)) : zeros(FT, M + 1, 1),
        atol,
        rtol,
        maxrestarts,
        M,
    )
end

atol(solver::GeneralizedMinimalResidualSolver) = solver.atol
rtol(solver::GeneralizedMinimalResidualSolver) = solver.rtol
maxiters(solver::GeneralizedMinimalResidualSolver) = solver.maxrestarts

function initialize!(
    solver::GeneralizedMinimalResidualSolver,
    problem::StandardProblem,
    args...,
)
    krylov_basis = solver.krylov_basis
    g0 = solver.g0
    
    # Store the initial residual in krylov_basis[1].
    problem.f!(krylov_basis[1], problem.Q, args...)
    krylov_basis[1] .= problem.frhs .- krylov_basis[1]
    residual_norm = norm(krylov_basis[1], weighted_norm)

    g0[1] = residual_norm
    g0[2:end] .= zero(eltype(g0))
    krylov_basis[1] ./= residual_norm

    println("initialize, $(problem.Q), $(problem.frhs), $(krylov_basis[1])")

    return residual_norm
end

function doiteration!(
    solver::GeneralizedMinimalResidualSolver,
    problem::StandardProblem,
    threshold,
    iters,
    args...,
)
    preconditioner = solver.preconditioner
    krylov_basis = solver.krylov_basis
    H = solver.H
    g0 = solver.g0
    Q = problem.Q

    Ω = LinearAlgebra.Rotation{eltype(Q)}([])
    residual_norm = typemax(eltype(Q))

    has_converged = false
    j = 0
    while !has_converged && j < solver.M
        j += 1

        # Apply the right preconditioner.
        preconditioner_solve!(preconditioner, krylov_basis[j])

        # Apply the linear operator.
        problem.f!(krylov_basis[j + 1], krylov_basis[j], args...)

        # Do Arnoldi iteration using modified Gram Schmidt orthonormalization.
        for i in 1:j
            H[i, j] = dot(krylov_basis[j + 1], krylov_basis[i], weighted_norm)
            krylov_basis[j + 1] .-= H[i, j] .* krylov_basis[i]
        end
        H[j + 1, j] = norm(krylov_basis[j + 1], weighted_norm)
        krylov_basis[j + 1] ./= H[j + 1, j]

        # Apply the previous Givens rotations to the new column of H.
        @views H[1:j, j:j] .= Ω * H[1:j, j:j]

        # Compute a new Givens rotation to zero out H[j + 1, j].
        G = givens(H, j, j + 1, j)[1]

        # Apply the new rotation to H and g0.
        H .= G * H
        g0 .= G * g0

        # Compose the new rotation with the others.
        Ω = lmul!(G, Ω)
        println("inner iteration $j, $(krylov_basis[j + 1]), $(krylov_basis[j])")
        println("inner iteration $j, $H")
        println("inner iteration $j, $g0")
        println("inner iteration $j, $G")

        # Check whether the algorithm has converged.
        residual_norm = abs(g0[j + 1])
        has_converged = check_convergence(residual_norm, threshold, iters)
    end

    # Solve the triangular system.
    # TODO: Should this be `NTuple{j}(...)` or just `...`? See Solvent.jl.
    y = SVector{j}(@views UpperTriangular(H[1:j, 1:j]) \ g0[1:j])

    # Compose the solution vector.
    # TODO: Should this be `for i in 1:j Q .+= y[i] .* krylov_basis[i] end`?
    event = Event(array_device(Q))
    event = linearcombination!(array_device(Q), 256)(
        realview(Q),
        y,
        realview.(krylov_basis),
        true;
        ndrange = length(Q),
        dependencies = (event,),
    )
    wait(array_device(Q), event)

    # Un-apply the right preconditioner.
    preconditioner_solve!(preconditioner, Q)

    # Restart if the algorithm did not converge.
    # TODO: Should this be
    # if !has_converged
    #     residual_norm = initialize!(solver, problem, args...)
    #     has_converged = check_convergence(residual_norm, threshold, iters)
    # end
    has_converged || initialize!(solver, problem, args...)

    return has_converged
end
