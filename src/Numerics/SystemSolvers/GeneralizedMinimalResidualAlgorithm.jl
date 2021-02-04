export GeneralizedMinimalResidualAlgorithm

"""
    GeneralizedMinimalResidualAlgorithm(;
        preconditioner::Union{AbstractPreconditioner, Nothing} = nothing,
        atol::Union{Real, Nothing} = nothing,
        rtol::Union{Real, Nothing} = nothing,
        maxrestarts::Union{Int, Nothing} = nothing,
        M::Union{Int, Nothing} = nothing,
        sarrays::Union{Bool, Nothing} = nothing,
        groupsize::Union{Int, Nothing} = nothing,
    )

Constructor for a `GeneralizedMinimalResidualAlgorithm`, which solves an
equation of the form `f(Q) = rhs`, where `f` is assumed to be a linear function
of `Q`.

This algorithm uses the restarted Generalized Minimal Residual method of Saad
and Schultz (1986).

# Keyword Arguments
- `preconditioner`: right preconditioner; defaults to NoPreconditioner
- `atol`: absolute tolerance; defaults to `eps(eltype(Q))`
- `rtol`: relative tolerance; defaults to `√eps(eltype(Q))`
- `maxrestarts`: maximum number of restarts; defaults to `10`
- `M`: number of steps after which the algorithm restarts, and number of basis
    vectors in the Kyrlov subspace; defaults to `min(20, length(Q))`
- `sarrays`: whether to use statically sized arrays; defaults to `true`
- `groupsize`: group size for kernel abstractions; defaults to `256`

## References

 - [Saad1986](@cite)
"""
struct GeneralizedMinimalResidualAlgorithm <: KrylovAlgorithm
    preconditioner
    atol
    rtol
    maxrestarts
    M
    sarrays
    groupsize
    function GeneralizedMinimalResidualAlgorithm(;
        preconditioner::Union{AbstractPreconditioner, Nothing} = nothing,
        atol::Union{Real, Nothing} = nothing,
        rtol::Union{Real, Nothing} = nothing,
        maxrestarts::Union{Int, Nothing} = nothing,
        M::Union{Int, Nothing} = nothing,
        sarrays::Union{Bool, Nothing} = nothing,
        groupsize::Union{Int, Nothing} = nothing,
    )
        @checkargs(
            "be positive", arg -> arg > 0,
            atol, rtol, maxrestarts, M, groupsize
        )
        return new(preconditioner, atol, rtol, maxrestarts, M, sarrays, groupsize)
    end
end

struct GeneralizedMinimalResidualSolver{PT, KT, AT, GT, HT, FT} <: IterativeSolver
    preconditioner::PT # right preconditioner
    krylovbasis::KT    # containers for Krylov basis vectors
    w::AT              # work vector
    g0::GT             # container for r.h.s. of least squares problem
    H::HT              # container for Hessenberg matrix
    atol::FT           # absolute tolerance
    rtol::FT           # relative tolerance
    maxrestarts::Int   # maximum number of restarts
    M::Int             # number of steps after which the algorithm restarts
    groupsize::Int     # group size for kernel abstractions
end

function IterativeSolver(
    algorithm::GeneralizedMinimalResidualAlgorithm,
    Q,
    f!,
    rhs,
)
    @assert(size(Q) == size(rhs), string(
            "Must solve a square system, Q must have the same dimensions as rhs,",
            "\nbut their dimensions are $(size(Q)) and $(size(rhs)), respectively."
    ))

    FT = eltype(Q)

    preconditioner = isnothing(algorithm.preconditioner) ? NoPreconditioner() :
        algorithm.preconditioner
    atol = isnothing(algorithm.atol) ? eps(FT) : FT(algorithm.atol)
    rtol = isnothing(algorithm.rtol) ? √eps(FT) : FT(algorithm.rtol)
    maxrestarts = isnothing(algorithm.maxrestarts) ? 10 : algorithm.maxrestarts
    M = isnothing(algorithm.M) ? min(20, length(Q)) : algorithm.M
    sarrays = isnothing(algorithm.sarrays) ? true : algorithm.sarrays
    groupsize = isnothing(algorithm.groupsize) ? 256 : algorithm.groupsize

    return GeneralizedMinimalResidualSolver(
        preconditioner,
        ntuple(i -> similar(Q), M + 1),
        similar(Q),
        sarrays ? (@MArray zeros(FT, M + 1, 1)) : zeros(FT, M + 1, 1),
        sarrays ? (@MArray zeros(FT, M + 1, M)) : zeros(FT, M + 1, M),
        atol,
        rtol,
        maxrestarts,
        M,
        groupsize,
    )
end

atol(solver::GeneralizedMinimalResidualSolver) = solver.atol
rtol(solver::GeneralizedMinimalResidualSolver) = solver.rtol
maxiters(solver::GeneralizedMinimalResidualSolver) = solver.maxrestarts + 1

function residual!(
    solver::GeneralizedMinimalResidualSolver,
    threshold,
    iters,
    Q,
    f!,
    rhs,
    args...;
)
    krylovbasis = solver.krylovbasis
    g0 = solver.g0
    
    # Store the residual in krylovbasis[1].
    f!(krylovbasis[1], Q, args...)
    krylovbasis[1] .= rhs .- krylovbasis[1]

    residual_norm = norm(krylovbasis[1], weighted_norm)
    has_converged = check_convergence(residual_norm, threshold, iters)

    # Normalize krylovbasis[1] and update g0.
    if !has_converged
        krylovbasis[1] ./= residual_norm
        g0[1] = residual_norm
        g0[2:end] .= zero(eltype(g0))
    end

    return residual_norm, has_converged
end

function initialize!(
    solver::GeneralizedMinimalResidualSolver,
    threshold,
    iters,
    args...;
)
    return residual!(solver, threshold, iters, args...)
end

function doiteration!(
    solver::GeneralizedMinimalResidualSolver,
    threshold,
    iters,
    Q,
    f!,
    rhs,
    args...;
)
    preconditioner = solver.preconditioner
    krylovbasis = solver.krylovbasis
    g0 = solver.g0
    H = solver.H
    w = solver.w
    w .= krylovbasis[1] # initialize work vector

    Ω = LinearAlgebra.Rotation{eltype(Q)}([])

    has_converged = false
    j = 0
    while !has_converged && j < solver.M
        j += 1

        # Apply the right preconditioner.
        preconditioner_solve!(preconditioner, w)

        # Apply the linear operator.
        f!(krylovbasis[j + 1], w, args...)

        # Do Arnoldi iteration using modified Gram Schmidt orthonormalization.
        for i in 1:j
            H[i, j] = dot(krylovbasis[j + 1], krylovbasis[i], weighted_norm)
            krylovbasis[j + 1] .-= H[i, j] .* krylovbasis[i]
        end
        H[j + 1, j] = norm(krylovbasis[j + 1], weighted_norm)
        krylovbasis[j + 1] ./= H[j + 1, j]
        w .= krylovbasis[j + 1]

        # Apply the previous Givens rotations to the new column of H.
        @views H[1:j, j:j] .= Ω * H[1:j, j:j]

        # Compute a new Givens rotation to zero out H[j + 1, j].
        G = givens(H, j, j + 1, j)[1]

        # Apply the new rotation to H and g0.
        H .= G * H
        g0 .= G * g0

        # Compose the new rotation with the others.
        Ω = lmul!(G, Ω)

        # Check whether the algorithm has converged.
        has_converged = check_convergence(abs(g0[j + 1]), threshold, iters)
    end

    # Solve the triangular system.
    # TODO: Should this be `NTuple{j}(...)` or just `...`? See Solvent.jl.
    y = SVector{j}(@views UpperTriangular(H[1:j, 1:j]) \ g0[1:j])

    # Compose the solution vector.
    ΔQ = krylovbasis[1]
    ΔQ .= y[1] .* krylovbasis[1]
    for i in 2:j
        ΔQ .+= krylovbasis[i] .* y[i]
    end

    # event = Event(array_device(Q))
    # event = linearcombination!(array_device(Q), solver.groupsize)(
    #     realview(Q),
    #     view(y),
    #     realview.(krylovbasis),
    #     true;
    #     ndrange = length(Q),
    #     dependencies = (event,),
    # )
    # wait(array_device(Q), event)

    # Un-apply the right preconditioner.
    preconditioner_solve!(preconditioner, ΔQ)
    Q .+= ΔQ

    has_converged && return has_converged, j

    # Restart if the algorithm did not converge.
    _, has_converged = residual!(solver, threshold, iters, Q, f!, rhs, args...)
    return has_converged, j
end
