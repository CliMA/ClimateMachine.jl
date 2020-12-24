export GeneralizedMinimalResidualAlgorithm

struct GeneralizedMinimalResidualAlgorithm <: KrylovAlgorithm
    preconditioner
    atol
    rtol
    maxrestarts
    M
    sarrays
    groupsize
end

"""
    GeneralizedMinimalResidualAlgorithm(
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
and Schultz (1986). As a Krylov subspace method, it can only solve square
linear systems, so `rhs` must have the same size as `Q`.

## References

 - [Saad1986](@cite)

# Keyword Arguments
- `preconditioner`: right preconditioner; defaults to NoPreconditioner
- `atol`: absolute tolerance; defaults to `eps(eltype(Q))`
- `rtol`: relative tolerance; defaults to `√eps(eltype(Q))`
- `maxrestarts`: maximum number of restarts; defaults to 10
- `M`: number of steps after which the algorithm restarts, and number of basis
    vectors in the Kyrlov subspace; defaults to `min(20, length(Q))`
- `sarrays`: whether to use statically sized arrays; defaults to `true`
- `groupsize`: group size for kernel abstractions; defaults to 256
"""
function GeneralizedMinimalResidualAlgorithm(;
    preconditioner::Union{AbstractPreconditioner, Nothing} = nothing,
    atol::Union{Real, Nothing} = nothing,
    rtol::Union{Real, Nothing} = nothing,
    maxrestarts::Union{Int, Nothing} = nothing,
    M::Union{Int, Nothing} = nothing,
    sarrays::Union{Bool, Nothing} = nothing,
    groupsize::Union{Int, Nothing} = nothing,
)
    @check_positive(atol, rtol, maxrestarts, M, groupsize)
    return GeneralizedMinimalResidualAlgorithm(
        preconditioner,
        atol,
        rtol,
        maxrestarts,
        M,
        sarrays,
        groupsize,
    )
end

struct GeneralizedMinimalResidualSolver{PT, KT, GT, HT, FT} <: IterativeSolver
    preconditioner::PT # right preconditioner
    krylovbasis::KT    # containers for Krylov basis vectors
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
    f!,
    Q,
    rhs,
)
    FT = eltype(Q)

    preconditioner = isnothing(algorithm.preconditioner) ? NoPreconditioner() :
        algorithm.preconditioner
    atol = isnothing(algorithm.atol) ? eps(FT) : FT(algorithm.atol)
    rtol = isnothing(algorithm.rtol) ? √eps(FT) : FT(algorithm.rtol)
    maxrestarts = isnothing(algorithm.maxrestarts) ? 10 : algorithm.maxrestarts
    M = isnothing(algorithm.M) ? min(20, length(Q)) : algorithm.M
    sarrays = isnothing(algorithm.sarrays) ? true : algorithm.sarrays
    groupsize = isnothing(algorithm.groupsize) ? 256 : algorithm.groupsize

    @assert(size(Q) == size(rhs), string(
        "Krylov subspace methods can only solve square linear systems, so Q ",
        "must have the same dimensions as rhs,\nbut their dimensions are ",
        size(Q), " and ", size(rhs), ", respectively"
    ))

    return GeneralizedMinimalResidualSolver(
        preconditioner,
        ntuple(i -> similar(Q), M + 1),
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
maxiters(solver::GeneralizedMinimalResidualSolver) = solver.maxrestarts

function initialize!(
    solver::GeneralizedMinimalResidualSolver,
    threshold,
    iters,
    f!,
    Q,
    rhs,
    args...,
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

    return residual_norm, has_converged, 1
end

function doiteration!(
    solver::GeneralizedMinimalResidualSolver,
    threshold,
    iters,
    f!,
    Q,
    rhs,
    args...,
)
    preconditioner = solver.preconditioner
    krylovbasis = solver.krylovbasis
    g0 = solver.g0
    H = solver.H

    Ω = LinearAlgebra.Rotation{eltype(Q)}([])

    has_converged = false
    j = 0
    while !has_converged && j < solver.M
        j += 1

        # Apply the right preconditioner.
        preconditioner_solve!(preconditioner, krylovbasis[j])

        # Apply the linear operator.
        f!(krylovbasis[j + 1], krylovbasis[j], args...)

        # Do Arnoldi iteration using modified Gram Schmidt orthonormalization.
        for i in 1:j
            H[i, j] = dot(krylovbasis[j + 1], krylovbasis[i], weighted_norm)
            krylovbasis[j + 1] .-= H[i, j] .* krylovbasis[i]
        end
        H[j + 1, j] = norm(krylovbasis[j + 1], weighted_norm)
        krylovbasis[j + 1] ./= H[j + 1, j]

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
    # TODO: Should this be `for i in 1:j Q .+= y[i] .* krylovbasis[i] end`?
    event = Event(array_device(Q))
    event = linearcombination!(array_device(Q), solver.groupsize)(
        realview(Q),
        y,
        realview.(krylovbasis),
        true;
        ndrange = length(Q),
        dependencies = (event,),
    )
    wait(array_device(Q), event)

    # Un-apply the right preconditioner.
    preconditioner_solve!(preconditioner, Q)

    has_converged && return has_converged, j

    # Restart if the algorithm did not converge.
    _, has_converged, initfcalls =
        initialize!(solver, threshold, iters, f!, Q, rhs, args...)
    return has_converged, j + initfcalls
end
