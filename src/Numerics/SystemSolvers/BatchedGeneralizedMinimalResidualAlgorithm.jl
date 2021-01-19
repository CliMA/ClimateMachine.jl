export BatchedGeneralizedMinimalResidualAlgorithm

# TODO: Determine whether we should use PermutedDimsArray. Since permutedims!()
#       internally creates a PermutedDimsArray and calls _copy!() on it,
#       directly using a PermutedDimsArray might be much more efficient.
#       This might make it possible to eliminate basisveccurr and
#       basisvecprev; another way to get rid of them would be to let a
#       DGModel execute batched operations. Here is how the PermutedDimsArray
#       version could work:
#     perm = invperm((batchdimindices..., remainingdimindices...))
#     batchsize = prod(dims[[batchdimindices...]])
#     nbatches = prod(dims[[remainingdimindices...]])
#     ΔQ = similar(Q)
#     ΔQs = reshape(
#         PermutedDimsArray(reshape(realview(ΔQ), dims), perm),
#         (batchsize, nbatches)
#     )
#     krylovbasis = ntuple(i -> similar(Q), M + 1)
#     krylovbases = ntuple(
#         i -> reshape(
#             PermutedDimsArray(reshape(realview(krylovbasis[i]), dims), perm),
#             (batchsize, nbatches)
#         ),
#         M + 1
#     )

# A useful struct that can transform an array into a batched format and back.
# If forward_reshape is the same as the array's original size, the reshape()
# calls do nothing, and only the permutedims!() calls have any effect.
# Otherwise, the reshape() calls make new arrays with the same underlying data.
# If the dimensions are not permuted (forward_permute == backward_permute), the
# permutedims!() calls just call copyto!(). If unbatched is already in batched
# form, reshape() does nothing and permutedims!() calls copyto!(), which is
# quite inefficient; it would be better to make batched and unbatched the same
# array in this situation.
# TODO: Maybe write an edge case to handle the last situation more efficiently.
struct Batcher{T}
    forward_reshape::T
    forward_permute::T
    backward_reshape::T
    backward_permute::T
end
function Batcher(forward_reshape, forward_permute)
    return Batcher(
        forward_reshape,
        forward_permute,
        forward_reshape[[forward_permute...]],
        invperm(forward_permute),
    )
end
function batch!(batched, unbatched, b::Batcher)
    reshaped_batched = reshape(batched, b.backward_reshape)
    reshaped_unbatched = reshape(unbatched, b.forward_reshape)
    permutedims!(reshaped_batched, reshaped_unbatched, b.forward_permute)
    return nothing
end
function unbatch!(unbatched, batched, b::Batcher)
    reshaped_batched = reshape(batched, b.backward_reshape)
    reshaped_unbatched = reshape(unbatched, b.forward_reshape)
    permutedims!(reshaped_unbatched, reshaped_batched, b.backward_permute)
    return nothing
end

struct BatchedGeneralizedMinimalResidualAlgorithm <: KrylovAlgorithm
    preconditioner
    atol
    rtol
    maxrestarts
    M
    coupledstates
    dims
    batchdimindices
    groupsize
end

"""
    BatchedGeneralizedMinimalResidualAlgorithm(
        preconditioner::Union{AbstractPreconditioner, Nothing} = nothing,
        atol::Union{AbstractFloat, Nothing} = nothing,
        rtol::Union{AbstractFloat, Nothing} = nothing,
        maxrestarts::Union{Int, Nothing} = nothing,
        M::Union{Int, Nothing} = nothing,
        coupledstates::Union{Bool, Nothing} = nothing,
        dims::Union{Dims, Nothing} = nothing,
        batchdimindices::Union{Dims, Nothing} = nothing,
        groupsize::Union{Int, Nothing} = nothing,
    )

Constructor for the `BatchedGeneralizedMinimalResidualAlgorithm`, which solves
an equation of the form `f(Q) = rhs`, where `f` is assumed to be a linear
function of `Q`.
    
If the equation can be broken up into smaller independent linear systems of
equal size, this algorithm can solve those linear systems in parallel, using
the restarted Generalized Minimal Residual method of Saad and Schultz (1986) to
solve each system. As a Krylov subspace method, it can only solve square linear
systems, so `rhs` must have the same size as `Q`.

## References

 - [Saad1986](@cite)

# Keyword Arguments
- `preconditioner`: right preconditioner; defaults to `NoPreconditioner`
- `atol`: absolute tolerance; defaults to `eps(eltype(Q))`
- `rtol`: relative tolerance; defaults to `√eps(eltype(Q))`
- `maxrestarts`: maximum number of restarts; defaults to `10`
- `M`: number of steps after which the algorithm restarts, and number of basis
    vectors in each Kyrlov subspace; defaults to `min(20, length(Q))`
- `coupledstates`: only relevant when `f` uses a `DGModel`; denotes whether the
    states in the `DGModel` are coupled to each other; defaults to `true`
- `dims`: dimensions from which to select batch dimensions; does not need to
    match the actual dimensions of `Q`, but must have the property that
    `prod(dims) == length(Q)`; defaults to `size(Q)` when `f` does not use a
    `DGModel`, `(npoints, nstates, nelems)` when `f` uses a `DGModel` with
    `EveryDirection`, and `(nhorzpoints, nvertpoints, nstates, nvertelems,
    nhorzelems)` when `f` uses a `DGModel` with `HorizontalDirection` or
    `VerticalDirection`; default value will be used unless `batchdimindices` is
    also specified
- `batchdimindices`: indices of dimensions in `dims` that form each batch; is
    assumed to define batches that form independent linear systems; defaults to
    `Tuple(1:ndims(Q))` when `f` does not use a `DGModel`, `(1, 2, 3)` or
    `(1, 3)` when `f` uses a `DGModel` with `EveryDirection` (the former for
    coupled states and the latter for uncoupled states), `(1, 3, 5)` or
    `(1, 5)` when `f` uses a `DGModel` with `HorizontalDirection`, and
    `(2, 3, 4)` or `(2, 4)` when `f` uses a `DGModel` with `VerticalDirection`;
    default value will be used unless `dims` is also specified
- `groupsize`: group size for kernel abstractions; defaults to `256`
"""
function BatchedGeneralizedMinimalResidualAlgorithm(;
    preconditioner::Union{AbstractPreconditioner, Nothing} = nothing,
    atol::Union{AbstractFloat, Nothing} = nothing,
    rtol::Union{AbstractFloat, Nothing} = nothing,
    maxrestarts::Union{Int, Nothing} = nothing,
    M::Union{Int, Nothing} = nothing,
    coupledstates::Union{Bool, Nothing} = nothing,
    dims::Union{Dims, Nothing} = nothing,
    batchdimindices::Union{Dims, Nothing} = nothing,
    groupsize::Union{Int, Nothing} = nothing,
)
    @checkargs(
        "be positive", arg -> arg > 0,
        atol, rtol, maxrestarts, M, groupsize
    )
    @checkargs(
        "contain positive values", arg -> length(arg) > 0 && minimum(arg) > 0,
        dims, batchdimindices
    )
    @checkargs(
        "be a tuple of unique indices", arg -> allunique(arg),
        batchdimindices
    )

    if xor(isnothing(dims), isnothing(batchdimindices))
        @warn string(
            "Both dims and batchdimindices must be specified in order to ",
            "override default values."
        )
    end
    if !isnothing(dims) && !isnothing(batchdimindices)
        @assert(maximum(batchdimindices) <= length(dims), string(
            "batchdimindices must contain a subset of the indices of ",
            "dimensions in dims, ", dims, ", but it was set to ",
            batchdimindices
        ))
    end

    return BatchedGeneralizedMinimalResidualAlgorithm(
        preconditioner,
        atol,
        rtol,
        maxrestarts,
        M,
        coupledstates,
        dims,
        batchdimindices,
        groupsize,
    )
end

function defaultbatches(Q, f!::Any, coupledstates)
    @warn string(
        "All computations will be done on a single batch.\nIf this was not ",
        "intended, consider using a GeneralizedMinimalResidualAlgorithm ",
        "instead of a BatchedGeneralizedMinimalResidualAlgorithm."
    )
    return size(Q), Tuple(1:ndims(Q))
end
function defaultbatches(Q, op::EulerOperator, coupledstates)
    return defaultbatches(Q, op.f!, coupledstates)
end
function defaultbatches(Q, dg::DGModel, coupledstates)
    direction = dg.direction
    grid = dg.grid
    topology = grid.topology
    N = polynomialorders(grid)
    nvertpoints = N[end] + 1
    nhorzpoints = length(N) == 3 ? (N[1] + 1) * (N[2] + 1) : N[1] + 1
    nstates = size(Q)[2] # This could be obtained from dg with number_states.
    nelems = length(topology.realelems)
    nvertelems = topology.stacksize
    nhorzelems = div(nelems, nvertelems)

    if isa(direction, EveryDirection)
        dims = (nhorzpoints * nvertpoints, nstates, nelems)
        if coupledstates
            @warn string(
                "DGModel uses EveryDirection and has coupled states, so all ",
                "computations will be done on a single batch.\nTo use ",
                "multiple batches, either limit the directionality or set ",
                "coupledstates = false.\nIf this is not possible, consider ",
                "using a GeneralizedMinimalResidualAlgorithm instead of a ",
                "BatchedGeneralizedMinimalResidualAlgorithm."
            )
            batchdimindices = (1, 2, 3)
        else
            batchdimindices = (1, 3)
        end
    else
        dims = (nhorzpoints, nvertpoints, nstates, nvertelems, nhorzelems)
        if isa(direction, HorizontalDirection)
            batchdimindices = coupledstates ? (1, 3, 5) : (1, 5)
        else # VerticalDirection
            batchdimindices = coupledstates ? (2, 3, 4) : (2, 4)
        end
    end

    return dims, batchdimindices
end

struct BatechedGeneralizedMinimalResidualSolver{BT, PT, AT, BMT, BAT, FT} <:
        IterativeSolver
    batcher::BT        # batcher that can transform, e.g., basisveccurr to Qs
    preconditioner::PT # right preconditioner
    basisvecprev::AT   # container for previous unbatched Krylov basis vector
    basisveccurr::AT   # container for current unbatched Krylov basis vector
    krylovbases::BMT   # container for Krylov basis vectors of each batch
    g0s::BAT           # container for r.h.s. of least squares problem of each batch
    Hs::BMT            # container for Hessenberg matrix of each batch
    Ωs::BAT            # container for Givens rotation matrix of each batch
    ΔQs::BAT           # container for solution vector of each batch
    atol::FT           # absolute tolerance
    rtol::FT           # relative tolerance
    maxrestarts::Int   # maximum number of restarts
    M::Int             # number of steps after which the algorithm restarts
    batchsize::Int     # number of elements in each batch
    nbatches::Int      # number of batches
    groupsize::Int     # group size for kernel abstractions
end

function IterativeSolver(
    algorithm::BatchedGeneralizedMinimalResidualAlgorithm,
    Q,
    f!,
    rhs,
)
    @assert(size(Q) == size(rhs), string(
        "Must solve a square system, Q must have the same dimensions as rhs,",
        "\nbut their dimensions are $(size(Q)) and $(size(rhs)), respectively."
    ))
    @assert(prod(dims) == length(Q), string(
        "dims must contain the dimensions of an array with the same length ",
        "as Q, ", length(Q), ", but it was set to ", dims
    ))
    
    FT = eltype(Q)

    preconditioner = isnothing(algorithm.preconditioner) ?
        NoPreconditioner() : algorithm.preconditioner
    atol = isnothing(algorithm.atol) ? eps(FT) : FT(algorithm.atol)
    rtol = isnothing(algorithm.rtol) ? √eps(FT) : FT(algorithm.rtol)
    maxrestarts = isnothing(algorithm.maxrestarts) ? 10 : algorithm.maxrestarts
    M = isnothing(algorithm.M) ? min(20, length(Q)) : algorithm.M
    coupledstates = isnothing(algorithm.coupledstates) ?
        true : algorithm.coupledstates
    dims, batchdimindices =
        isnothing(algorithm.dims) || isnothing(algorithm.batchdimindices) ?
        defaultbatches(Q, f!, coupledstates) :
        algorithm.dims, algorithm.batchdimindices
    groupsize = isnothing(algorithm.groupsize) ? 256 : algorithm.groupsize

    remainingdimindices = Tuple(setdiff(1:length(dims), batchdimindices))
    batchsize = prod(dims[[batchdimindices...]])
    nbatches = prod(dims[[remainingdimindices...]])
    rvQ = realview(Q)
    return BatechedGeneralizedMinimalResidualSolver(
        Batcher(dims, (batchdimindices..., remainingdimindices...)),
        preconditioner,
        similar(Q),
        similar(Q),
        similar(rvQ, batchsize, M + 1, nbatches),
        similar(rvQ, M + 1, nbatches),
        similar(rvQ, M + 1, M, nbatches),
        similar(rvQ, 2 * M, nbatches),
        similar(rvQ, batchsize, nbatches),
        atol,
        rtol,
        maxrestarts,
        M,
        batchsize,
        nbatches,
        groupsize,
    )
end

atol(solver::BatechedGeneralizedMinimalResidualSolver) = solver.atol
rtol(solver::BatechedGeneralizedMinimalResidualSolver) = solver.rtol
maxiters(solver::BatechedGeneralizedMinimalResidualSolver) = solver.maxrestarts

function residual!(
    solver::BatechedGeneralizedMinimalResidualSolver,
    threshold,
    iters,
    Q,
    f!,
    rhs,
    args...;
)
    basisveccurr = solver.basisveccurr
    krylovbases = solver.krylovbases
    g0s = solver.g0s
    device = array_device(Q)
    
    # Compute the residual and store its batches in krylovbases[:, 1, :].
    f!(basisveccurr, Q, args...)
    basisveccurr .= rhs .- basisveccurr
    batch!(view(krylovbases, :, 1, :), realview(basisveccurr), solver.batcher)

    # Calculate krylovbases[:, 1, :] and g0s[:, :] in batches.
    event = Event(device)
    event = batched_residual!(device, solver.groupsize)(
        krylovbases,
        g0s,
        solver.M,
        solver.batchsize;
        ndrange = solver.nbatches,
        dependencies = (event,),
    )
    wait(device, event)

    # Check whether the algorithm has already converged.
    residual_norm = maximum(view(g0s, 1, :)) # TODO: Make this norm(view(g0s, 1, :)), since the overall norm is the norm of the batch norms.
    has_converged = check_convergence(residual_norm, threshold, iters)

    return residual_norm, has_converged, 1
end

function initialize!(
    solver::BatechedGeneralizedMinimalResidualSolver,
    threshold,
    iters,
    args...;
)
    return residual!(solver, threshold, iters, args...)
end

function doiteration!(
    solver::BatechedGeneralizedMinimalResidualSolver,
    threshold,
    iters,
    Q,
    f!,
    rhs,
    args...;
)
    preconditioner = solver.preconditioner
    basisvecprev = solver.basisvecprev
    basisveccurr = solver.basisveccurr
    krylovbases = solver.krylovbases
    g0s = solver.g0s
    Hs = solver.Hs
    Ωs = solver.Ωs
    ΔQs = solver.ΔQs
    device = array_device(Q)

    has_converged = false
    m = 0
    while !has_converged && m < solver.M
        m += 1

        # Unbatch the previous Krylov basis vector.
        unbatch!(
            realview(basisvecprev),
            view(krylovbases, :, m, :),
            solver.batcher
        )

        # Apply the right preconditioner to the previous Krylov basis vector.
        preconditioner_solve!(preconditioner, basisvecprev)

        # Apply the linear operator to get the current Krylov basis vector.
        f!(basisveccurr, basisvecprev, args...)

        # Batch the current Krylov basis vector.
        batch!(
            view(krylovbases, :, m + 1, :),
            realview(basisveccurr),
            solver.batcher
        )

        # Calculate krylovbases[:, m + 1, :], g0s[m:m + 1, :],
        # Hs[1:m + 1, m, :], and Ωs[2 * m - 1:2 * m, :] in batches.
        event = Event(device)
        event = batched_arnoldi_process!(device, solver.groupsize)(
            krylovbases,
            g0s,
            Hs,
            Ωs,
            m,
            solver.batchsize;
            ndrange = solver.nbatches,
            dependencies = (event,),
        )
        wait(device, event)

        # Check whether the algorithm has converged.
        has_converged = check_convergence(
            maximum(abs.(view(g0s, m + 1, :))), # TODO: Make this norm(view(g0s, m + 1, :)), for the same reason as above.
            threshold,
            iters
        )
    end

    # Calculate ΔQs[:, :] in batches, overriding g0s[:, :] in the process.
    event = Event(device)
    event = batched_update!(device, solver.groupsize)(
        krylovbases,
        g0s,
        Hs,
        ΔQs,
        m,
        solver.batchsize;
        ndrange = solver.nbatches,
        dependencies = (event,),
    )
    wait(device, event)

    # Temporarily use basisvecprev as container for the update vector ΔQ.
    ΔQ = basisvecprev

    # Unbatch the update vector.
    unbatch!(realview(ΔQ), ΔQs, solver.batcher)

    # Unapply the right preconditioner.
    preconditioner_solve!(preconditioner, ΔQ)

    # Update the solution vector.
    Q .+= ΔQ

    # Restart if the algorithm did not converge.
    has_converged && return has_converged, m, m
    _, has_converged, initfcalls =
        residual!(solver, threshold, iters, Q, f!, rhs, args...)
    return has_converged, m + initfcalls, m
end

@kernel function batched_residual!(krylovbases, g0s, M, batchsize)
    b = @index(Global)
    FT = eltype(g0s)

    @inbounds begin
        # Set the r.h.s. vector g0s[:, b] to ∥r₀∥₂ e₁, where e₁ is the unit
        # vector along the first axis and r₀ is the initial residual of batch
        # b, which is already stored in krylovbases[:, 1, b].
        for m in 1:M + 1
            g0s[m, b] = zero(FT)
        end
        for i in 1:batchsize
            g0s[1, b] += krylovbases[i, 1, b]^2
        end
        g0s[1, b] = sqrt(g0s[1, b])

        # Normalize the initial Krylov basis vector krylovbases[:, 1, b].
        for i in 1:batchsize
            krylovbases[i, 1, b] /= g0s[1, b]
        end
    end
end

@kernel function batched_arnoldi_process!(
    krylovbases,
    g0s,
    Hs,
    Ωs,
    m,
    batchsize,
)
    b = @index(Global)
    FT = eltype(g0s)

    @inbounds begin
        # Use a modified Gram-Schmidt procedure to generate a new column of the
        # Hessenberg matrix, H[1:m, m, b]. Orthogonalize the new Krylov basis
        # vector krylovbases[:, m + 1, b] with respect to the previous ones.
        for n in 1:m
            Hs[n, m, b] = zero(FT)
            for i in 1:batchsize
                Hs[n, m, b] += krylovbases[i, m + 1, b] * krylovbases[i, n, b]
            end
            for i in 1:batchsize
                krylovbases[i, m + 1, b] -= Hs[n, m, b] * krylovbases[i, n, b]
            end
        end

        # Set Hs[m + 1, m, b] to the norm of krylovbases[:, m + 1, b].
        Hs[m + 1, m, b] = zero(FT)
        for i in 1:batchsize
            Hs[m + 1, m, b] += krylovbases[i, m + 1, b]^2
        end
        Hs[m + 1, m, b] = sqrt(Hs[m + 1, m, b])

        # Normalize krylovbases[:, m + 1, b].
        for i in 1:batchsize
            krylovbases[i, m + 1, b] /= Hs[m + 1, m, b]
        end

        # TODO: Switch the negative signs on the sines after testing.

        # Apply the previous Givens rotations stored in Ωs[:, b] to the new
        # column of the Hessenberg matrix, Hs[1:m, m, b].
        for n in 1:(m - 1)
            cos = Ωs[2 * n - 1, b]
            sin = Ωs[2 * n, b]
            temp = -sin * Hs[n, m, b] + cos * Hs[n + 1, m, b]
            Hs[n, m, b] = cos * Hs[n, m, b] + sin * Hs[n + 1, m, b]
            Hs[n + 1, m, b] = temp
        end

        # Compute a new Givens rotation so that Hs[m + 1, m, b] is zeroed out:
        #     |  cos sin | |   Hs[m, m, b]   |  = | Hs[m, m, b]' |
        #     | -sin cos | | Hs[m + 1, m, b] |  = |      0       |
        cos = Hs[m, m, b]
        sin = Hs[m + 1, m, b]
        temp = sqrt(cos^2 + sin^2)
        cos /= temp
        sin /= temp
        
        # Apply the new Givens rotation to Hs[1:m + 1, m, b].
        Hs[m, m, b] = temp
        Hs[m + 1, m, b] = zero(FT)

        # Apply the new Givens rotation to the r.h.s. vector g0s[1:m + 1, b].
        temp = -sin * g0s[m, b] + cos * g0s[m + 1, b]
        g0s[m, b] = cos * g0s[m, b] + sin * g0s[m + 1, b]
        g0s[m + 1, b] = temp

        # Store the new Givens rotation in Ωs[:, b].
        Ωs[2 * m - 1, b] = cos
        Ωs[2 * m, b] = sin
    end
end

@kernel function batched_update!(
    krylovbases,
    g0s,
    Hs,
    ΔQs,
    m,
    batchsize,
)
    b = @index(Global)
    FT = eltype(g0s)

    @inbounds begin
        # Set g0s[1:m, b] to UpperTriangular(H[1:m, 1:m, b]) \ g0[1:m, b]. This
        # corresponds to the vector of coefficients yₙ that minimizes the
        # residual norm ∥rhs - f(∑ₙ yₙ Ψₙ)∥₂, where Ψₙ is the n-th of the m
        # Krylov basis vectors.
        for n in m:-1:1
            g0s[n, b] /= Hs[n, n, b]
            for l in 1:(n - 1)
                g0s[l, b] -= Hs[l, n, b] * g0s[n, b]
            end
        end

        # Set ΔQs[:, b] to the GMRES solution vector ∑ₙ yₙ Ψₙ.
        for i in 1:batchsize
            ΔQs[i, b] = zero(FT)
        end
        for n in 1:m
            for i in 1:batchsize
                ΔQs[i, b] += g0s[n, b] * krylovbases[i, n, b]
            end
        end
    end
end