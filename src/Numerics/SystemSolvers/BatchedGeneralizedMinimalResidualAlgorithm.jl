export BatchedGeneralizedMinimalResidualAlgorithm, BatchedGeneralizedMinimalResidualSolver # TODO: Remove solver export.

# TODO: Determine whether we should use PermutedDimsArray. Since permutedims!()
#       internally creates a PermutedDimsArray and calls _copy!() on it,
#       directly using a PermutedDimsArray might be much more efficient.
#       This might make it possible to eliminate Ψinit and
#       Ψ; another way to get rid of them would be to let a
#       DGModel execute batched operations. Here is how the PermutedDimsArray
#       version could work:
#     perm = invperm((batchdimindices..., remainingdimindices...))
#     batchsize = prod(dims[[batchdimindices...]])
#     nbatches = prod(dims[[remainingdimindices...]])
#     ΔQ = similar(Q)
#     PΔQs = reshape(
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

# A struct that is used to transform an array into a batched format and back.
# If forward_reshape is the same size as the original array, the reshape()
# calls do nothing, and only the permutedims!() calls have any effect.
# Otherwise, the reshape() calls make new arrays with the same underlying data.
# If the dimensions are not permuted (forward_permute == backward_permute), the
# permutedims!() calls just call copyto!(). So, if unbatched is already in a
# batched format, reshape() does nothing and permutedims!() calls copyto!(),
# which is a tad inefficient; in this situation, it would be better to make
# batched and unbatched the same array.
# TODO: Maybe write an edge case to handle the last situation more efficiently.
#       Note that it only occurs when the solution vector Q is a 2D array, each
#       of whose columns is the solution vector for an independent linear
#       system of equations (i.e., everything along dimension 1 is already
#       batched). Of course, MPIStateArrays have 3 dimensions, so this problem
#       does not apply to them.
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

"""
    BatchedGeneralizedMinimalResidualAlgorithm(
        preconditioner::Union{PreconditioningAlgorithm, Nothing} = nothing,
        atol::Union{AbstractFloat, Nothing} = nothing,
        rtol::Union{AbstractFloat, Nothing} = nothing,
        groupsize::Union{Int, Nothing} = nothing,
        coupledstates::Union{Bool, Nothing} = nothing,
        dims::Union{Dims, Nothing} = nothing,
        batchdimindices::Union{Dims, Nothing} = nothing,
        M::Union{Int, Nothing} = nothing,
        maxrestarts::Union{Int, Nothing} = nothing,
    )

Constructor for the `BatchedGeneralizedMinimalResidualAlgorithm`, which solves
an equation of the form `f(Q) = rhs`, where `f` is assumed to be a linear
function of `Q`.
    
If the equation can be broken up into smaller independent linear systems of
equal size, this algorithm can solve those linear systems in parallel, using
the restarted Generalized Minimal Residual method of Saad and Schultz (1986) to
solve each system.

## References

 - [Saad1986](@cite)

# Keyword Arguments
- `preconditioner`: right preconditioner; defaults to `NoPreconditioningAlgorithm`
- `atol`: absolute tolerance; defaults to `eps(eltype(Q))`
- `rtol`: relative tolerance; defaults to `√eps(eltype(Q))`
- `groupsize`: group size for kernel abstractions; defaults to `256`
- `coupledstates`: only used when `f` contains a `DGModel`; indicates whether
    the states in the `DGModel` are coupled to each other; defaults to `true`
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
- `M`: number of steps after which the algorithm restarts, and number of basis
    vectors in each Kyrlov subspace; defaults to `min(20, batchsize)`, where
    `batchsize` is the number of elements in each batch
- `maxrestarts`: maximum number of times the algorithm can restart; defaults to
    `cld(batchsize, M) - 1`, so that the maximum number of steps the algorithm
    can take is no less than `batchsize`, while also being as close to
    `batchsize` as possible
"""
struct BatchedGeneralizedMinimalResidualAlgorithm <: KrylovAlgorithm
    preconditioner
    atol
    rtol
    groupsize
    coupledstates
    dims
    batchdimindices
    M
    maxrestarts
    function BatchedGeneralizedMinimalResidualAlgorithm(;
        preconditioner::Union{PreconditioningAlgorithm, Nothing} = nothing,
        atol::Union{AbstractFloat, Nothing} = nothing,
        rtol::Union{AbstractFloat, Nothing} = nothing,
        groupsize::Union{Int, Nothing} = nothing,
        coupledstates::Union{Bool, Nothing} = nothing,
        dims::Union{Dims, Nothing} = nothing,
        batchdimindices::Union{Dims, Nothing} = nothing,
        M::Union{Int, Nothing} = nothing,
        maxrestarts::Union{Int, Nothing} = nothing,
    )
        @checkargs("be positive", arg -> arg > 0, atol, rtol, groupsize, M)
        @checkargs("be nonnegative", arg -> arg >= 0, maxrestarts)
        @checkargs(
            "contain one or more positive dimensions",
            arg -> length(arg) > 0 && minimum(arg) > 0,
            dims,
        )
        @checkargs(
            "contain one or more dimension indices, each of which is unique",
            arg -> length(arg) > 0 && minimum(arg) > 0 && allunique(arg),
            batchdimindices,
        )
    
        if xor(isnothing(dims), isnothing(batchdimindices))
            @warn string(
                "Both dims and batchdimindices must be specified in order to ",
                "override their default values.",
            )
        end
        if !isnothing(dims) && !isnothing(batchdimindices)
            @assert(maximum(batchdimindices) <= length(dims), string(
                "batchdimindices must contain the indices of dimensions in ",
                "dims, $dims, but it was set to $batchdimindices",
            ))
        end
    
        return new(
            preconditioner,
            atol,
            rtol,
            groupsize,
            coupledstates,
            dims,
            batchdimindices,
            M,
            maxrestarts,
        )
    end
end

function defaultbatches(Q, f!::Any, coupledstates)
    @warn string(
        "All computations will be done on one unpermuted batch.\nIf this was ",
        "not intended, consider using a GeneralizedMinimalResidualAlgorithm ",
        "instead of a BatchedGeneralizedMinimalResidualAlgorithm.",
    )
    return size(Q), Tuple(1:ndims(Q))
end

defaultbatches(Q, jvp!::JacobianVectorProduct, coupledstates) =
    defaultbatches(Q, jvp!.f!, coupledstates)

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
                "computations will be done on a single unpermuted batch.\nTo ",
                "use multiple batches, either limit the directionality or ",
                "set coupledstates = false.\nIf this is not possible, ",
                "consider using a GeneralizedMinimalResidualAlgorithm ",
                "instead of a BatchedGeneralizedMinimalResidualAlgorithm.",
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

struct BatchedGeneralizedMinimalResidualSolver{
    KT1,
    KT2,
    KT3,
    PT,
    BT,
    AT,
    BVT,
    BMT,
    FT,
} <: IterativeSolver
    batched_residual!::KT1 # kernel that is cached for efficiency
    batched_arnoldi!::KT2  # kernel that is cached for efficiency
    batched_update!::KT3   # kernel that is cached for efficiency
    preconditioner::PT     # right preconditioner
    batcher::BT            # object that is used to transform unbatched vectors
                           # into batched vectors and vice versa
    Ψinit::AT              # container for the initial values of the current
                           # Krylov basis vectors in unbatched form
    Ψ::AT                  # container for the current Krylov basis vectors in
                           # unbatched form
    Ψinits::BVT            # container for the initial value of the current
                           # Krylov basis vector of each batch; note that this
                           # should under no circumstances be replaced with
                           # view(krylovbases, :, m + 1, :), as that would make
                           # batch! more than two orders of magnitude slower
    Ψs::BVT                # container for the current Krylov basis vector of
                           # each batch; note that this could be replaced with
                           # view(krylovbases, :, m, :), which would use less
                           # memory and shave a tiny bit of time off of
                           # batched_residual! and batched_arnoldi!; however,
                           # that appears to slow down unbatch! by a slightly
                           # longer amount of time, so it is not done here
    g0s::BVT               # container for the right-hand side of the least
                           # squares problem of each batch
    Hs::BMT                # container for the Hessenberg matrix of each batch
    krylovbases::BMT       # container for the Krylov basis of each batch,
                           # which is stored as a matrix
    Ωs::BMT                # container for the Givens rotations of each batch,
                           # which are stored as a matrix
    atol::FT               # absolute tolerance
    rtol::FT               # relative tolerance
    batchsize::Int         # number of elements in each batch
    M::Int                 # number of steps after which the algorithm
                           # restarts
    maxrestarts::Int       # maximum number of times the algorithm can restart
end

function IterativeSolver(
    algorithm::BatchedGeneralizedMinimalResidualAlgorithm,
    Q,
    f!,
    rhs,
)
    check_krylov_args(Q, rhs)
    if !isnothing(algorithm.dims)
        @assert(prod(algorithm.dims) == length(Q), string(
            "dims must contain the dimensions of an array with the same ",
            "length as Q, $(length(Q)), but it was set to $(algorithm.dims)",
        ))
    end
    FT = eltype(Q)

    preconditioner = isnothing(algorithm.preconditioner) ?
        NoPreconditioningAlgorithm() : algorithm.preconditioner
    atol = isnothing(algorithm.atol) ? eps(FT) : FT(algorithm.atol)
    rtol = isnothing(algorithm.rtol) ? √eps(FT) : FT(algorithm.rtol)
    groupsize = isnothing(algorithm.groupsize) ? 256 : algorithm.groupsize # TODO: Optimize on GPU; maybe make it depend on remainingdimindices?
    coupledstates = isnothing(algorithm.coupledstates) ?
        true : algorithm.coupledstates
    
    dims, batchdimindices =
        isnothing(algorithm.dims) || isnothing(algorithm.batchdimindices) ?
        defaultbatches(Q, f!, coupledstates) :
        (algorithm.dims, algorithm.batchdimindices)
    remainingdimindices = Tuple(setdiff(1:length(dims), batchdimindices))
    batchsize = prod(dims[[batchdimindices...]])
    nbatches = prod(dims[[remainingdimindices...]])

    M = isnothing(algorithm.M) ? min(20, batchsize) : algorithm.M
    maxrestarts = isnothing(algorithm.maxrestarts) ?
        cld(length(Q), M) - 1 : algorithm.maxrestarts # TODO: Change length(Q) to batchsize after comparison testing.

    device = array_device(Q)
    rvQ = realview(Q)
    return BatchedGeneralizedMinimalResidualSolver(
        batched_residual!(device, groupsize, nbatches),
        batched_arnoldi!(device, groupsize, nbatches),
        batched_update!(device, groupsize, nbatches),
        Preconditioner(preconditioner, Q, f!),
        Batcher(dims, (batchdimindices..., remainingdimindices...)),
        similar(Q),
        similar(Q),
        similar(rvQ, batchsize, nbatches),
        similar(rvQ, batchsize, nbatches),
        similar(rvQ, M + 1, nbatches),
        similar(rvQ, M + 1, M, nbatches),
        similar(rvQ, batchsize, M + 1, nbatches),
        similar(rvQ, 2, M, nbatches),
        atol,
        rtol,
        batchsize,
        M,
        maxrestarts,
    )
end

atol(solver::BatchedGeneralizedMinimalResidualSolver) = solver.atol
rtol(solver::BatchedGeneralizedMinimalResidualSolver) = solver.rtol
maxiters(solver::BatchedGeneralizedMinimalResidualSolver) = solver.maxrestarts + 1

function residual!(
    solver::BatchedGeneralizedMinimalResidualSolver,
    threshold,
    iters,
    Q,
    f!,
    rhs,
    args...;
)
    Ψinit = solver.Ψinit
    Ψinits = solver.Ψinits
    Ψs = solver.Ψs
    g0s = solver.g0s
    krylovbases = solver.krylovbases
    
    # Compute the residual, and store its batches in Ψinits[:, :].
    f!(Ψinit, Q, args...)
    Ψinit .= rhs .- Ψinit
    batch!(Ψinits, realview(Ψinit), solver.batcher)

    # Calculate Ψs[:, :] and g0s[:, :] in batches, and store Ψs[:, :] in
    # krylovbases[:, 1, :].
    event = solver.batched_residual!(
        Ψs,
        g0s,
        krylovbases,
        Ψinits,
        solver.M,
        solver.batchsize,
    )
    wait(event)

    # Check whether the algorithm has already converged.
    residual_norm = maximum(view(g0s, 1, :)) # TODO: Make this norm(view(g0s, 1, :)), since the overall norm is the norm of the batch norms.
    has_converged = check_convergence(residual_norm, threshold, iters)

    return residual_norm, has_converged
end

function initialize!(
    solver::BatchedGeneralizedMinimalResidualSolver,
    threshold,
    iters,
    args...;
)
    return residual!(solver, threshold, iters, args...)
end

function bginnercount(solver::BatchedGeneralizedMinimalResidualSolver, threshold, iters, oters) return nothing end

function doiteration!(
    solver::BatchedGeneralizedMinimalResidualSolver,
    threshold,
    iters,
    Q,
    f!,
    rhs,
    args...;
)
    preconditioner = solver.preconditioner
    batcher = solver.batcher
    Ψinit = solver.Ψinit
    Ψ = solver.Ψ
    Ψinits = solver.Ψinits
    Ψs = solver.Ψs
    g0s = solver.g0s
    Hs = solver.Hs
    krylovbases = solver.krylovbases
    Ωs = solver.Ωs
    batchsize = solver.batchsize

    has_converged = false
    m = 0
    while !has_converged && m < solver.M
        m += 1
bginnercount(solver, threshold, iters, iters)
        # Unbatch the final value of the last Krylov basis vector, and apply the
        # right preconditioner to it.
        unbatch!(realview(Ψ), Ψs, batcher)
        preconditioner(Ψ)

        # Apply the linear operator to get the initial value of the new Krylov
        # Krylov basis vector, and store its batches in Ψinits[:, :].
        f!(Ψinit, Ψ, args...)
        batch!(Ψinits, realview(Ψinit), solver.batcher)

        # Calculate Ψs[:, :], g0s[m:m + 1, :], Hs[1:m + 1, m, :], and
        # Ωs[:, m, :] in batches, and store Ψs[:, :] in
        # krylovbases[:, m + 1, :].
        event = solver.batched_arnoldi!(
            Ψs,
            g0s,
            Hs,
            krylovbases,
            Ωs,
            Ψinits,
            m,
            batchsize,
        )
        wait(event)
        
        # Check whether the algorithm has converged.
        has_converged = check_convergence(
            maximum(abs, view(g0s, m + 1, :)), # TODO: Make this norm(view(g0s, m + 1, :)), for the same reason as above.
            threshold,
            iters,
        )
    end

    # Temporarily use Ψinit and Ψinits as containers for the update vectors.
    ΔQ = Ψinit
    PΔQs = Ψinits

    # Calculate PΔQs[:, :] in batches, overriding g0s[:, :] in the process.
    event = solver.batched_update!(g0s, PΔQs, Hs, krylovbases, m, batchsize)
    wait(event)

    # Unbatch the update vector, and unapply the right preconditioner from it.
    unbatch!(realview(ΔQ), PΔQs, batcher)
    preconditioner(ΔQ)

    # Update the solution vector.
    Q .+= ΔQ

    # Restart if the algorithm did not converge.
    has_converged && return has_converged, m
    _, has_converged, =
        residual!(solver, threshold, iters, Q, f!, rhs, args...)
    return has_converged, m
end

@kernel function batched_residual!(
    Ψs,
    g0s,
    krylovbases,
    @Const(Ψinits),
    M,
    batchsize,
)
    b = @index(Global)
    FT = eltype(g0s)

    @inbounds begin
        # Set the right-hand side vector g0s[:, b] to ∥r₀∥₂ e₁, where e₁ is the
        # unit vector along the first axis and r₀ is the initial residual of
        # batch b, which is already stored in Ψinits[:, b].
        for m in 1:M + 1
            g0s[m, b] = zero(FT)
        end
        for i in 1:batchsize
            g0s[1, b] += Ψinits[i, b]^2
        end
        g0s[1, b] = sqrt(g0s[1, b])

        # Set the Krylov basis vector Ψs[:, b] to r₀/∥r₀∥₂, and store the
        # result in krylovbases[:, 1, b].
        for i in 1:batchsize
            Ψs[i, b] = Ψinits[i, b] / g0s[1, b]
            krylovbases[i, 1, b] = Ψs[i, b]
        end
    end
end

@kernel function batched_arnoldi!(
    Ψs,
    g0s,
    Hs,
    krylovbases,
    Ωs,
    @Const(Ψinits),
    m,
    batchsize,
)
    b = @index(Global)
    FT = eltype(g0s)

    @inbounds begin
        # Initialize the Krylov basis vector Ψs[:, b] to Ψinits[:, b].
        for i in 1:batchsize
            Ψs[i, b] = Ψinits[i, b]
        end

        # Use a modified Gram-Schmidt procedure to generate a new column of the
        # Hessenberg matrix, Hs[1:m, m, b], and make Ψs[:, b] orthogonal to the
        # previous Krylov basis vectors.
        for n in 1:m
            Hs[n, m, b] = zero(FT)
            for i in 1:batchsize
                Hs[n, m, b] += Ψs[i, b] * krylovbases[i, n, b]
            end
            for i in 1:batchsize
                Ψs[i, b] -= Hs[n, m, b] * krylovbases[i, n, b]
            end
        end

        # Set Hs[m + 1, m, b] to the norm of Ψs[:, b].
        Hs[m + 1, m, b] = zero(FT)
        for i in 1:batchsize
            Hs[m + 1, m, b] += Ψs[i, b]^2
        end
        Hs[m + 1, m, b] = sqrt(Hs[m + 1, m, b])

        # Normalize Ψs[:, b], and store the result in krylovbases[:, m + 1, b].
        for i in 1:batchsize
            Ψs[i, b] /= Hs[m + 1, m, b]
            krylovbases[i, m + 1, b] = Ψs[i, b]
        end

        # TODO: Switch the negative signs on the sines after testing.

        # Apply the previous Givens rotations stored in Ωs[:, b] to the new
        # column of the Hessenberg matrix, Hs[1:m, m, b].
        for n in 1:(m - 1)
            cos = Ωs[1, n, b]
            sin = Ωs[2, n, b]
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
        Ωs[1, m, b] = cos
        Ωs[2, m, b] = sin
    end
end

@kernel function batched_update!(
    g0s,
    PΔQs,
    @Const(Hs),
    @Const(krylovbases),
    m,
    batchsize,
)
    b = @index(Global)
    FT = eltype(g0s)

    @inbounds begin
        # Set g0s[1:m, b] to UpperTriangular(H[1:m, 1:m, b]) \ g0[1:m, b]. This
        # corresponds to the vector of coefficients y that minimizes the
        # residual norm ∥rhs - f(P⁻¹ ∑ₙ yₙ P Ψₙ)∥₂, where P is the
        # preconditioner and Ψₙ is the n-th of the m Krylov basis vectors.
        for n in m:-1:1
            g0s[n, b] /= Hs[n, n, b]
            for l in 1:(n - 1)
                g0s[l, b] -= Hs[l, n, b] * g0s[n, b]
            end
        end

        # Set PΔQs[:, b] to ∑ₙ yₙ P Ψₙ.
        for i in 1:batchsize
            PΔQs[i, b] = zero(FT)
        end
        for n in 1:m
            for i in 1:batchsize
                PΔQs[i, b] += g0s[n, b] * krylovbases[i, n, b]
            end
        end
    end
end