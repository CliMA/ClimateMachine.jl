
using CUDA

export BatchedGeneralizedMinimalResidual

"""
    BatchedGeneralizedMinimalResidual(
        Q,
        dofperbatch,
        Nbatch;
        M = min(20, length(Q)),
        rtol = √eps(eltype(AT)),
        atol = eps(eltype(AT)),
        forward_reshape = size(Q),
        forward_permute = Tuple(1:length(size(Q))),
    )

# BGMRES
This is an object for solving batched linear systems using the GMRES algorithm.
The constructor parameter `M` is the number of steps after which the algorithm
is restarted (if it has not converged), `Q` is a reference state used only
to allocate the solver internal state, `dofperbatch` is the size of each batched
system (assumed to be the same throughout), `Nbatch` is the total number
of independent linear systems, and `rtol` specifies the convergence
criterion based on the relative residual norm (max across all batched systems).
The argument `forward_reshape` is a tuple of integers denoting the reshaping
(if required) of the solution vectors for batching the Arnoldi routines.
The argument `forward_permute` describes precisely which indices of the
array `Q` to permute. This object is intended to be passed to
the [`linearsolve!`](@ref) command.

This uses a batched-version of the restarted Generalized Minimal Residual method
of Saad and Schultz (1986).

# Note
Eventually, we'll want to do something like this:

    i = @index(Global)
    linearoperator!(Q[:, :, :, i], args...)

This will help stop the need for constantly
reshaping the work arrays. It would also potentially
save us some memory.
"""
mutable struct BatchedGeneralizedMinimalResidual{
    I,
    T,
    AT,
    BKT,
    OmT,
    HT,
    gT,
    sT,
    resT,
    res0T,
    FRS,
    FPR,
    BRS,
    BPR,
} <: AbstractIterativeSystemSolver

    "global Krylov basis at present step"
    krylov_basis::AT
    "global Krylov basis at previous step"
    krylov_basis_prev::AT
    "global batched Krylov basis"
    batched_krylov_basis::BKT
    "Storage for the Givens rotation matrices"
    Ω::OmT
    "Hessenberg matrix in each column"
    H::HT
    "rhs of the least squares problem in each column"
    g0::gT
    "The GMRES iterate in each batched column"
    sol::sT
    "Relative tolerance"
    rtol::T
    "Absolute tolerance"
    atol::T
    "Maximum number of GMRES iterations (global across all columns)"
    max_iter::I
    "total number of batched columns"
    batch_size::I
    "total number of dofs per batched column"
    dofperbatch::I
    "residual norm in each column"
    resnorms::resT
    "initial residual norm in each column"
    initial_resnorms::res0T
    forward_reshape::FRS
    forward_permute::FPR
    backward_reshape::BRS
    backward_permute::BPR

    function BatchedGeneralizedMinimalResidual(
        Q::AT,
        dofperbatch,
        Nbatch;
        M = min(20, length(Q)),
        rtol = √eps(eltype(AT)),
        atol = eps(eltype(AT)),
        forward_reshape = size(Q),
        forward_permute = Tuple(1:length(size(Q))),
    ) where {AT}
        # Get ArrayType information
        if isa(array_device(Q), CPU)
            ArrayType = Array
        else
            # Sanity check since we don't support anything else
            @assert isa(array_device(Q), CUDADevice)
            ArrayType = CuArray
        end

        # FIXME: If we can batch the application of linearoperator!, then we dont
        # need these two temporary work vectors (unpermuted/reshaped)
        krylov_basis = similar(Q)
        krylov_basis_prev = similar(Q)

        FT = eltype(AT)
        # Create storage for holding the batched Krylov basis
        batched_krylov_basis =
            fill!(ArrayType{FT}(undef, M + 1, dofperbatch, Nbatch), 0)

        # Create storage for doing the batched Arnoldi process
        Ω = fill!(ArrayType{FT}(undef, Nbatch, 2 * M), 0)
        H = fill!(ArrayType{FT}(undef, Nbatch, M + 1, M), 0)
        g0 = fill!(ArrayType{FT}(undef, Nbatch, M + 1), 0)

        # Create storage for constructing the global gmres iterate
        # and recording column-norms
        sol = fill!(ArrayType{FT}(undef, dofperbatch, Nbatch), 0)
        resnorms = fill!(ArrayType{FT}(undef, Nbatch), 0)
        initial_resnorms = fill!(ArrayType{FT}(undef, Nbatch), 0)

        @assert dofperbatch * Nbatch == length(Q)

        # Define the back permutation and reshape
        backward_permute = Tuple(sortperm([forward_permute...]))
        tmp_reshape_tuple_b = [forward_reshape...]
        permute!(tmp_reshape_tuple_b, [forward_permute...])
        backward_reshape = Tuple(tmp_reshape_tuple_b)

        # FIXME: Is there a better way of doing this?
        BKT = typeof(batched_krylov_basis)
        OmT = typeof(Ω)
        HT = typeof(H)
        gT = typeof(g0)
        sT = typeof(sol)
        resT = typeof(resnorms)
        res0T = typeof(initial_resnorms)
        FRS = typeof(forward_reshape)
        FPR = typeof(forward_permute)
        BRS = typeof(backward_reshape)
        BPR = typeof(backward_permute)

        return new{
            typeof(Nbatch),
            eltype(Q),
            AT,
            BKT,
            OmT,
            HT,
            gT,
            sT,
            resT,
            res0T,
            FRS,
            FPR,
            BRS,
            BPR,
        }(
            krylov_basis,
            krylov_basis_prev,
            batched_krylov_basis,
            Ω,
            H,
            g0,
            sol,
            rtol,
            atol,
            M,
            Nbatch,
            dofperbatch,
            resnorms,
            initial_resnorms,
            forward_reshape,
            forward_permute,
            backward_reshape,
            backward_permute,
        )
    end
end

"""
    BatchedGeneralizedMinimalResidual(
        dg::DGModel,
        Q::MPIStateArray;
        atol = sqrt(eps(eltype(Q))),
        rtol = sqrt(eps(eltype(Q))),
        max_subspace_size = nothing,
        independent_states = false,
    )

# Description
Specialized constructor for `BatchedGeneralizedMinimalResidual` struct, using
a `DGModel` to infer state-information and determine appropriate reshaping
and permutations.

# Arguments
- `dg`: (DGModel) A `DGModel` containing all relevant grid and topology
        information.
- `Q` : (MPIStateArray) An `MPIStateArray` containing field information.

# Keyword Arguments
- `atol`: (float) absolute tolerance. `DEFAULT = sqrt(eps(eltype(Q)))`
- `rtol`: (float) relative tolerance. `DEFAULT = sqrt(eps(eltype(Q)))`
- `max_subspace_size` : (Int).    Maximal dimension of each (batched)
                                  Krylov subspace. DEFAULT = nothing
- `independent_states`: (boolean) An optional flag indicating whether
                                  or not degrees of freedom are coupled
                                  internally (within a column).
                                  `DEFAULT = false`
# Return
instance of `BatchedGeneralizedMinimalResidual` struct
"""
function BatchedGeneralizedMinimalResidual(
    dg::DGModel,
    Q::MPIStateArray;
    atol = sqrt(eps(eltype(Q))),
    rtol = sqrt(eps(eltype(Q))),
    max_subspace_size = nothing,
    independent_states = false,
)
    grid = dg.grid
    topology = grid.topology
    dim = dimensionality(grid)

    # Number of Gauss-Lobatto quadrature points in 1D
    Nq = polynomialorder(grid) + 1

    # Assumes same number of quadrature points in all spatial directions
    Np = Tuple([Nq for i in 1:dim])

    # Number of states and elements (in vertical and horizontal directions)
    num_states = size(Q)[2]
    nelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nelem, nvertelem)

    # Definition of a "column" here is a vertical stack of degrees
    # of freedom. For example, consider a mesh consisting of a single
    # linear element:
    #    o----------o
    #    |\ d1   d2 |\
    #    | \        | \
    #    |  \ d3    d4 \
    #    |   o----------o
    #    o--d5---d6-o   |
    #     \  |       \  |
    #      \ |        \ |
    #       \|d7    d8 \|
    #        o----------o
    # There are 4 total 1-D columns, each containing two
    # degrees of freedom. In general, a mesh of stacked elements will
    # have `Nq^2 * nhorzelem` total 1-D columns.
    # A single 1-D column has `Nq * nvertelem * num_states`
    # degrees of freedom.
    #
    # nql = length(Np)
    # indices:      (1...nql, nql + 1 , nql + 2, nql + 3)

    # for 3d case, this is [ni, nj, nk, num_states, nvertelem, nhorzelem]
    # here ni, nj, nk are number of Gauss quadrature points in each element in x-y-z directions
    # Q = reshape(Q, reshaping_tup), leads to the column-wise fashion Q
    reshaping_tup = (Np..., num_states, nvertelem, nhorzelem)

    if independent_states
        m = Nq * nvertelem
        n = (Nq^(dim - 1)) * nhorzelem * num_states
    else
        m = Nq * nvertelem * num_states
        n = (Nq^(dim - 1)) * nhorzelem
    end

    if max_subspace_size === nothing
        max_subspace_size = m
    end

    # permute [ni, nj, nk, num_states, nvertelem, nhorzelem]
    # to      [nvertelem, nk, num_states, ni, nj, nhorzelem]
    permute_size = length(reshaping_tup)
    permute_tuple_f = (dim + 1, dim, dim + 2, (1:(dim - 1))..., permute_size)

    return BatchedGeneralizedMinimalResidual(
        Q,
        m,
        n;
        M = max_subspace_size,
        atol = atol,
        rtol = rtol,
        forward_reshape = reshaping_tup,
        forward_permute = permute_tuple_f,
    )
end

function initialize!(
    linearoperator!,
    Q,
    Qrhs,
    solver::BatchedGeneralizedMinimalResidual,
    args...;
    restart = false,
)
    g0 = solver.g0
    krylov_basis = solver.krylov_basis
    rtol, atol = solver.rtol, solver.atol

    batched_krylov_basis = solver.batched_krylov_basis
    Ndof = solver.dofperbatch
    forward_reshape = solver.forward_reshape
    forward_permute = solver.forward_permute
    resnorms = solver.resnorms
    initial_resnorms = solver.initial_resnorms
    max_iter = solver.max_iter

    # Device and groupsize information
    device = array_device(Q)
    groupsize = 256

    @assert size(Q) == size(krylov_basis)

    # PRECONDITIONER:  PQ0 ->  P*Q0,
    # the first basis is (J Pinv)PQ0 = b, kry1 = b - J Q0
    linearoperator!(krylov_basis, Q, args...)
    krylov_basis .= Qrhs .- krylov_basis

    # Convert into a batched Krylov basis vector
    # REMARK: Ugly hack on the GPU. Can we fix this?
    tmp_array = similar(batched_krylov_basis, size(batched_krylov_basis)[2:3])
    convert_structure!(
        tmp_array,
        krylov_basis,
        forward_reshape,
        forward_permute,
    )
    batched_krylov_basis[1, :, :] .= tmp_array

    # Now we initialize across all columns (solver.batch_size).
    # This function also computes the residual norm in each column
    event = Event(device)
    event = batched_initialize!(device, groupsize)(
        resnorms,
        g0,
        batched_krylov_basis,
        Ndof,
        max_iter;
        ndrange = solver.batch_size,
        dependencies = (event,),
    )
    wait(device, event)

    # When restarting, we do not want to overwrite the initial threshold,
    # otherwise we may not get an accurate indication that we have sufficiently
    # reduced the GMRES residual.
    if !restart
        initial_resnorms .= resnorms
    end
    residual_norm = maximum(resnorms)
    initial_residual_norm = maximum(initial_resnorms)
    converged =
        check_convergence(residual_norm, initial_residual_norm, atol, rtol)

    converged, residual_norm
end

function doiteration!(
    linearoperator!,
    preconditioner,
    Q,
    Qrhs,
    solver::BatchedGeneralizedMinimalResidual,
    threshold,
    args...,
)
    FT = eltype(Q)
    krylov_basis = solver.krylov_basis
    krylov_basis_prev = solver.krylov_basis_prev
    Hs = solver.H
    g0s = solver.g0
    Ωs = solver.Ω
    sols = solver.sol
    batched_krylov_basis = solver.batched_krylov_basis
    Ndof = solver.dofperbatch
    rtol, atol = solver.rtol, solver.atol
    max_iter = solver.max_iter

    forward_reshape = solver.forward_reshape
    forward_permute = solver.forward_permute
    backward_reshape = solver.backward_reshape
    backward_permute = solver.backward_permute
    resnorms = solver.resnorms
    initial_resnorms = solver.initial_resnorms
    initial_residual_norm = maximum(initial_resnorms)

    # Device and groupsize information
    device = array_device(Q)
    groupsize = 256

    # Main batched-GMRES iteration cycle
    converged = false
    residual_norm = typemax(FT)
    j = 1
    for outer j in 1:max_iter
        # FIXME: Remove this back-and-forth reshaping by exploiting the
        # data layout in a similar way that the ColumnwiseLU solver does

        convert_structure!(
            krylov_basis_prev,
            view(batched_krylov_basis, j, :, :),
            backward_reshape,
            backward_permute,
        )

        # PRECONDITIONER: batched_krylov_basis[j+1] =  J P^{-1}batched_krylov_basis[j]
        # set krylov_basis_prev = P^{-1}batched_krylov_basis[j]
        preconditioner_solve!(preconditioner, krylov_basis_prev)

        # Global operator application to get new Krylov basis vector
        linearoperator!(krylov_basis, krylov_basis_prev, args...)

        # Now that we have a global Krylov vector, we reshape and batch
        # the Arnoldi iterations across all columns
        convert_structure!(
            view(batched_krylov_basis, j + 1, :, :),
            krylov_basis,
            forward_reshape,
            forward_permute,
        )

        event = Event(device)
        event = batched_arnoldi_process!(device, groupsize)(
            resnorms,
            g0s,
            Hs,
            Ωs,
            batched_krylov_basis,
            j,
            Ndof;
            ndrange = solver.batch_size,
            dependencies = (event,),
        )
        wait(device, event)

        # Current stopping criteria is based on the maximal column norm
        # TODO: Once we are able to batch the operator application, we
        # should revisit the termination criteria.
        residual_norm = maximum(resnorms)
        converged =
            check_convergence(residual_norm, initial_residual_norm, atol, rtol)
        if converged
            break
        end
    end

    # Reshape the solution vector to construct the new GMRES iterate
    # PRECONDITIONER Q =  Q0 + Pinv PΔQ = Q0 + Pinv (Kry * y)
    # sol = PΔQ = Kry * y
    sols .= 0

    # Solve the triangular system (minimization problem for optimal linear coefficients
    # in the GMRES iterate) and construct the current iterate in each column
    event = Event(device)
    event = construct_batched_gmres_iterate!(device, groupsize)(
        batched_krylov_basis,
        Hs,
        g0s,
        sols,
        j,
        Ndof;
        ndrange = solver.batch_size,
        dependencies = (event,),
    )
    wait(device, event)

    # Use krylov_basis_prev as container for ΔQ
    ΔQ = krylov_basis_prev
    # Unwind reshaping and return solution in standard format
    convert_structure!(ΔQ, sols, backward_reshape, backward_permute)
    # PRECONDITIONER: Q ->  Pinv Q
    preconditioner_solve!(preconditioner, ΔQ)
    Q .+= ΔQ

    # if not converged, then restart
    converged ||
    initialize!(linearoperator!, Q, Qrhs, solver, args...; restart = true)

    (converged, j, residual_norm)
end

@kernel function batched_initialize!(
    resnorms,
    g0,
    batched_krylov_basis,
    Ndof,
    M,
)
    cidx = @index(Global)
    FT = eltype(batched_krylov_basis)

    # Initialize entire RHS storage
    @inbounds for j in 1:(M + 1)
        g0[cidx, j] = FT(0.0)
    end

    # Now we compute the first element of g0[cidx, :],
    # which is determined by the column norm of the initial residual ∥r0∥_2:
    # g0 = ∥r0∥_2 e1
    @inbounds for j in 1:Ndof
        g0[cidx, 1] +=
            batched_krylov_basis[1, j, cidx] * batched_krylov_basis[1, j, cidx]
    end
    @inbounds g0[cidx, 1] = sqrt(g0[cidx, 1])

    # Normalize the batched_krylov_basis by the (local) residual norm
    @inbounds for j in 1:Ndof
        batched_krylov_basis[1, j, cidx] /= g0[cidx, 1]
    end

    # Record initialize residual norm in the column
    @inbounds resnorms[cidx] = g0[cidx, 1]

    nothing
end

@kernel function batched_arnoldi_process!(
    resnorms,
    g0,
    H,
    Ω,
    batched_krylov_basis,
    j,
    Ndof,
)
    cidx = @index(Global)
    FT = eltype(batched_krylov_basis)

    #  Arnoldi process in the local column `cidx`
    @inbounds for i in 1:j
        H[cidx, i, j] = FT(0.0)
        # Modified Gram-Schmidt procedure to generate the Hessenberg matrix
        for k in 1:Ndof
            H[cidx, i, j] +=
                batched_krylov_basis[j + 1, k, cidx] *
                batched_krylov_basis[i, k, cidx]
        end
        # Orthogonalize new Krylov vector against previous one
        for k in 1:Ndof
            batched_krylov_basis[j + 1, k, cidx] -=
                H[cidx, i, j] * batched_krylov_basis[i, k, cidx]
        end
    end

    # And finally, normalize latest Krylov basis vector
    local_norm = FT(0.0)
    @inbounds for i in 1:Ndof
        local_norm +=
            batched_krylov_basis[j + 1, i, cidx] *
            batched_krylov_basis[j + 1, i, cidx]
    end
    @inbounds H[cidx, j + 1, j] = sqrt(local_norm)
    @inbounds for i in 1:Ndof
        batched_krylov_basis[j + 1, i, cidx] /= H[cidx, j + 1, j]
    end

    # Loop over previously computed Krylov basis vectors
    # and apply the Givens rotations
    @inbounds for i in 1:(j - 1)
        cos_tmp = Ω[cidx, 2 * i - 1]
        sin_tmp = Ω[cidx, 2 * i]

        # Apply the Givens rotations
        # | cos -sin | | hi   |
        # | sin  cos | | hi+1 |
        tmp1 = cos_tmp * H[cidx, i, j] + sin_tmp * H[cidx, i + 1, j]
        H[cidx, i + 1, j] =
            -sin_tmp * H[cidx, i, j] + cos_tmp * H[cidx, i + 1, j]
        H[cidx, i, j] = tmp1
    end

    # Eliminate the last element hj+1 and update the rotation matrix
    # | cos -sin | | hj   |  = | hj'|
    # | sin  cos | | hj+1 |  = | 0  |
    # where cos, sin = hj+1/sqrt(hj^2 + hj+1^2), hj/sqrt(hj^2 + hj+1^2),
    # and update for next iteration
    @inbounds begin
        Ω[cidx, 2 * j - 1] = H[cidx, j, j]
        Ω[cidx, 2 * j] = H[cidx, j + 1, j]
        H[cidx, j, j] = sqrt(Ω[cidx, 2 * j - 1]^2 + Ω[cidx, 2 * j]^2)
        H[cidx, j + 1, j] = FT(0.0)
        Ω[cidx, 2 * j - 1] /= H[cidx, j, j]
        Ω[cidx, 2 * j] /= H[cidx, j, j]

        # And now to the rhs g0
        cos_tmp = Ω[cidx, 2 * j - 1]
        sin_tmp = Ω[cidx, 2 * j]
        tmp1 = cos_tmp * g0[cidx, j] + sin_tmp * g0[cidx, j + 1]
        g0[cidx, j + 1] = -sin_tmp * g0[cidx, j] + cos_tmp * g0[cidx, j + 1]
        g0[cidx, j] = tmp1

        # Record estimate for the gmres residual
        resnorms[cidx] = abs(g0[cidx, j + 1])
    end

    nothing
end

@kernel function construct_batched_gmres_iterate!(
    batched_krylov_basis,
    Hs,
    g0s,
    sols,
    j,
    Ndof,
)
    # Solve for the GMRES coefficients (yⱼ) at the `j`-th
    # iteration that minimizes ∥ b - A xⱼ ∥_2, where
    # xⱼ = ∑ᵢ yᵢ Ψᵢ, with Ψᵢ denoting the Krylov basis vectors
    cidx = @index(Global)

    # Do the upper-triangular backsolve
    @inbounds for i in j:-1:1
        g0s[cidx, i] /= Hs[cidx, i, i]
        for k in 1:(i - 1)
            g0s[cidx, k] -= Hs[cidx, k, i] * g0s[cidx, i]
        end
    end

    # Having determined yᵢ, we now construct the GMRES solution
    # in each column: xⱼ = ∑ᵢ yᵢ Ψᵢ
    @inbounds for i in 1:j
        for k in 1:Ndof
            sols[k, cidx] += g0s[cidx, i] * batched_krylov_basis[i, k, cidx]
        end
    end

    nothing
end

"""
    convert_structure!(
        x,
        y,
        reshape_tuple,
        permute_tuple,
    )

Computes a tensor transpose and stores result in x

# Arguments
- `x`: (array) [OVERWRITTEN]. target destination for storing the y data
- `y`: (array). data that we want to copy
- `reshape_tuple`: (tuple) reshapes y to be like that of x, up to a permutation
- `permute_tuple`: (tuple) permutes the reshaped array into the correct structure
"""
@inline function convert_structure!(x, y, reshape_tuple, permute_tuple)
    alias_y = reshape(y, reshape_tuple)
    permute_y = permutedims(alias_y, permute_tuple)
    copyto!(x, reshape(permute_y, size(x)))
    nothing
end
@inline convert_structure!(x, y::MPIStateArray, reshape_tuple, permute_tuple) =
    convert_structure!(x, y.realdata, reshape_tuple, permute_tuple)
@inline convert_structure!(x::MPIStateArray, y, reshape_tuple, permute_tuple) =
    convert_structure!(x.realdata, y, reshape_tuple, permute_tuple)

function check_convergence(residual_norm, initial_residual_norm, atol, rtol)
    relative_residual = residual_norm / initial_residual_norm
    converged = false
    if (residual_norm ≤ atol || relative_residual ≤ rtol)
        converged = true
    end
    return converged
end
