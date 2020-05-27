module BatchedGeneralizedMinimalResidualSolver

export BatchedGeneralizedMinimalResidual

using ..LinearSolvers
const LS = LinearSolvers
using Adapt, CuArrays, KernelAbstractions, LinearAlgebra
using ..MPIStateArrays
using ..Mesh.Grids: dimensionality, polynomialorder
using ..DGmethods: DGModel

# struct
"""
# Description

Launches n independent GMRES solves

# Members

- atol::FT (float) absolute tolerance
- rtol::FT (float) relative tolerance
- m::IT (int) size of vector in each independent instance
- n::IT (int) number of independent GMRES
- k_n::IT (int) Krylov Dimension for each GMRES. It is also the number of GMRES iterations before nuking the subspace
- residual::VT (vector) residual values for each independent linear solve
- b::VT (vector) permutation of the rhs. probably can be removed if memory is an issue
- x::VT (vector) permutation of the initial guess. probably can be removed if memory is an issue
- sol::VT (vector) solution vector, it is used twice. First to represent Aqⁿ (the latest Krylov vector without being normalized), the second to represent the solution to the linear system
- rhs::VT (vector) rhs vector.
- cs::VT (vector) Sequence of Gibbs Rotation matrices in compact form. This is implicitly the Qᵀ of the QR factorization of the upper hessenberg matrix H.
- H::VT (vector) The latest column of the Upper Hessenberg Matrix. The previous columns are discarded since they are unnecessary
- Q::AT (array) Orthonormalized Krylov Subspace
- R::AT (array) The R of the QR factorization of the UpperHessenberg matrix H. A factor of  or so in memory can be saved here
- reshape_tuple_f::TT1 (tuple), reshapes structure of array that plays nice with the linear operator to a format compatible with struct
- permute_tuple_f::TT1 (tuple). forward permute tuple. permutes structure of array that plays nice with the linear operator to a format compatible with struct
- reshape_tuple_b::TT2 (tuple). reshapes structure of array that plays nice with struct to play nice with the linear operator
- permute_tuple_b::TT2 (tuple). backward permute tuple. permutes structure of array that plays nice with struct to play nice with the linear operator

# Intended Use
Solving n linear systems iteratively

# Comments on Improvement
- Too much memory in H and R struct: Could use a sparse representation to cut memory use in half (or more)
- Needs to perform a transpose of original data structure into current data structure: Could perhaps do a transpose free version, but the code gets a bit clunkier and the memory would no longer be coalesced for the heavy operations
"""
struct BatchedGeneralizedMinimalResidual{FT, IT, VT, AT, TT1, TT2} <:
       LS.AbstractIterativeLinearSolver
    atol::FT
    rtol::FT
    m::IT
    n::IT
    k_n::IT
    residual::VT
    b::VT
    x::VT
    sol::VT
    rhs::VT
    cs::VT
    Q::AT
    H::VT
    R::AT
    reshape_tuple_f::TT1
    permute_tuple_f::TT1
    reshape_tuple_b::TT2
    permute_tuple_b::TT2
end

# So that the struct can be passed into kernels
Adapt.adapt_structure(to, x::BatchedGeneralizedMinimalResidual) =
    BatchedGeneralizedMinimalResidual(
        x.atol,
        x.rtol,
        x.m,
        x.n,
        x.k_n,
        adapt(to, x.residual),
        adapt(to, x.b),
        adapt(to, x.x),
        adapt(to, x.sol),
        adapt(to, x.rhs),
        adapt(to, x.cs),
        adapt(to, x.Q),
        adapt(to, x.H),
        adapt(to, x.R),
        x.reshape_tuple_f,
        x.permute_tuple_f,
        x.reshape_tuple_b,
        x.permute_tuple_b,
    )

"""
BatchedGeneralizedMinimalResidual(Qrhs; m = length(Qrhs[:,1]), n = length(Qrhs[1,:]), subspace_size = m, atol = sqrt(eps(eltype(Qrhs))), rtol = sqrt(eps(eltype(Qrhs))), ArrayType = Array, reshape_tuple_f = size(Qrhs), permute_tuple_f = Tuple(1:length(size(Qrhs))), reshape_tuple_b = size(Qrhs), permute_tuple_b = Tuple(1:length(size(Qrhs))))

# Description
Generic constructor for BatchedGeneralizedMinimalResidual

# Arguments
- `Qrhs`: (array) Array structure that linear_operator! acts on

# Keyword Arguments
- `m`: (int) size of vector space for each independent linear solve. This is assumed to be the same for each and every linear solve. DEFAULT = size(Qrhs)[1]
- `n`: (int) number of independent linear solves, DEFAULT = size(Qrhs)[end]
- `atol`: (float) absolute tolerance. DEFAULT = sqrt(eps(eltype(Qrhs)))
- `rtol`: (float) relative tolerance. DEFAULT = sqrt(eps(eltype(Qrhs)))
- `ArrayType`: (type). used for either using CuArrays or Arrays. DEFAULT = Array
- `reshape_tuple_f`: (tuple). used in the wrapper function for flexibility. DEFAULT = size(Qrhs). this means don't do anything
- `permute_tuple_f`: (tuple). used in the wrapper function for flexibility. DEFAULT = Tuple(1:length(size(Qrhs))). this means, don't do anything.

# Return
instance of BatchedGeneralizedMinimalResidual struct
"""
function BatchedGeneralizedMinimalResidual(
    Qrhs;
    m = size(Qrhs)[1],
    n = size(Qrhs)[end],
    subspace_size = m,
    atol = sqrt(eps(eltype(Qrhs))),
    rtol = sqrt(eps(eltype(Qrhs))),
    ArrayType = Array,
    reshape_tuple_f = size(Qrhs),
    permute_tuple_f = Tuple(1:length(size(Qrhs))),
)
    k_n = subspace_size
    # define the back permutations and reshapes
    permute_tuple_b = Tuple(sortperm([permute_tuple_f...]))
    tmp_reshape_tuple_b = [reshape_tuple_f...]
    permute!(tmp_reshape_tuple_b, [permute_tuple_f...])
    reshape_tuple_b = Tuple(tmp_reshape_tuple_b)
    # allocate memory
    residual = ArrayType(zeros(eltype(Qrhs), (k_n, n)))
    b = ArrayType(zeros(eltype(Qrhs), (m, n)))
    x = ArrayType(zeros(eltype(Qrhs), (m, n)))
    sol = ArrayType(zeros(eltype(Qrhs), (m, n)))
    rhs = ArrayType(zeros(eltype(Qrhs), (k_n + 1, n)))
    cs = ArrayType(zeros(eltype(Qrhs), (2 * k_n, n)))
    Q = ArrayType(zeros(eltype(Qrhs), (m, k_n + 1, n)))
    H = ArrayType(zeros(eltype(Qrhs), (k_n + 1, n)))
    R = ArrayType(zeros(eltype(Qrhs), (k_n + 1, k_n, n)))
    return BatchedGeneralizedMinimalResidual(
        atol,
        rtol,
        m,
        n,
        k_n,
        residual,
        b,
        x,
        sol,
        rhs,
        cs,
        Q,
        H,
        R,
        reshape_tuple_f,
        permute_tuple_f,
        reshape_tuple_b,
        permute_tuple_b,
    )
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
Specialized constructor for BatchedGeneralizedMinimalResidual struct, using
a `DGModel` to infer state-information and determine appropraite reshaping
and permutations.

# Arguments
- `dg`: (DGModel) A `DGModel` containing all relevant grid and topology
        information.
- `Q` : (MPIStateArray) An `MPIStateArray` containing field information.

# Keyword Arguments
- `atol`: (float) absolute tolerance. DEFAULT = sqrt(eps(eltype(Q)))
- `rtol`: (float) relative tolerance. DEFAULT = sqrt(eps(eltype(Q)))
- `max_subspace_size` : (Int).    Maximal dimension of each (batched)
                                  Krylov subspace. DEFAULT = nothing
- `independent_states`: (boolean) An optional flag indicating whether
                                  or not degrees of freedom are coupled
                                  internally (within a column).
                                  DEFAULT = false
# Return
instance of BatchedGeneralizedMinimalResidual struct
"""
function BatchedGeneralizedMinimalResidual(
    dg::DGModel,
    Q::MPIStateArray;
    atol = sqrt(eps(eltype(Q))),
    rtol = sqrt(eps(eltype(Q))),
    max_subspace_size = nothing,
    independent_states = false,
)

    # Need to determine array type for storage vectors
    if isa(Q.data, Array)
        ArrayType = Array
    else
        ArrayType = CuArray
    end

    grid = dg.grid
    topology = grid.topology
    dim = dimensionality(grid)

    # Number of Gauss-Lobatto quadrature points in 1D
    Nq = polynomialorder(grid) + 1

    # Assumes same number of quadrature points in all spatial directions
    Np = Tuple([Nq for i in 1:dim])

    # Number of states and elements (in vertical and horizontal directions)
    num_states = size(Q)[2]
    nelem = length(topology.elems)
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

    # Now we need to determine an appropriate permutation
    # of the MPIStateArray to perform column-wise strides.
    # Total size of the permute tuple
    permute_size = length(reshaping_tup)

    # Index associated with number of GL points
    # in the 'vertical' direction
    nql = length(Np)
    # Want: (index associated with the stack size of the column,
    #        index associated with GL pts in vertical direction,
    #        index associated with number of states)
    # FIXME: Better way to do this?
    #             (vert stack, GL pts, num states)
    column_strides = (nql + 2, nql, nql + 1)
    diff = Tuple(setdiff(Set([i for i in 1:permute_size]), Set(column_strides)))
    permute_tuple_f = (column_strides..., diff...)

    return BatchedGeneralizedMinimalResidual(
        Q;
        m = m,
        n = n,
        subspace_size = max_subspace_size,
        atol = atol,
        rtol = rtol,
        ArrayType = ArrayType,
        reshape_tuple_f = reshaping_tup,
        permute_tuple_f = permute_tuple_f,
    )
end

# initialize function (1)
function LS.initialize!(
    linearoperator!,
    Q,
    Qrhs,
    solver::BatchedGeneralizedMinimalResidual,
    args...,
)
    # body of initialize function in abstract iterative solver
    return false, zero(eltype(Q))
end

# iteration function (2)
function LS.doiteration!(
    linearoperator!,
    Q,
    Qrhs,
    gmres::BatchedGeneralizedMinimalResidual,
    threshold,
    args...,
)
    # Get device and groupsize information
    if isa(gmres.b, Array)
        device = CPU()
        groupsize = Threads.nthreads()
    else
        device = CUDA()
        groupsize = 256
    end

    # initialize gmres.x
    convert_structure!(gmres.x, Q, gmres.reshape_tuple_f, gmres.permute_tuple_f)
    # apply linear operator to construct residual
    r_vector = copy(Q)
    linearoperator!(r_vector, Q, args...)
    @. r_vector = Qrhs - r_vector
    # The following ar and rr are technically not correct in general cases
    ar = norm(r_vector)
    rr = norm(r_vector) / norm(Qrhs)
    # check if the initial guess is fantastic
    if (ar < gmres.atol) || (rr < gmres.rtol)
        return true, 0, ar
    end
    # initialize gmres.b
    convert_structure!(
        gmres.b,
        r_vector,
        gmres.reshape_tuple_f,
        gmres.permute_tuple_f,
    )
    # apply linear operator to construct second krylov vector
    linearoperator!(Q, r_vector, args...)
    # initialize gmres.sol
    convert_structure!(
        gmres.sol,
        Q,
        gmres.reshape_tuple_f,
        gmres.permute_tuple_f,
    )
    # initialize the rest of gmres
    event = Event(device)
    event = initialize_gmres_kernel!(device, groupsize)(
        gmres;
        ndrange = gmres.n,
        dependencies = (event,),
    )
    wait(device, event)

    ar, rr = compute_residuals(gmres, 1)
    # check if converged
    if (ar < gmres.atol) || (rr < gmres.rtol)
        event = Event(device)
        event = construct_solution_kernel!(device, groupsize)(
            1,
            gmres;
            ndrange = size(gmres.x),
            dependencies = (event,),
        )
        wait(device, event)

        convert_structure!(
            Q,
            gmres.x,
            gmres.reshape_tuple_b,
            gmres.permute_tuple_b,
        )
        return true, 1, ar
    end
    # body of iteration
    @inbounds for i in 2:(gmres.k_n)
        convert_structure!(
            r_vector,
            view(gmres.Q, :, i, :),
            gmres.reshape_tuple_b,
            gmres.permute_tuple_b,
        )
        linearoperator!(Q, r_vector, args...)
        convert_structure!(
            gmres.sol,
            Q,
            gmres.reshape_tuple_f,
            gmres.permute_tuple_f,
        )

        event = Event(device)
        event = gmres_update_kernel!(device, groupsize)(
            i,
            gmres;
            ndrange = gmres.n,
            dependencies = (event,),
        )
        wait(device, event)

        ar, rr = compute_residuals(gmres, i)
        # check if converged
        if (ar < gmres.atol) || (rr < gmres.rtol)
            event = Event(device)
            event = construct_solution_kernel!(device, groupsize)(
                i,
                gmres;
                ndrange = size(gmres.x),
                dependencies = (event,),
            )
            wait(device, event)

            convert_structure!(
                Q,
                gmres.x,
                gmres.reshape_tuple_b,
                gmres.permute_tuple_b,
            )
            return true, i, ar
        end
    end

    event = Event(device)
    event = construct_solution_kernel!(device, groupsize)(
        gmres.k_n,
        gmres;
        ndrange = size(gmres.x),
        dependencies = (event,),
    )
    wait(device, event)

    convert_structure!(Q, gmres.x, gmres.reshape_tuple_b, gmres.permute_tuple_b)
    ar, rr = compute_residuals(gmres, gmres.k_n)
    converged = (ar < gmres.atol) || (rr < gmres.rtol)
    return converged, gmres.k_n, ar
end

# The function(s) that probably needs the most help
"""
function convert_structure!(x, y, reshape_tuple, permute_tuple)

# Description
Computes a tensor transpose and stores result in x
- This needs to be improved!

# Arguments
- `x`: (array) [OVERWRITTEN]. target destination for storing the y data
- `y`: (array). data that we want to copy
- `reshape_tuple`: (tuple) reshapes y to be like that of x, up to a permutation
- `permute_tuple`: (tuple) permutes the reshaped array into the correct structure

# Keyword Arguments
- `convert`: (bool). decides whether or not permute and convert. The default is true

# Return
nothing
"""
@inline function convert_structure!(
    x,
    y,
    reshape_tuple,
    permute_tuple;
    convert = true,
)
    if convert
        alias_y = reshape(y, reshape_tuple)
        permute_y = permutedims(alias_y, permute_tuple)
        x[:] .= permute_y[:]
    end
    return nothing
end

# MPIStateArray dispatch
@inline convert_structure!(x, y::MPIStateArray, reshape_tuple, permute_tuple) =
    convert_structure!(x, y.data, reshape_tuple, permute_tuple)
@inline convert_structure!(x::MPIStateArray, y, reshape_tuple, permute_tuple) =
    convert_structure!(x.data, y, reshape_tuple, permute_tuple)

# Kernels
"""
initialize_gmres_kernel!(gmres)

# Description
Initializes the gmres struct by calling
- initialize_arnoldi
- initialize_QR!
- update_arnoldi!
- update_QR!
- solve_optimization!
It is assumed that the first two krylov vectors are already constructed

# Arguments
- `gmres`: (struct) gmres struct

# Return
(implicitly) kernel abstractions function closure
"""
@kernel function initialize_gmres_kernel!(gmres)
    I = @index(Global)
    initialize_arnoldi!(gmres, I)
    update_arnoldi!(1, gmres, I)
    initialize_QR!(gmres, I)
    update_QR!(1, gmres, I)
    solve_optimization!(1, gmres, I)
end

"""
gmres_update_kernel!(i, gmres, I)

# Description
kernel that calls
- update_arnoldi!
- update_QR!
- solve_optimization!
Which is the heart of the gmres algorithm

# Arguments
- `i`: (int) interation index
- `gmres`: (struct) gmres struct
- `I`: (int) thread index

# Return
kernel object from KernelAbstractions
"""
@kernel function gmres_update_kernel!(i, gmres)
    I = @index(Global)
    update_arnoldi!(i, gmres, I)
    update_QR!(i, gmres, I)
    solve_optimization!(i, gmres, I)
end

"""
construct_solution_kernel!(i, gmres)

# Description
given step i of the gmres iteration, constructs the "best" solution of the linear system for the given Krylov subspace

# Arguments
- `i`: (int) gmres iteration
- `gmres`: (struct) gmres struct

# Return
kernel object from KernelAbstractions
"""
@kernel function construct_solution_kernel!(i, gmres)
    M, I = @index(Global, NTuple)
    tmp = zero(eltype(gmres.b))
    @inbounds for j in 1:i
        tmp += gmres.Q[M, j, I] * gmres.sol[j, I]
    end
    gmres.x[M, I] += tmp # since previously gmres.x held the initial value
end

# Helper Functions

"""
initialize_arnoldi!(g, I)

# Description
- First step of Arnoldi Iteration is to define first Krylov vector. Additionally sets things equal to zero

# Arguments
- `g`: (struct) [OVERWRITTEN] the gmres struct
- `I`: (int) thread index

# Return
nothing
"""
@inline function initialize_arnoldi!(gmres, I)
    # set (almost) everything to zero to be sure
    # the assumption is that gmres.k_n is small enough
    # to where these loops don't matter that much
    ft_zero = zero(eltype(gmres.H)) # float type zero

    @inbounds for i in 1:(gmres.k_n + 1)
        gmres.rhs[i, I] = ft_zero
        gmres.H[i, I] = ft_zero
        @inbounds for j in 1:(gmres.k_n)
            gmres.R[i, j, I] = ft_zero
        end
    end
    # gmres.x was initialized as the initial x
    # gmres.sol was initialized right before this function call
    # gmres.b was initialized right before this function call
    # compute norm
    @inbounds for i in 1:(gmres.m)
        gmres.rhs[1, I] += gmres.b[i, I] * gmres.b[i, I]
    end
    gmres.rhs[1, I] = sqrt(gmres.rhs[1, I])
    # now start computations
    @inbounds for i in 1:(gmres.m)
        gmres.sol[i, I] /= gmres.rhs[1, I]
        gmres.Q[i, 1, I] = gmres.b[i, I] / gmres.rhs[1, I] # First Krylov vector
    end
    return nothing
end

"""
initialize_QR!(gmres::BatchedGeneralizedMinimalResidual, I)

# Description
initializes the QR decomposition of the UpperHesenberg Matrix

# Arguments
- `gmres`: (struct) [OVERWRITTEN] the gmres struct
- `I`: (int) thread index

# Return
nothing
"""
@inline function initialize_QR!(gmres, I)
    gmres.cs[1, I] = gmres.H[1, I]
    gmres.cs[2, I] = gmres.H[2, I]
    gmres.R[1, 1, I] = sqrt(gmres.cs[1, I]^2 + gmres.cs[2, I]^2)
    gmres.cs[1, I] /= gmres.R[1, 1, I]
    gmres.cs[2, I] /= -gmres.R[1, 1, I]
    return nothing
end

# The meat of gmres with updates that leverage information from the previous iteration
"""
update_arnoldi!(n, gmres, I)
# Description
Perform an Arnoldi iteration update

# Arguments
- `n`: current iteration number
- `gmres`: gmres struct that gets overwritten
- `I`: (int) thread index
# Return
- nothing
# linear_operator! Arguments
- `linear_operator!(x,y)`
# Description
- Performs Linear operation on vector and overwrites it
# Arguments
- `y`: (array)
# Return
nothing

"""
@inline function update_arnoldi!(n, gmres, I)
    # make new Krylov Vector orthogonal to previous ones
    @inbounds for j in 1:n
        gmres.H[j, I] = 0
        # dot products
        @inbounds for i in 1:(gmres.m)
            gmres.H[j, I] += gmres.Q[i, j, I] * gmres.sol[i, I]
        end
        # orthogonalize latest Krylov Vector
        @inbounds for i in 1:(gmres.m)
            gmres.sol[i, I] -= gmres.H[j, I] * gmres.Q[i, j, I]
        end
    end
    norm_q = 0.0
    @inbounds for i in 1:(gmres.m)
        norm_q += gmres.sol[i, I] * gmres.sol[i, I]
    end
    gmres.H[n + 1, I] = sqrt(norm_q)
    @inbounds for i in 1:(gmres.m)
        gmres.Q[i, n + 1, I] = gmres.sol[i, I] / gmres.H[n + 1, I]
    end
    return nothing
end

"""
update_QR!(n, gmres, I)

# Description
Given a QR decomposition of the first n-1 columns of an upper hessenberg matrix, this computes the QR decomposition associated with the first n columns
# Arguments
- `gmres`: (struct) [OVERWRITTEN] the struct has factors that are updated
- `n`: (integer) column that needs to be updated
- `I`: (int) thread index
# Return
- nothing

# Comment
What is actually produced by the algorithm isn't the Q in the QR decomposition but rather Q^*. This is convenient since this is what is actually needed to solve the linear system
"""
@inline function update_QR!(n, gmres, I)
    # Apply previous Q to new column
    @inbounds for i in 1:n
        gmres.R[i, n, I] = gmres.H[i, I]
    end
    # apply rotation
    @inbounds for i in 1:(n - 1)
        tmp1 =
            gmres.cs[1 + 2 * (i - 1), I] * gmres.R[i, n, I] -
            gmres.cs[2 * i, I] * gmres.R[i + 1, n, I]
        gmres.R[i + 1, n, I] =
            gmres.cs[2 * i, I] * gmres.R[i, n, I] +
            gmres.cs[1 + 2 * (i - 1), I] * gmres.R[i + 1, n, I]
        gmres.R[i, n, I] = tmp1
    end
    # Now update, cs and R
    gmres.cs[1 + 2 * (n - 1), I] = gmres.R[n, n, I]
    gmres.cs[2 * n, I] = gmres.H[n + 1, I]
    gmres.R[n, n, I] =
        sqrt(gmres.cs[1 + 2 * (n - 1), I]^2 + gmres.cs[2 * n, I]^2)
    gmres.cs[1 + 2 * (n - 1), I] /= gmres.R[n, n, I]
    gmres.cs[2 * n, I] /= -gmres.R[n, n, I]
    return nothing
end

"""
solve_optimization!(iteration, gmres, I)

# Description
Solves the optimization problem in GMRES
# Arguments
- `iteration`: (int) current iteration number
- `gmres`: (struct) [OVERWRITTEN]
- `I`: (int) thread index
# Return
nothing
"""
@inline function solve_optimization!(n, gmres, I)
    # just need to update rhs from previous iteration
    # apply latest givens rotation
    tmp1 =
        gmres.cs[1 + 2 * (n - 1), I] * gmres.rhs[n, I] -
        gmres.cs[2 * n, I] * gmres.rhs[n + 1, I]
    gmres.rhs[n + 1, I] =
        gmres.cs[2 * n, I] * gmres.rhs[n, I] +
        gmres.cs[1 + 2 * (n - 1), I] * gmres.rhs[n + 1, I]
    gmres.rhs[n, I] = tmp1
    # gmres.rhs[iteration+1] is the residual. Technically convergence should be checked here.
    gmres.residual[n, I] = abs.(gmres.rhs[n + 1, I])
    # copy for performing the backsolve and saving gmres.rhs
    @inbounds for i in 1:n
        gmres.sol[i, I] = gmres.rhs[i, I]
    end
    # do the backsolve
    @inbounds for i in n:-1:1
        gmres.sol[i, I] /= gmres.R[i, i, I]
        @inbounds for j in 1:(i - 1)
            gmres.sol[j, I] -= gmres.R[j, i, I] * gmres.sol[i, I]
        end
    end
    return nothing
end

"""
compute_residuals(gmres)

# Description
Compute atol and rtol of current iteration

# Arguments
- `gmres`: (struct)
- `i`: (current iteration)

# Return
- `atol`: (float) absolute tolerance
- `rtol`: (float) relative tolerance

# Comment
sometimes gmres.R[1, 1,:] term has some very large components which makes rtol quite small
"""
function compute_residuals(gmres, i)
    atol = maximum(gmres.residual[i, :])
    rtol = maximum(gmres.residual[i, :] ./ norm(gmres.R[1, 1, :]))
    return atol, rtol
end

end # end of module
