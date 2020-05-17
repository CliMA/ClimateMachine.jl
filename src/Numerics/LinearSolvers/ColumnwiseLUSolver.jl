module ColumnwiseLUSolver

export ManyColumnLU, SingleColumnLU

using ..Mesh.Grids
using ..Mesh.Topologies
using ..DGmethods
using ..DGmethods:
    BalanceLaw, DGModel, number_state_conservative, number_state_gradient_flux
using ..LinearSolvers
const LS = LinearSolvers
using ..MPIStateArrays
using LinearAlgebra
using KernelAbstractions

abstract type AbstractColumnLUSolver <: AbstractLinearSolver end

"""
    ManyColumnLU()

This solver is used for systems that are block diagonal where each block is
associated with a column of the mesh.  The systems are solved using a
non-pivoted LU factorization.
"""
struct ManyColumnLU <: AbstractColumnLUSolver end

"""
    SingleColumnLU()

This solver is used for systems that are block diagonal where each block is
associated with a column of the mesh.  Moreover, each block is assumed to be
the same.  The systems are solved using a non-pivoted LU factorization.
"""
struct SingleColumnLU <: AbstractColumnLUSolver end

struct ColumnwiseLU{F, AT}
    f::F
    A::AT
end

function LS.prefactorize(op, solver::AbstractColumnLUSolver, Q, args...)
    dg = op.f!

    # TODO: can we get away with just passing the grid?
    A = banded_matrix(
        op,
        dg,
        similar(Q),
        similar(Q),
        args...;
        single_column = typeof(solver) <: SingleColumnLU,
    )

    band_lu!(A, dg)

    ColumnwiseLU(dg, A)
end

function LS.linearsolve!(
    clu::ColumnwiseLU{F},
    ::AbstractColumnLUSolver,
    Q,
    Qrhs,
    args...,
) where {F <: DGModel}
    device = typeof(Q.data) <: Array ? CPU() : CUDA()

    dg = clu.f
    A = clu.A
    Q .= Qrhs

    band_forward!(Q, A, dg)
    band_back!(Q, A, dg)
end

"""
    band_lu!(A, dg::DGModel)

"""
function band_lu!(A, dg::DGModel)
    bl = dg.balance_law
    grid = dg.grid
    topology = grid.topology
    @assert isstacked(topology)
    @assert typeof(dg.direction) <: VerticalDirection

    FT = eltype(A)
    device = typeof(A) <: Array ? CPU() : CUDA()

    nstate = number_state_conservative(bl, FT)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqj = dimensionality(grid) == 2 ? 1 : Nq

    eband = number_state_gradient_flux(bl, FT) == 0 ? 1 : 2

    nrealelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    groupsize = (Nq, Nqj)
    ndrange = (nhorzelem * Nq, Nqj)

    if ndims(A) == 2
        # single column case
        #
        # TODO Would it be faster to copy the matrix to the host and factorize it
        # there?
        groupsize = (1, 1)
        ndrange = groupsize
        A = reshape(A, 1, 1, size(A)..., 1)
    end

    event = Event(device)
    event = band_lu_kernel!(device, groupsize)(
        A,
        Val(Nq),
        Val(groupsize[1]),
        Val(groupsize[2]),
        Val(nstate),
        Val(nvertelem),
        Val(ndrange[end]),
        Val(eband);
        ndrange = ndrange,
        dependencies = (event,),
    )
    wait(device, event)
end

function band_forward!(Q, A, dg::DGModel)
    bl = dg.balance_law
    grid = dg.grid
    topology = grid.topology
    @assert isstacked(topology)
    @assert typeof(dg.direction) <: VerticalDirection

    FT = eltype(A)
    device = typeof(Q.data) <: Array ? CPU() : CUDA()

    nstate = number_state_conservative(bl, FT)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqj = dimensionality(grid) == 2 ? 1 : Nq

    eband = number_state_gradient_flux(bl, FT) == 0 ? 1 : 2

    nrealelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    event = Event(device)
    event = band_forward_kernel!(device, (Nq, Nqj))(
        Q.data,
        A,
        Val(Nq),
        Val(Nqj),
        Val(nstate),
        Val(nvertelem),
        Val(nhorzelem),
        Val(eband);
        ndrange = (nhorzelem * Nq, Nqj),
        dependencies = (event,),
    )
    wait(device, event)
end

function band_back!(Q, A, dg::DGModel)
    bl = dg.balance_law
    grid = dg.grid
    topology = grid.topology
    @assert isstacked(topology)
    @assert typeof(dg.direction) <: VerticalDirection

    FT = eltype(A)
    device = typeof(Q.data) <: Array ? CPU() : CUDA()

    nstate = number_state_conservative(bl, FT)
    N = polynomialorder(grid)
    Nq = N + 1
    Nqj = dimensionality(grid) == 2 ? 1 : Nq

    eband = number_state_gradient_flux(bl, FT) == 0 ? 1 : 2

    nrealelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    event = Event(device)
    event = band_back_kernel!(device, (Nq, Nqj))(
        Q.data,
        A,
        Val(Nq),
        Val(Nqj),
        Val(nstate),
        Val(nvertelem),
        Val(nhorzelem),
        Val(eband);
        ndrange = (nhorzelem * Nq, Nqj),
        dependencies = (event,),
    )
    wait(device, event)
end


"""
    banded_matrix(dg::DGModel, [Q::MPIStateArray, dQ::MPIStateArray,
                  single_column=false])

Forms the banded matrices for each the column operator defined by the `DGModel`
dg.  If `single_column=false` then a banded matrix is stored for each column and
if `single_column=true` only the banded matrix associated with the first column
of the first element is stored. The bandwidth of the DG column banded matrix is
`p = q = (polynomialorder + 1) * nstate * nvertelem - 1` with `p` and `q` being
the upper and lower bandwidths.

The banded matrices are stored in the LAPACK band storage format
<https://www.netlib.org/lapack/lug/node124.html>.

The banded matrices are returned as an arrays where the array type matches that
of `Q`. If `single_column=false` then the returned array has 5 dimensions, which
are:
- first horizontal column index
- second horizontal column index
- band index (-q:p)
- vertical DOF index with state `s`, vertical DOF index `k`, and vertical
  element `ev` mapping to `s + nstate * (k - 1) + nstate * nvertelem * (ev - 1)`
- horizontal element index

If the `single_column=true` then the returned array has 2 dimensions which are
the band index and the vertical DOF index.
"""
function banded_matrix(
    dg::DGModel,
    Q::MPIStateArray = MPIStateArray(dg),
    dQ::MPIStateArray = MPIStateArray(dg);
    single_column = false,
)
    banded_matrix(
        (dQ, Q) -> dg(dQ, Q, nothing, 0; increment = false),
        dg,
        Q,
        dQ;
        single_column = single_column,
    )
end

"""
    banded_matrix(f!::Function, dg::DGModel,
                  Q::MPIStateArray = MPIStateArray(dg),
                  dQ::MPIStateArray = MPIStateArray(dg), args...;
                  single_column = false, args...)

Forms the banded matrices for each the column operator defined by the linear
operator `f!` which is assumed to have the same banded structure as the
`DGModel` dg.  If `single_column=false` then a banded matrix is stored for each
column and if `single_column=true` only the banded matrix associated with the
first column of the first element is stored. The bandwidth of the DG column
banded matrix is `p = q = (polynomialorder + 1) * nstate * nvertelem - 1` with
`p` and `q` being the upper and lower bandwidths.

The banded matrices are stored in the LAPACK band storage format
<https://www.netlib.org/lapack/lug/node124.html>.

The banded matrices are returned as an arrays where the array type matches that
of `Q`. If `single_column=false` then the returned array has 5 dimensions, which
are:
- first horizontal column index
- second horizontal column index
- band index (-q:p)
- vertical DOF index with state `s`, vertical DOF index `k`, and vertical
  element `ev` mapping to `s + nstate * (k - 1) + nstate * nvertelem * (ev - 1)`
- horizontal element index

If the `single_column=true` then the returned array has 2 dimensions which are
the band index and the vertical DOF index.

Here `args` are passed to `f!`.
"""
function banded_matrix(
    f!,
    dg::DGModel,
    Q::MPIStateArray = MPIStateArray(dg),
    dQ::MPIStateArray = MPIStateArray(dg),
    args...;
    single_column = false,
)
    bl = dg.balance_law
    grid = dg.grid
    topology = grid.topology
    @assert isstacked(topology)
    @assert typeof(dg.direction) <: VerticalDirection

    FT = eltype(Q.data)
    device = typeof(Q.data) <: Array ? CPU() : CUDA()

    nstate = number_state_conservative(bl, FT)
    N = polynomialorder(grid)
    Nq = N + 1

    # p is lower bandwidth
    # q is upper bandwidth
    eband = number_state_gradient_flux(bl, FT) == 0 ? 1 : 2
    p = q = nstate * Nq * eband - 1

    nrealelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    dim = dimensionality(grid)

    Nqj = dim == 2 ? 1 : Nq

    # first horizontal DOF index
    # second horizontal DOF index
    # band index -q:p
    # vertical DOF index
    # horizontal element index
    A = if single_column
        similar(Q.data, p + q + 1, Nq * nstate * nvertelem)
    else
        similar(Q.data, Nq, Nqj, p + q + 1, Nq * nstate * nvertelem, nhorzelem)
    end
    fill!(A, zero(FT))

    # loop through all DOFs in a column and compute the matrix column
    for ev in 1:nvertelem
        for s in 1:nstate
            for k in 1:Nq
                # Set a single 1 per column and rest 0
                event = Event(device)
                event = kernel_set_banded_data!(device, (Nq, Nqj, Nq))(
                    bl,
                    Val(dim),
                    Val(N),
                    Val(nvertelem),
                    Q.data,
                    k,
                    s,
                    ev,
                    1:nhorzelem,
                    1:nvertelem;
                    ndrange = (nvertelem * Nq, nhorzelem * Nqj, Nq),
                    dependencies = (event,),
                )
                wait(device, event)

                # Get the matrix column
                f!(dQ, Q, args...)

                # Store the banded matrix
                event = Event(device)
                event = kernel_set_banded_matrix!(device, (Nq, Nqj, Nq))(
                    bl,
                    Val(dim),
                    Val(N),
                    Val(nvertelem),
                    Val(p),
                    Val(q),
                    Val(eband + 1),
                    A,
                    dQ.data,
                    k,
                    s,
                    ev,
                    1:nhorzelem,
                    (-eband):eband;
                    ndrange = ((2eband + 1) * Nq, nhorzelem * Nqj, Nq),
                    dependencies = (event,),
                )
                wait(device, event)
            end
        end
    end
    A
end


"""
    banded_matrix_vector_product!(dg::DGModel, A, dQ::MPIStateArray,
                                  Q::MPIStateArray)

Compute a matrix vector product `dQ = A * Q` where `A` is assumed to be a matrix
created using the `banded_matrix` function.

This function is primarily for testing purposes.
"""
function banded_matrix_vector_product!(
    dg::DGModel,
    A,
    dQ::MPIStateArray,
    Q::MPIStateArray,
)
    bl = dg.balance_law
    grid = dg.grid
    topology = grid.topology
    @assert isstacked(topology)
    @assert typeof(dg.direction) <: VerticalDirection

    FT = eltype(Q.data)
    device = typeof(Q.data) <: Array ? CPU() : CUDA()

    eband = number_state_gradient_flux(bl, FT) == 0 ? 1 : 2
    nstate = number_state_conservative(bl, FT)
    N = polynomialorder(grid)
    Nq = N + 1
    p = q = nstate * Nq * eband - 1

    nrealelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    dim = dimensionality(grid)

    Nqj = dim == 2 ? 1 : Nq

    event = Event(device)
    event = kernel_banded_matrix_vector_product!(device, (Nq, Nqj, Nq))(
        bl,
        Val(dim),
        Val(N),
        Val(nvertelem),
        Val(p),
        Val(q),
        dQ.data,
        A,
        Q.data,
        1:nhorzelem,
        1:nvertelem;
        ndrange = (nvertelem * Nq, nhorzelem * Nqj, Nq),
        dependencies = (event,),
    )
    wait(device, event)
end

using StaticArrays
using KernelAbstractions.Extras: @unroll

@doc """
    band_lu_kernel!(A, Val(Nq), Val(Nqi), Val(Nqj), Val(nstate), Val(nvertelem),
                 Val(nhorzelem), Val(eband))

This performs Band Gaussian Elimination (Algorithm 4.3.1 of Golub and Van
Loan).  The array `A` contains a band matrix for each vertical column.  For
example, `A[i, j, :, :, h]`, is the band matrix associated with the `(i, j)`th
degree of freedom in the horizontal element `h`.

Each `n` by `n` band matrix is assumed to have upper bandwidth `q` and lower
bandwidth `p` where `n = nstate * Nq * nvertelem` and `p = q = nstate * Nq *
eband - 1`.

Each band matrix is stored in the [LAPACK band storage](https://www.netlib.org/lapack/lug/node124.html).
For example the band matrix

    B = [b₁₁ b₁₂ 0   0   0
         b₂₁ b₂₂ b₂₃ 0   0
         b₃₁ b₃₂ b₃₃ b₃₄ 0
         0   b₄₂ b₄₃ b₄₄ b₄₅
         0   0   b₅₃ b₅₄ b₅₅]

is stored as

    B = [0   b₁₂ b₂₃ b₃₄ b₄₅
         b₁₁ b₂₂ b₃₃ b₄₄ b₅₅
         b₂₁ b₃₂ b₄₃ b₅₄ 0
         b₃₁ b₄₂ b₅₃ 0   0]

### Reference

    @book{GolubVanLoan,
      title = {Matrix Computations},
      author = {Gene H. Golub and Charles F. Van Loan},
      edition = {4th},
      isbn = {9781421407944},
      publisher = {Johns Hopkins University Press},
      address = {Baltimore, MD, USA},
      url = {http://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm},
      year = 2013
    }

""" band_lu_kernel!
@kernel function band_lu_kernel!(
    A,
    ::Val{Nq},
    ::Val{Nqi},
    ::Val{Nqj},
    ::Val{nstate},
    ::Val{nvertelem},
    ::Val{nhorzelem},
    ::Val{eband},
) where {Nq, Nqi, Nqj, nstate, nvertelem, nhorzelem, eband}
    @uniform begin
        FT = eltype(A)
        n = nstate * Nq * nvertelem
        p = q = nstate * Nq * eband - 1
    end

    h = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        for v in 1:nvertelem
            for k in 1:Nq
                for s in 1:nstate
                    kk = s + (k - 1) * nstate + (v - 1) * nstate * Nq

                    Aq = A[i, j, q + 1, kk, h]
                    for ii in 1:p
                        A[i, j, q + ii + 1, kk, h] /= Aq
                    end

                    for jj in 1:q
                        if jj + kk ≤ n
                            Ajj = A[i, j, q - jj + 1, jj + kk, h]
                            for ii in 1:p
                                A[i, j, q + ii - jj + 1, jj + kk, h] -=
                                    A[i, j, q + ii + 1, kk, h] * Ajj
                            end
                        end
                    end
                end
            end
        end
    end
end

@doc """
    band_forward_kernel!(b, LU, Val(Nq), Val(Nqj), Val(nstate), Val(nvertelem),
                      Val(nhorzelem), Val(eband))

This performs Band Forward Substitution (Algorithm 4.3.2 of Golub and Van
Loan), i.e., the right-hand side `b` is replaced with the solution of `L*x=b`.

The array `b` is of the size `(Nq * Nqj * Nq, nstate, nvertelem * nhorzelem)`.

The LU-factorization array `LU` contains a single band matrix or one
for each vertical column, see [`band_lu!`](@ref).

Each `n` by `n` band matrix is assumed to have upper bandwidth `q` and lower
bandwidth `p` where `n = nstate * Nq * nvertelem` and `p = q = nstate * Nq *
eband - 1`.

### Reference

    @book{GolubVanLoan,
      title = {Matrix Computations},
      author = {Gene H. Golub and Charles F. Van Loan},
      edition = {4th},
      isbn = {9781421407944},
      publisher = {Johns Hopkins University Press},
      address = {Baltimore, MD, USA},
      url = {http://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm},
      year = 2013
    }

""" band_forward_kernel!
@kernel function band_forward_kernel!(
    b,
    LU::AbstractArray{T, N},
    ::Val{Nq},
    ::Val{Nqj},
    ::Val{nstate},
    ::Val{nvertelem},
    ::Val{nhorzelem},
    ::Val{eband},
) where {T, N, Nq, Nqj, nstate, nvertelem, nhorzelem, eband}
    @uniform begin
        FT = eltype(b)
        n = nstate * Nq * nvertelem
        p = q = eband * nstate * Nq - 1

        l_b = MArray{Tuple{p + 1}, FT}(undef)
    end

    h = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        @unroll for v in 1:eband
            @unroll for k in 1:Nq
                @unroll for s in 1:nstate
                    ijk = i + Nqj * (j - 1) + Nq * Nqj * (k - 1)
                    ee = v + nvertelem * (h - 1)
                    ii = s + (k - 1) * nstate + (v - 1) * nstate * Nq
                    l_b[ii] = nvertelem ≥ v ? b[ijk, s, ee] : zero(FT)
                end
            end
        end

        for v in 1:nvertelem
            @unroll for k in 1:Nq
                @unroll for s in 1:nstate
                    jj = s + (k - 1) * nstate + (v - 1) * nstate * Nq

                    @unroll for ii in 2:(p + 1)
                        Lii = N == 2 ? LU[ii + q, jj] : LU[i, j, ii + q, jj, h]
                        l_b[ii] -= Lii * l_b[1]
                    end

                    ijk = i + Nqj * (j - 1) + Nq * Nqj * (k - 1)
                    ee = v + nvertelem * (h - 1)

                    b[ijk, s, ee] = l_b[1]

                    @unroll for ii in 1:p
                        l_b[ii] = l_b[ii + 1]
                    end

                    if jj + p < n
                        (idx, si) = fldmod1(jj + p + 1, nstate)
                        (vi, ki) = fldmod1(idx, Nq)

                        ijk = i + Nqj * (j - 1) + Nq * Nqj * (ki - 1)
                        ee = vi + nvertelem * (h - 1)

                        l_b[p + 1] = b[ijk, si, ee]
                    end
                end
            end
        end
    end
end

@doc """
    band_back_kernel!(b, LU, Val(Nq), Val(Nqj), Val(nstate), Val(nvertelem),
                   Val(nhorzelem), Val(eband))

This performs Band Back Substitution (Algorithm 4.3.3 of Golub and Van
Loan), i.e., the right-hand side `b` is replaced with the solution of `U*x=b`.

The array `b` is of the size `(Nq * Nqj * Nq, nstate, nvertelem * nhorzelem)`.

The LU-factorization array `LU` contains a single band matrix or one
for each vertical column, see [`band_lu!`](@ref).

Each `n` by `n` band matrix is assumed to have upper bandwidth `q` and lower
bandwidth `p` where `n = nstate * Nq * nvertelem` and `p = q = nstate * Nq *
eband - 1`.

### Reference

    @book{GolubVanLoan,
      title = {Matrix Computations},
      author = {Gene H. Golub and Charles F. Van Loan},
      edition = {4th},
      isbn = {9781421407944},
      publisher = {Johns Hopkins University Press},
      address = {Baltimore, MD, USA},
      url = {http://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm},
      year = 2013
    }

""" band_back_kernel!
@kernel function band_back_kernel!(
    b,
    LU::AbstractArray{T, N},
    ::Val{Nq},
    ::Val{Nqj},
    ::Val{nstate},
    ::Val{nvertelem},
    ::Val{nhorzelem},
    ::Val{eband},
) where {T, N, Nq, Nqj, nstate, nvertelem, nhorzelem, eband}
    @uniform begin
        FT = eltype(b)
        n = nstate * Nq * nvertelem
        q = nstate * Nq * eband - 1

        l_b = MArray{Tuple{q + 1}, FT}(undef)
    end

    h = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        @unroll for v in nvertelem:-1:(nvertelem - eband + 1)
            @unroll for k in Nq:-1:1
                @unroll for s in nstate:-1:1
                    vi = eband - nvertelem + v
                    ii = s + (k - 1) * nstate + (vi - 1) * nstate * Nq

                    ijk = i + Nqj * (j - 1) + Nq * Nqj * (k - 1)
                    ee = v + nvertelem * (h - 1)

                    l_b[ii] = b[ijk, s, ee]
                end
            end
        end

        for v in nvertelem:-1:1
            @unroll for k in Nq:-1:1
                @unroll for s in nstate:-1:1
                    jj = s + (k - 1) * nstate + (v - 1) * nstate * Nq

                    l_b[q + 1] /=
                        N == 2 ? LU[q + 1, jj] : LU[i, j, q + 1, jj, h]

                    @unroll for ii in 1:q
                        Uii = N == 2 ? LU[ii, jj] : LU[i, j, ii, jj, h]
                        l_b[ii] -= Uii * l_b[q + 1]
                    end

                    ijk = i + Nqj * (j - 1) + Nq * Nqj * (k - 1)
                    ee = v + nvertelem * (h - 1)

                    b[ijk, s, ee] = l_b[q + 1]

                    @unroll for ii in q:-1:1
                        l_b[ii + 1] = l_b[ii]
                    end

                    if jj - q > 1
                        (idx, si) = fldmod1(jj - q - 1, nstate)
                        (vi, ki) = fldmod1(idx, Nq)

                        ijk = i + Nqj * (j - 1) + Nq * Nqj * (ki - 1)
                        ee = vi + nvertelem * (h - 1)

                        l_b[1] = b[ijk, si, ee]
                    end
                end
            end
        end
    end
end

@kernel function kernel_set_banded_data!(
    bl::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    Q,
    kin,
    sin,
    evin,
    helems,
    velems,
) where {dim, N, nvertelem}
    @uniform begin
        FT = eltype(Q)

        Nq = N + 1
        Nqj = dim == 2 ? 1 : Nq
        nstate = number_state_conservative(bl, FT)
    end

    ev, eh = @index(Group, NTuple)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        e = ev + (eh - 1) * nvertelem
        ijk = i + Nqj * (j - 1) + Nq * Nqj * (k - 1)
        @unroll for s in 1:nstate
            if k == kin && s == sin && evin == ev
                Q[ijk, s, e] = 1
            else
                Q[ijk, s, e] = 0
            end
        end
    end
end

@kernel function kernel_set_banded_matrix!(
    bl::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    ::Val{p},
    ::Val{q},
    ::Val{eshift},
    A::AbstractArray{FT, AN},
    dQ,
    kin,
    sin,
    evin,
    helems,
    vpelems,
) where {dim, N, nvertelem, p, q, eshift, FT, AN}
    @uniform begin
        Nq = N + 1
        Nqj = dim == 2 ? 1 : Nq
        nstate = number_state_conservative(bl, FT)

        # sin, kin, evin are the state, vertical fod, and vert element we are
        # handling

        # column index of matrix
        jj = sin + (kin - 1) * nstate + (evin - 1) * nstate * Nq
    end

    ep, eh = @index(Group, NTuple)
    ep = ep - eshift
    i, j, k = @index(Local, NTuple)

    # one thread is launch for dof that might contribute to column jj's band
    @inbounds begin
        # ep is the shift we need to add to evin to get the element we need to
        # consider
        ev = ep + evin
        if 1 ≤ ev ≤ nvertelem
            e = ev + (eh - 1) * nvertelem
            ijk = i + Nqj * (j - 1) + Nq * Nqj * (k - 1)
            @unroll for s in 1:nstate
                # row index of matrix
                ii = s + (k - 1) * nstate + (ev - 1) * nstate * Nq
                # row band index
                bb = ii - jj
                # make sure we're in the bandwidth
                if -q ≤ bb ≤ p
                    if AN === 5
                        A[i, j, bb + q + 1, jj, eh] = dQ[ijk, s, e]
                    elseif AN === 2
                        if (i, j, eh) == (1, 1, 1)
                            A[bb + q + 1, jj] = dQ[ijk, s, e]
                        end
                    end
                end
            end
        end
    end
end

@kernel function kernel_banded_matrix_vector_product!(
    bl::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    ::Val{p},
    ::Val{q},
    dQ,
    A::AbstractArray{FT, AN},
    Q,
    helems,
    velems,
) where {dim, N, nvertelem, p, q, FT, AN}

    @uniform begin
        Nq = N + 1
        Nqj = dim == 2 ? 1 : Nq
        nstate = number_state_conservative(bl, FT)

        elo = div(q, Nq * nstate - 1)
        eup = div(p, Nq * nstate - 1)
    end

    ev, eh = @index(Group, NTuple)
    i, j, k = @index(Local, NTuple)

    # matrix row loops
    @inbounds begin
        e = ev + nvertelem * (eh - 1)
        @unroll for s in 1:nstate
            Ax = -zero(FT)
            ii = s + (k - 1) * nstate + (ev - 1) * nstate * Nq

            # banded matrix column loops
            @unroll for evv in max(1, ev - elo):min(nvertelem, ev + eup)
                ee = evv + nvertelem * (eh - 1)
                @unroll for kk in 1:Nq
                    ijk = i + Nqj * (j - 1) + Nq * Nqj * (kk - 1)
                    @unroll for ss in 1:nstate
                        jj = ss + (kk - 1) * nstate + (evv - 1) * nstate * Nq
                        bb = ii - jj
                        if -q ≤ bb ≤ p
                            if AN === 5
                                Ax +=
                                    A[i, j, bb + q + 1, jj, eh] * Q[ijk, ss, ee]
                            elseif AN === 2
                                Ax += A[bb + q + 1, jj] * Q[ijk, ss, ee]
                            end
                        end
                    end
                end
            end
            ijk = i + Nqj * (j - 1) + Nq * Nqj * (k - 1)
            dQ[ijk, s, e] = Ax
        end
    end
end
end
