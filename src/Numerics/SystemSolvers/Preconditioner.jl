export PreconditioningAlgorithm,
    Preconditioner,
    ColumnwiseLUPreconditioningAlgorithm,
    ColumnwiseLUPReconNdiTIoneR,
    NoPreconditioningAlgorithm,
    NoPReconNdiTIoneR,
    preconditioner_update!

"""
    Interface for all preconditioners.
"""
abstract type PreconditioningAlgorithm end

abstract type Preconditioner end

function Preconditioner(::PreconditioningAlgorithm, Q, f!) end

function preconditioner_update!(
    preconditioner::Preconditioner,
    args...,
) end

function (::Preconditioner)(Q) end

"""
mutable struct NoPreconditioner
end

Do nothing
"""
struct NoPreconditioningAlgorithm <: PreconditioningAlgorithm end

mutable struct NoPReconNdiTIoneR <: Preconditioner end

Preconditioner(::NoPreconditioningAlgorithm, Q, f!) = NoPReconNdiTIoneR()

"""
mutable struct ColumnwiseLUPreconditioner{AT}
    A::DGColumnBandedMatrix
    Q::AT
    PQ::AT
    counter::Int
    update_period::Int
end

...
# Arguments
- `A`: the lu factor of the precondition (approximated Jacobian), in the DGColumnBandedMatrix format
- `Q`: MPIArray container, used to update A
- `PQ`: MPIArray container, used to update A
- `counter`: count the number of Newton, when counter > update_period or counter < 0, update precondition
- `update_period`: preconditioner is rebuilt once every `update_period` iterations
...
"""
struct ColumnwiseLUPreconditioningAlgorithm <: PreconditioningAlgorithm
    update_period
    function ColumnwiseLUPreconditioningAlgorithm(;
        update_period::Union{Int, Nothing} = nothing,
    )
        @checkargs("be nonnegative", arg -> arg >= 0, update_period)
        return new(update_period)
    end
end

mutable struct ColumnwiseLUPReconNdiTIoneR{AT} <: Preconditioner
    A::DGColumnBandedMatrix
    Q::AT
    PQ::AT
    counter::Int
    update_period::Int
end

function Preconditioner(algorithm::ColumnwiseLUPreconditioningAlgorithm, Q0::MPIStateArray, dg::DGModel)
    update_period = isnothing(algorithm.update_period) ?
        100 : algorithm.update_period

    Q = similar(Q0)
    PQ = similar(Q0)
    A = empty_banded_matrix(dg, Q; single_column = false)

    # Set counter to -1, which indicates that the preconditioner is empty.
    return ColumnwiseLUPReconNdiTIoneR(A, Q, PQ, -1, update_period)
end

"""
    preconditioner_update!(
        preconditioner::ColumnwiseLUPreconditioner,
        op,
        dg::DGModel,
        args...
    )

Update the DGColumnBandedMatrix by the finite difference approximation

# Arguments
- `op`: operator used to compute the finte difference information
- `dg`: the DG model, use only the grid information
"""
function preconditioner_update!(
    preconditioner::ColumnwiseLUPReconNdiTIoneR,
    op,
    dg::DGModel,
    args...;
)

    # preconditioner.counter < 0, means newly constructed empty preconditioner
    if preconditioner.counter >= 0 &&
       (preconditioner.counter < preconditioner.update_period)
        preconditioner.counter += 1
        return
    end

    A = preconditioner.A
    Q = preconditioner.Q
    PQ = preconditioner.PQ

    update_banded_matrix!(A, op, dg, Q, PQ, args...)
    band_lu!(A)

    preconditioner.counter = 0
end

"""
Inplace applying the preconditioner

Q = P⁻¹ * Q
"""
function (preconditioner!::ColumnwiseLUPReconNdiTIoneR)(Q)
    A = preconditioner!.A
    band_forward!(Q, A)
    band_back!(Q, A)
end
