using Test
using StaticArrays

abstract type AbstractModel end
abstract type OneLayerModel <: AbstractModel end

struct EmptyModel <: OneLayerModel end
state(m::EmptyModel, T) = @vars()

struct ScalarModel <: OneLayerModel end
state(m::ScalarModel, T) = @vars(x::T)

struct VectorModel{N} <: OneLayerModel end
state(m::VectorModel{N}, T) where {N} = @vars(x::SVector{N, T})

struct MatrixModel{N, M} <: OneLayerModel end
state(m::MatrixModel{N, M}, T) where {N, M} =
    @vars(x::SHermitianCompact{N, T, M})

abstract type TwoLayerModel <: AbstractModel end

struct CompositModel{EM, SM, VM, MM} <: TwoLayerModel
    empty_model::EM
    scalar_model::SM
    vector_model::VM
    matrix_model::MM
end
function CompositModel(
    Nv,
    N,
    M;
    empty_model = EmptyModel(),
    scalar_model = ScalarModel(),
    vector_model = VectorModel{Nv}(),
    matrix_model = MatrixModel{N, M}(),
)
    args = (empty_model, scalar_model, vector_model, matrix_model)
    return CompositModel{typeof.(args)...}(args...)
end


function state(m::CompositModel, T)
    @vars begin
        empty_model::state(m.empty_model, T)
        scalar_model::state(m.scalar_model, T)
        vector_model::state(m.vector_model, T)
        matrix_model::state(m.matrix_model, T)
    end
end

Base.@kwdef struct NTupleModel{S} <: OneLayerModel
    scalar_model::S = ScalarModel()
end

function state(m::NTupleModel, T)
    @vars begin
        scalar_model::state(m.scalar_model, T)
    end
end

state(m::NTuple{N, NTupleModel}, FT) where {N} =
    Tuple{ntuple(i -> state(m[i], FT), N)...}

struct NTupleContainingModel{N, NTM, VM, SM} <: TwoLayerModel
    ntuple_model::NTM
    vector_model::VM
    scalar_model::SM
end
function NTupleContainingModel(
    N,
    Nv;
    ntuple_model = ntuple(i -> NTupleModel(), N),
    vector_model = VectorModel{Nv}(),
    scalar_model = ScalarModel(),
)
    args = (ntuple_model, vector_model, scalar_model)
    return NTupleContainingModel{N, typeof.(args)...}(args...)
end

function state(m::NTupleContainingModel, T)
    @vars begin
        ntuple_model::state(m.ntuple_model, T)
        vector_model::state(m.vector_model, T)
        scalar_model::state(m.scalar_model, T)
    end
end
