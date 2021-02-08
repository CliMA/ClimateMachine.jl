using ForwardDiff: Dual, Partials
using StructArrays: StructArray, fieldarrays
using Base.Broadcast: Broadcasted

using ..MPIStateArrays: MPIStateArray, transform_array
import ..MPIStateArrays: array_device, transform_broadcasted
using ..DGMethods: DGModel

"""
    enable_duals(value, n = 1, tag = nothing)
    
Creates an object that behaves in the same way as `value` while also accepting
dual numbers. Specifically, if `value` contains numbers of type `FT` where it
needs to accept dual numbers, `enable_duals` replaces them with numbers of type
`ForwardDiff.Dual{tag, FT, n}`. Note that `enable_duals` should not be called
on its own output, as the result may contain dual numbers of dual numbers.

# Arguments
- `value`: object which needs to accept dual numbers
- `n`: number of partials in the dual numbers
- `tag`: tag on the dual numbers
"""
enable_duals(value) = enable_duals(value, 1)
enable_duals(value, n) = enable_duals(value, n, nothing)

# By default, assume that `value` already accepts dual numbers.
enable_duals(value, n, tag) = value

# Wrap an array in a StructArray to make it accept dual numbers, allocating
# similar arrays for storing the partials.
enable_duals(value::AbstractArray{FT}, n, tag) where {FT} =
    StructArray{Dual{tag, FT, n}}((
        value,
        StructArray{Partials{n, FT}}((
            StructArray{NTuple{n, FT}}(ntuple(i -> similar(value), n)),
        )),
    ))

# Modify the `data` field in an MPIStateArray to make it accept dual numbers,
# and update the `realdata` field appropriately. This overrides the default
# behavior for an AbstractArray.
function enable_duals(
    value::MPIStateArray{FT, V, DATN, DAI1, DAV, Buf, DATW},
    n,
    tag,
) where {FT, V, DATN, DAI1, DAV, Buf, DATW}
    data = enable_duals(value.data, n, tag)
    realdata =
        view(data, ntuple(i -> Colon(), ndims(data) - 1)..., value.realelems)
    return MPIStateArray{
        eltype(data),
        V,
        typeof(data),
        DAI1,
        typeof(realdata),
        Buf,
        DATW,
    }(
        value.mpicomm,
        data,
        realdata,
        value.realelems,
        value.ghostelems,
        value.vmaprecv,
        value.vmapsend,
        value.sendreq,
        value.recvreq,
        value.send_buffer,
        value.recv_buffer,
        value.nabrtorank,
        value.nabrtovmaprecv,
        value.nabrtovmapsend,
        value.weights,
    )
end

# Modify every MPIStateArray in a DGModel to make it accept dual numbers, and
# set the partials in each one to 0. The second step is necessary because some
# of those partials are used to compute the partials in the prognostic state.
# TODO: Determine whether we need to set the partials to 0 elsewhere.
function enable_duals(value::DGModel, n, tag)
    dual = DGModel(
        value.balance_law,
        value.grid,
        value.numerical_flux_first_order,
        value.numerical_flux_second_order,
        value.numerical_flux_gradient,
        enable_duals(value.state_auxiliary, n, tag),
        enable_duals(value.state_gradient_flux, n, tag),
        enable_duals.(value.states_higher_order, n, tag),
        value.direction,
        value.diffusion_direction,
        value.modeldata,
    )
    FT = eltype(value.state_auxiliary)
    for i in 1:n
        fill!(partial(dual.state_auxiliary, i), zero(FT))
        fill!(partial(dual.state_gradient_flux, i), zero(FT))
        fill!.(partial.(dual.states_higher_order, i), zero(FT))
    end
    return dual
end

"""
    update_duals!(dual, value)

Generates the result of `enable_duals(value, n, tag)`, where `n` and `tag` are
such that `typeof(dual) == typeof(enable_duals(value, n, tag))`. Once `dual` is
available, this function should be used instead of `enable_duals` because it
avoids unnecessary memory allocations. If `dual` is mutable, it gets modified;
otherwise, a new object gets constructed that reuses some components of `dual`.
"""
# If `dual` cannot be generated from `value` with a call to `enable_duals`,
# throw an error.
update_duals!(dual, value) =
    throw(ArgumentError(string(
        "update_duals! was called with invalid arguments dual and value; ",
        "ensure that typeof(dual) == typeof(enable_duals(value, n, tag)) for ",
        "some n and tag"
    )))

# If `value` has the same type as `dual`, it can just be returned, since that
# is what `enable_duals` would do.
update_duals!(dual::T, value::T) where {T} = value

# Since a StructArray is immutable, construct a new one, reusing the partials
# from the old one. Note that the type signature is so convoluted because this
# method ensures that `dual` has the same type as `enable_duals(value, n, tag)`
# for some `n` and `tag`. Other methods of `update_duals!` end up calling this
# one, which lets them get away with not performing such checks.
update_duals!(dual::StructArrayT, value::ArrayT) where {
    ArrayT,
    NTup <: NamedTuple{(:value, :partials), <:Tuple{ArrayT, StructArray}},
    StructArrayT <: StructArray{<:Dual, <:Any, NTup},
} = StructArrayT(NTup((value, dual.partials)))

# Since an MPIStateArray is mutable, update it directly.
function update_duals!(dual::MPIStateArray, value)
    dual.mpicomm = value.mpicomm
    dual.data = update_duals!(dual.data, value.data)
    dual.realdata = view(
        dual.data,
        ntuple(i -> Colon(), ndims(dual.data) - 1)...,
        value.realelems,
    )
    dual.realelems = value.realelems
    dual.ghostelems = value.ghostelems
    dual.vmaprecv = value.vmaprecv
    dual.vmapsend = value.vmapsend
    dual.sendreq = value.sendreq
    dual.recvreq = value.recvreq
    dual.send_buffer = value.send_buffer
    dual.recv_buffer = value.recv_buffer
    dual.nabrtorank = value.nabrtorank
    dual.nabrtovmaprecv = value.nabrtovmaprecv
    dual.nabrtovmapsend = value.nabrtovmapsend
    dual.weights = value.weights
    return dual
end

# Since a DGModel is immutable, construct a new one.
function update_duals!(dual::T, value) where {T <: DGModel}
    return T(
        value.balance_law,
        value.grid,
        value.numerical_flux_first_order,
        value.numerical_flux_second_order,
        value.numerical_flux_gradient,
        update_duals!(dual.state_auxiliary, value.state_auxiliary),
        update_duals!(dual.state_gradient_flux, value.state_gradient_flux),
        update_duals!.(dual.states_higher_order, value.states_higher_order),
        value.direction,
        value.diffusion_direction,
        value.modeldata,
    )
end

"""
    value(dual)
    
Returns the values of the dual numbers in `dual`, extending the functionality
of `ForwardDiff.value()` to arrays.
"""
value(dual::Dual) = dual.value
value(dual::StructArray{<:Dual}) = dual.value
value(dual::MPIStateArray{<:Dual}) = value(dual.realdata)

"""
    partial(dual, i = 1)
    
Returns the `i`-th partials of the dual numbers in `dual`, extending the
functionality of `ForwardDiff.partials()` to arrays.
"""
partial(dual) = partial(dual, 1)
partial(dual::Dual, i) = getproperty(dual.partials.values, i)
partial(dual::StructArray{<:Dual}, i) = getproperty(dual.partials.values, i)
partial(dual::MPIStateArray{<:Dual}, i) = partial(dual.realdata, i)

# These functions are required by an MPIStateArray that contains a StructArray.
array_device(s::StructArray) = array_device(fieldarrays(s)[1])
transform_broadcasted(bc::Broadcasted, ::StructArray) = transform_array(bc)

# The following functions are for bypassing the default broadcasting mechanism
# for a StructArray of dual numbers.
# TODO: Check whether all of these actually improve efficiency. The second
#       copyto! should be identical to the default version for typeof(src) <:
#       Union{StructArray, MPIStateArray}, but the others should be faster.
# TODO: Consider unrolling all the loops `i in 1:N` with generated functions.
function Base.fill!(
    A::StructArray{DT},
    x::FT,
) where {Tag, FT, N, DT <: Dual{Tag, FT, N}}
    fill!(value(A), x)
    for i in 1:N
        fill!(partial(A, i), zero(FT))
    end
    return A
end
function Base.fill!(
    A::StructArray{DT},
    x::DT,
) where {Tag, FT, N, DT <: Dual{Tag, FT, N}}
    fill!(value(A), x)
    for i in 1:N
        fill!(partial(A, i), partial(x, i))
    end
    return A
end
function Base.copyto!(
    dest::StructArray{DT},
    src::AbstractArray{FT},
) where {Tag, FT, N, DT <: Dual{Tag, FT, N}}
    copyto!(value(dest), src)
    for i in 1:N
        fill!(partial(dest, i), zero(FT))
    end
    return dest
end
function Base.copyto!(
    dest::StructArray{DT},
    src::AbstractArray{DT},
) where {Tag, FT, N, DT <: Dual{Tag, FT, N}}
    copyto!(value(dest), value(src))
    for i in 1:N
        copyto!(partial(dest, i), partial(src, i))
    end
    return dest
end
function Base.copyto!(
    dest::StructArray{DT},
    bc::Broadcasted{Nothing},
) where {Tag, FT, N, DT <: Dual{Tag, FT, N}}
    nlb = num_linear_bcs(DT, bc)
    if nlb == 0
        copyto!(value(dest), bc)
        for i in 1:N
            fill!(partial(dest, i), zero(FT))
        end
    elseif nlb == 1
        copyto!(value(dest), value_bc(DT, bc))
        for i in 1:N
            copyto!(partial(dest, i), partial_bc(DT, bc, i))
        end
    else
        # Default copyto! from broadcast.jl
        if bc.f === identity && bc.args isa Tuple{AbstractArray}
            A = bc.args[1]
            if axes(dest) == axes(A)
                return copyto!(dest, A)
            end
        end
        bc′ = preprocess(dest, bc)
        @simd for I in eachindex(bc′)
            @inbounds dest[I] = bc′[I]
        end
    end
    return dest
end

"""
    num_linear_bcs(T::Type, bc::Any)

Indicates whether bc is a linear broadcast with respect to objects of type T.

The possible return values are
- 0:   bc is not a broadcast that involves objects of type T
- 1:   bc is a broadcast that is linear with respect to objects of type T
- NaN: bc is a broadcast that is nonlinear with respect to objects of type T
"""
# An identity operation is a linear broadcast if its argument is one.
num_linear_bcs(
    ::Type{T},
    bc::Broadcasted{S, A, typeof(identity)},
) where {T, S, A} = num_linear_bcs(T, bc.args[1])
# A sum/difference must contain 1 or more linear broadcasts in order to be one.
function num_linear_bcs(
    ::Type{T},
    bc::Broadcasted{S, A, F},
) where {T, S, A, F <: Union{typeof(+), typeof(-)}}
    s = sum(num_linear_bcs.(T, bc.args))
    return ifelse(s > 1, 1, s)
end
# A product must contain exactly 1 linear broadcast in order to be one. If it
# contains more than 1 linear broadcast, it is nonlinear.
function num_linear_bcs(
    ::Type{T},
    bc::Broadcasted{S, A, typeof(*)},
) where {T, S, A}
    s = sum(num_linear_bcs.(T, bc.args))
    return ifelse(s > 1, NaN, s)
end
# A quotient must contain a linear broadcast in only the numerator in order to
# be one. If it contains a linear or nonlinear broadcast in the denominator, it
# is nonlinear.
function num_linear_bcs(
    ::Type{T},
    bc::Broadcasted{S, A, typeof(/)},
) where {T, S, A}
    if num_linear_bcs(T, bc.args[2]) == 0
        return num_linear_bcs(T, bc.args[1])
    else
        return NaN
    end
end
# Any other operation that contains a linear or nonlinear broadcast is assumed
# to be nonlinear.
num_linear_bcs(::Type{T}, bc::Broadcasted{S, A, F}) where {T, S, A, F} =
    ifelse(sum(num_linear_bcs.(T, bc.args)) > 0, NaN, 0)
# An object of type T or an array of such objects is always a linear broadcast.
num_linear_bcs(::Type{T}, ::Union{T, AbstractArray{<:T}}) where {T} = 1
# Anything else is not a broadcast that involves objects of type T.
num_linear_bcs(::Type, ::Any) = 0

# Extract the values from a broadcasted operation that is known to be linear
# with respect to dual type T.
value_bc(::Type{T}, bc::Broadcasted) where {T} =
    Broadcasted(bc.f, value_bc.(T, bc.args), bc.axes)
value_bc(::Type{T}, Q::Union{T, AbstractArray{<:T}}) where {T} = value(Q)
value_bc(::Type, Q::Any) = Q

# Extract the partials from a broadcasted operation that is known to be linear
# with respect to dual type T. For sums and differences, this requires
# removing all arguments that do not have partials and extracting the partials
# from the remaining arguments. (This is equivalent to replacing the arguments
# that do not have partials with zeros.) Since the overall broadcast is known
# to be linear, at least one of its arguments must also be a linear broadcast,
# so at least one set of partials will be avialable for extraction.
partial_bc(::Type{T}, bc::Broadcasted{S, A, typeof(+)}, i) where {T, S, A} =
    Broadcasted(bc.f, rm_no_partials(T, i, bc.args...), bc.axes)
partial_bc(
    ::Type{T},
    bc::Broadcasted{S, A, typeof(-), <:Tuple{Any}},
    i
) where {T, S, A} =
    Broadcasted(bc.f, (partial_bc(T, bc.args[1], i),), bc.axes)
function partial_bc(
    ::Type{T},
    bc::Broadcasted{S, A, typeof(-), <:Tuple{Any, Any}},
    i
) where {T, S, A}
    c1 = has_partials(T, bc.args[1])
    c2 = has_partials(T, bc.args[2])
    if c1 && c2
        return Broadcasted(bc.f, partial_bc.(T, bc.args, i), bc.axes)
    elseif c1
        return partial_bc(T, bc.args[1], i)
    else # c2
        return Broadcasted(bc.f, (partial_bc(T, bc.args[2], i),), bc.axes)
    end
end
partial_bc(::Type{T}, bc::Any, i) where {T} = _partial_bc(T, bc, i)

# Utility functions for extracting partials.
function rm_no_partials(::Type{T}, i, bc, args...) where {T}
    if has_partials(T, bc)
        return (partial_bc(T, bc, i), rm_no_partials(T, i, args...)...)
    else
        return rm_no_partials(T, i, args...)
    end
end
rm_no_partials(::Type, i) = ()
has_partials(::Type{T}, bc::Broadcasted) where {T} =
    any(has_partials.(T, bc.args))
has_partials(::Type{T}, ::Union{T, AbstractArray{<:T}}) where {T} = true
has_partials(::Type, ::Any) = false
_partial_bc(::Type{T}, bc::Broadcasted, i) where {T} =
    Broadcasted(bc.f, _partial_bc.(T, bc.args, i), bc.axes)
_partial_bc(::Type{T}, Q::Union{T, AbstractArray{<:T}}, i) where {T} =
    partial(Q, i)
_partial_bc(::Type, Q::Any, i) = Q