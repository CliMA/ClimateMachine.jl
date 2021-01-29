using ForwardDiff: Dual, Partials
using StructArrays: StructArray, fieldarrays
using Base.Broadcast: Broadcasted

using ..DGMethods: DGModel
using ..MPIStateArrays: MPIStateArray, transform_array
import ..MPIStateArrays: array_device, transform_broadcasted

# If dual numbers will be used in multiple parts of the ClimateMachine, the
# following functions should be moved to the appropriate files (e.g.,
# DGModel.jl and MPIStateArrays.jl). However, if they will only be used by
# JacobianVectorProductAD, it is much simpler to keep everything in one file.

"""
    enable_duals(input::Any, n::Int = 1, tag = nothing)
    
Create an object that behaves in the same way as the input while also accepting
dual numbers.

...
# Arguments
- `input`: object which needs to accept dual numbers
- `n`: number of partials in the dual numbers
- `tag`: tag on the dual numbers
...
"""

# By default, assume that the input already accepts dual numbers.
enable_duals(input::Any, n::Int = 1, tag = nothing) = input

# If the input contains cached objects like operators or arrays, those objects
# should be modified to accept dual numbers.
function enable_duals(dg::DGModel, n::Int = 1, tag = nothing)
    newdg = DGModel(
        dg.balance_law,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        enable_duals(dg.state_auxiliary, n, tag),
        enable_duals(dg.state_gradient_flux, n, tag),
        enable_duals.(dg.states_higher_order, n, tag),
        dg.direction,
        dg.diffusion_direction,
        dg.modeldata,
    )
    FT = eltype(dg.state_auxiliary)
    for i in 1:n
        fill!(partial(newdg.state_auxiliary, i), zero(FT))
        fill!(partial(newdg.state_gradient_flux, i), zero(FT))
        for array in newdg.states_higher_order
            fill!(partial(array, i), zero(FT))
        end
    end
    return newdg
end

# If necessary, modify the data and realdata in an MPIStateArray to make them
# accept dual numbers.
enable_duals(Q::MPIStateArray{<:Dual}, n::Int = 1, tag = nothing) = Q
function enable_duals(
    Q::MPIStateArray{FT, V, DATN, DAI1, DAV, Buf, DATW},
    n::Int = 1,
    tag = nothing
) where {FT, V, DATN, DAI1, DAV, Buf, DATW}
    data = enable_duals(Q.data)
    realdata =
        view(data, ntuple(i -> Colon(), ndims(data) - 1)..., Q.realelems)
    return MPIStateArray{
        eltype(data),
        V,
        typeof(data),
        DAI1,
        typeof(realdata),
        Buf,
        DATW,
    }(
        Q.mpicomm,
        data,
        realdata,
        Q.realelems,
        Q.ghostelems,
        Q.vmaprecv,
        Q.vmapsend,
        Q.sendreq,
        Q.recvreq,
        Q.send_buffer,
        Q.recv_buffer,
        Q.nabrtorank,
        Q.nabrtovmaprecv,
        Q.nabrtovmapsend,
        Q.weights,
    )
end

# These functions are required by an MPIStateArray that contains a StructArray.
array_device(s::StructArray) = array_device(fieldarrays(s)[1])
transform_broadcasted(bc::Broadcasted, ::StructArray) = transform_array(bc)

# If necessary, wrap an array in a StructArray to make it accept dual numbers.
# Allocate similar arrays for storing the partials.
enable_duals(Q::AbstractArray{<:Dual}, n::Int = 1, tag = nothing) = Q
enable_duals(Q::AbstractArray{FT}, n::Int = 1, tag = nothing) where {FT} =
    StructArray{Dual{tag, FT, n}}((
        Q,
        StructArray{Partials{n, FT}}((
            StructArray{NTuple{n, FT}}(ntuple(i -> similar(Q), n)),
        )),
    ))

"""
    value(Q::AbstractArray{<:Dual})
    
Returns the values of the dual numbers in `Q`. Similar to the
`ForwardDiff.value()` function.
"""
value(Q::Dual) = Q.value
value(Q::StructArray{<:Dual}) = Q.value
value(Q::MPIStateArray{<:Dual}) = value(Q.realdata)

"""
    partial(Q::AbstractArray{<:Dual}, i::Int = 1)
    
Returns the `i`-th partials of the dual numbers in `Q`. Similar to the
`ForwardDiff.partials()` function.
"""
partial(Q::Dual, i::Int = 1) = getproperty(Q.partials.values, i)
partial(Q::StructArray{<:Dual}, i::Int = 1) = getproperty(Q.partials.values, i)
partial(Q::MPIStateArray{<:Dual}, i::Int = 1) = partial(Q.realdata, i)

"""
    setvalue!(Q::AbstractArray{<:Dual}, value)
    
Swaps out the array of values in `Q` for a new array. If possible, does the
operation in place.
"""
setvalue!(Q::StructArray{DT, N, NTup}, value::AT) where {
    DT <: Dual, N, AT,
    NTup <: NamedTuple{(:value, :partials), <:Tuple{AT, StructArray}}
} = StructArray{DT, N, NTup}(NTup((value, Q.partials)))
function setvalue!(
    Q::MPIStateArray{DT},
    value::MPIStateArray{FT}
) where {Tag, FT, N, DT <: Dual{Tag, FT, N}}
    Q.data = setvalue!(Q.data, value.data)
    Q.realdata =
        view(Q.data, ntuple(i -> Colon(), ndims(Q.data) - 1)..., Q.realelems)
    return Q
end

# The following functions are for bypassing the default broadcasting mechanism
# for a StructArray of dual numbers.
# TODO: Check whether all of these actually improve efficiency. The second
#       copyto! should be identical to the default version for typeof(src) <:
#       Union{StructArray, MPIStateArray}, but the others should be faster.
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
# to be linear, at least one of its arguments must also be a linear broadcast.
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