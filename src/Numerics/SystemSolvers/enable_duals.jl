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
enable_duals(op::EulerOperator, n::Int = 1, tag = nothing) =
    EulerOperator(enable_duals(op.f!, n, tag), op.ϵ)
enable_duals(dg::DGModel, n::Int = 1, tag = nothing) =
    DGModel(
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

# If necessary, modify the data and realdata in an MPIStateArray to make them
# accept dual numbers.
enable_duals(Q::MPIStateArray{<:Dual}, n::Int = 1, tag = nothing) = Q
function enable_duals(
    Q::MPIStateArray{FT},
    n::Int = 1,
    tag = nothing
) where {FT}
    data = enable_duals(Q.data)
    realdata =
        view(data, ntuple(i -> Colon(), ndims(data) - 1)..., Q.realelems)
    return MPIStateArray(
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
    NTup <: NamedTuple{(:value, :partials), Tuple{AT, <:StructArray}}
} = StructArray{DT, N, NTup}((value, Q.partials))
function setvalue!(
    Q::MPIStateArray{<:Dual{FT}},
    value::MPIStateArray{FT}
) where {FT}
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
function Base.copyto!(
    dest::StructArray{DT},
    src::AbstractArray{FT},
) where {Tag, FT, DT <: Dual{Tag, FT}}
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
    nlb = num_linear_broadcasts(DT, bc)
    if nlb == 0
        copyto!(value(dest), bc)
        for i in 1:N
            fill!(partial(dest, i), zero(FT))
        end
    elseif nlb == 1
        copyto!(value(dest), value_broadcast(DT, bc))
        for i in 1:N
            copyto!(partial(dest, i), partial_broadcast(DT, bc, i))
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
    num_linear_broadcasts(T::Type, bc::Any)

Indicates whether bc is a linear broadcast with respect to objects of type T.

The possible return values are
- 0:   bc is not a broadcast that involves objects of type T
- 1:   bc is a broadcast that is linear with respect to objects of type T
- NaN: bc is a broadcast that is nonlinear with respect to objects of type T
"""
# An identity operation is a linear broadcast if its argument is one.
function num_linear_broadcasts(
    ::Type{T},
    bc::Broadcasted{S, A, typeof(identity)},
) where {T, S, A}
    return num_linear_broadcasts(T, bc.args[1])
end
# A sum/difference must contain 1 or more linear broadcasts in order to be one.
function num_linear_broadcasts(
    ::Type{T},
    bc::Broadcasted{S, A, F},
) where {T, S, A, F <: Union{typeof(+), typeof(-)}}
    s = sum(num_linear_broadcasts.(T, bc.args))
    return ifelse(s > 1, 1, s)
end
# A product must contain exactly 1 linear broadcast in order to be one. If it
# contains more than 1 linear broadcast, it is nonlinear.
function num_linear_broadcasts(
    ::Type{T},
    bc::Broadcasted{S, A, typeof(*)},
) where {T, S, A}
    s = sum(num_linear_broadcasts.(T, bc.args))
    return ifelse(s > 1, NaN, s)
end
# A quotient must contain a linear broadcast in only the numerator in order to
# be one. If it contains a linear or nonlinear broadcast in the denominator, it
# is nonlinear.
function num_linear_broadcasts(
    ::Type{T},
    bc::Broadcasted{S, A, typeof(/)},
) where {T, S, A}
    if num_linear_broadcasts(T, bc.args[2]) == 0
        return num_linear_broadcasts(T, bc.args[1])
    else
        return NaN
    end
end
# Any other operation that contains a linear or nonlinear broadcast is assumed
# to be nonlinear.
function num_linear_broadcasts(
    ::Type{T},
    bc::Broadcasted{S, A, F},
) where {T, S, A, F}
    return ifelse(sum(num_linear_broadcasts.(T, bc.args)) > 0, NaN, 0)
end
# A single object of type T is always a linear broadcast.
num_linear_broadcasts(::Type{T}, ::T) where {T} = 1
# An array that contains objects of type T is always a linear broadcast.
num_linear_broadcasts(::Type{T}, ::AbstractArray{T}) where {T} = 1
# Anything else is not a broadcast that involves objects of type T.
num_linear_broadcasts(::Type, ::Any) = 0

# Extract the values from a broadcasted operation that is known to be linear
# with respect to dual type T.
value_broadcast(::Type{T}, bc::Broadcasted) where {T} =
    Broadcasted(bc.f, value_broadcast.(T, bc.args), bc.axes)
value_broadcast(::Type{T}, Q::T) where {T} = value(Q)
value_broadcast(::Type{T}, Q::AbstractArray{<:T}) where {T} = value(Q)
value_broadcast(::Type, Q::Any) = Q

# Extract the partials from a broadcasted operation that is known to be linear
# with respect to dual type T. For sums and differences, this requires
# removing all arguments that do not have partials and extracting the partials
# from the remaining arguments. (This is equivalent to replacing the arguments
# that do not have partials with zeros.) Since the overall broadcast is known
# to be linear, at least one of its arguments must also be a linear broadcast.
partial_broadcast(::Type{T}, bc::Broadcasted{S, A, typeof(+)}, i) where {T, S, A} =
    Broadcasted(bc.f, partial_broadcast.(T, remove_no_partials(T, bc.args...), i), bc.axes)
partial_broadcast(::Type{T}, bc::Broadcasted{S, A, typeof(-), Tuple{<:Any}}, i) where {T, S, A} =
    Broadcasted(bc.f, (partial_broadcast(T, bc.args[1], i),), bc.axes)
function partial_broadcast(
    ::Type{T},
    bc::Broadcasted{S, A, typeof(-), Tuple{<:Any, <:Any}},
    i
) where {T, S, A}
    c1 = check_broadcast(T, bc.args[1])
    c2 = check_broadcast(T, bc.args[2])
    if c1 && c2
        return Broadcasted(bc.f, partial_broadcast.(T, bc.args, i), bc.axes)
    elseif c1
        return partial_broadcast(T, bc.args[1], i)
    else # c2
        return Broadcasted(bc.f, (partial_broadcast(T, bc.args[2], i),), bc.axes)
    end
end
partial_broadcast(::Type{T}, bc::Any, i) where {T} = _partial_broadcast(T, bc, i)

# Utility functions for extracting partials.
_partial_broadcast(::Type{T}, bc::Broadcasted, i) where {T} =
    Broadcasted(bc.f, _partial_broadcast.(T, bc.args, i), bc.axes)
_partial_broadcast(::Type{T}, Q::T, i) where {T} = partial(Q, i)
_partial_broadcast(::Type{T}, Q::AbstractArray{<:T}, i) where {T} = partial(Q, i)
_partial_broadcast(::Type, Q::Any, i) = Q
check_broadcast(::Type{T}, bc::Broadcasted) where {T} =
    any(check_broadcast.(T, bc.args))
check_broadcast(::Type{T}, ::T) where {T} = true
check_broadcast(::Type{T}, ::AbstractArray{T}) where {T} = true
check_broadcast(::Type, ::Any) = false
function remove_no_partials(::Type{T}, bc, args...) where {T}
    if check_broadcast(T, bc)
        return (bc, remove_no_partials(T, args...)...)
    else
        return remove_no_partials(T, args...)
    end
end
remove_no_partials(::Type) = ()