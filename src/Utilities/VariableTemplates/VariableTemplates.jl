module VariableTemplates

export varsize, Vars, Grad, @vars, varsindex, varsindices

using StaticArrays

"""
    varsindex(S, p::Symbol, [sp::Symbol...])

Return a range of indices corresponding to the property `p` and
(optionally) its subproperties `sp` based on the template type `S`.

# Examples
```julia-repl
julia> S = @vars(x::Float64, y::Float64)
julia> varsindex(S, :y)
2:2

julia> S = @vars(x::Float64, y::@vars(α::Float64, β::SVector{3, Float64}))
julia> varsindex(S, :y, :β)
3:5
```
"""
function varsindex(::Type{S}, insym::Symbol) where {S <: NamedTuple}
    offset = 0
    for varsym in fieldnames(S)
        T = fieldtype(S, varsym)
        if T <: Real
            offset += 1
            varrange = offset:offset
        elseif T <: SHermitianCompact
            LT = StaticArrays.lowertriangletype(T)
            N = length(LT)
            varrange = offset .+ (1:N)
            offset += N
        elseif T <: StaticArray
            N = length(T)
            varrange = offset .+ (1:N)
            offset += N
        else
            varrange = offset .+ (1:varsize(T))
            offset += varsize(T)
        end
        if insym == varsym
            return varrange
        end
    end
    error("symbol '$insym' not found")
end
Base.@propagate_inbounds function varsindex(
    ::Type{S},
    sym::Symbol,
    rest::Symbol...,
) where {S <: NamedTuple}
    varsindex(S, sym)[varsindex(fieldtype(S, sym), rest...)]
end

"""
    varsindices(S, ps::Tuple)
    varsindices(S, ps...)

Return a tuple of indices corresponding to the properties
specified by `ps` based on the template type `S`. Properties
can be specified using either symbols or strings.

# Examples
```julia-repl
julia> S = @vars(x::Float64, y::Float64, z::Float64)
julia> varsindices(S, (:x, :z))
(1, 3)

julia> S = @vars(x::Float64, y::@vars(α::Float64, β::SVector{3, Float64}))
julia> varsindices(S, "x", "y.β")
(1, 3, 4, 5)
```
"""
function varsindices(::Type{S}, vars::Tuple) where {S <: NamedTuple}
    indices = Int[]
    for var in vars
        splitvar = split(string(var), '.')
        append!(indices, collect(varsindex(S, map(Symbol, splitvar)...)))
    end
    Tuple(indices)
end
varsindices(::Type{S}, vars...) where {S <: NamedTuple} = varsindices(S, vars)

"""
    varsize(S)

The number of elements specified by the template type `S`.
"""
varsize(::Type{T}) where {T <: Real} = 1
varsize(::Type{Tuple{}}) = 0
varsize(::Type{NamedTuple{(), Tuple{}}}) = 0
varsize(::Type{SVector{N, T}}) where {N, T <: Real} = N

include("var_names.jl")
include("flattened_tup_chain.jl")

# TODO: should be possible to get rid of @generated
@generated function varsize(::Type{S}) where {S}
    types = fieldtypes(S)
    isempty(types) ? 0 : sum(varsize, types)
end

function process_vars!(syms, typs, expr)
    if expr isa LineNumberNode
        return
    elseif expr isa Expr && expr.head == :block
        for arg in expr.args
            process_vars!(syms, typs, arg)
        end
        return
    elseif expr.head == :(::)
        push!(syms, expr.args[1])
        push!(typs, expr.args[2])
        return
    else
        error("Invalid expression")
    end
end

macro vars(args...)
    syms = Any[]
    typs = Any[]
    for arg in args
        process_vars!(syms, typs, arg)
    end
    :(NamedTuple{$(tuple(syms...)), Tuple{$(esc.(typs)...)}})
end

struct GetVarError <: Exception
    sym::Symbol
end
struct SetVarError <: Exception
    sym::Symbol
end

abstract type AbstractVars{S, A, offset} end

"""
    Vars{S,A,offset}(array::A)

Defines property overloading for `array` using the type `S` as a template. `offset` is used to shift the starting element of the array.
"""
struct Vars{S, A, offset} <: AbstractVars{S, A, offset}
    array::A
end
Vars{S}(array) where {S} = Vars{S, typeof(array), 0}(array)

Base.parent(v::AbstractVars) = getfield(v, :array)
Base.eltype(v::AbstractVars) = eltype(parent(v))
Base.propertynames(::AbstractVars{S}) where {S} = fieldnames(S)
Base.similar(v::AbstractVars) = typeof(v)(similar(parent(v)))

@generated function Base.getproperty(
    v::Vars{S, A, offset},
    sym::Symbol,
) where {S, A, offset}
    expr = quote
        Base.@_inline_meta
        array = parent(v)
    end
    for k in fieldnames(S)
        T = fieldtype(S, k)
        if T <: Real
            retexpr = :($T(array[$(offset + 1)]))
            offset += 1
        elseif T <: SHermitianCompact
            LT = StaticArrays.lowertriangletype(T)
            N = length(LT)
            retexpr = :($T($LT($([:(array[$(offset + i)]) for i in 1:N]...))))
            offset += N
        elseif T <: StaticArray
            N = length(T)
            retexpr = :($T($([:(array[$(offset + i)]) for i in 1:N]...)))
            offset += N
        else
            retexpr = :(Vars{$T, A, $offset}(array))
            offset += varsize(T)
        end
        push!(expr.args, :(
            if sym == $(QuoteNode(k))
                return @inbounds $retexpr
            end
        ))
    end
    push!(expr.args, :(throw(GetVarError(sym))))
    expr
end

@generated function Base.setproperty!(
    v::Vars{S, A, offset},
    sym::Symbol,
    val,
) where {S, A, offset}
    expr = quote
        Base.@_inline_meta
        array = parent(v)
    end
    for k in fieldnames(S)
        T = fieldtype(S, k)
        if T <: Real
            retexpr = :(array[$(offset + 1)] = val)
            offset += 1
        elseif T <: SHermitianCompact
            LT = StaticArrays.lowertriangletype(T)
            N = length(LT)
            retexpr = :(
                array[($(offset + 1)):($(offset + N))] .= $T(val).lowertriangle
            )
            offset += N
        elseif T <: StaticArray
            N = length(T)
            retexpr = :(array[($(offset + 1)):($(offset + N))] .= val[:])
            offset += N
        else
            offset += varsize(T)
            continue
        end
        push!(expr.args, :(
            if sym == $(QuoteNode(k))
                return @inbounds $retexpr
            end
        ))
    end
    push!(expr.args, :(throw(SetVarError(sym))))
    expr
end

"""
    Grad{S,A,offset}(array::A)

Defines property overloading along slices of the second dimension of `array` using the type `S` as a template. `offset` is used to shift the starting element of the array.
"""
struct Grad{S, A, offset} <: AbstractVars{S, A, offset}
    array::A
end
Grad{S}(array) where {S} = Grad{S, typeof(array), 0}(array)

@generated function Base.getproperty(
    v::Grad{S, A, offset},
    sym::Symbol,
) where {S, A, offset}
    if A <: SubArray
        M = size(fieldtype(A, 1), 1)
    else
        M = size(A, 1)
    end
    expr = quote
        Base.@_inline_meta
        array = parent(v)
    end
    for k in fieldnames(S)
        T = fieldtype(S, k)
        if T <: Real
            retexpr = :(SVector{$M, $T}(
                $([:(array[$i, $(offset + 1)]) for i in 1:M]...),
            ))
            offset += 1
        elseif T <: StaticArray
            N = length(T)
            retexpr = :(SMatrix{$M, $N, $(eltype(T))}(
                $([:(array[$i, $(offset + j)]) for i in 1:M, j in 1:N]...),
            ))
            offset += N
        else
            retexpr = :(Grad{$T, A, $offset}(array))
            offset += varsize(T)
        end
        push!(expr.args, :(
            if sym == $(QuoteNode(k))
                return @inbounds $retexpr
            end
        ))
    end
    push!(expr.args, :(throw(GetVarError(sym))))
    expr
end

@generated function Base.setproperty!(
    v::Grad{S, A, offset},
    sym::Symbol,
    val,
) where {S, A, offset}
    if A <: SubArray
        M = size(fieldtype(A, 1), 1)
    else
        M = size(A, 1)
    end
    expr = quote
        Base.@_inline_meta
        array = parent(v)
    end
    for k in fieldnames(S)
        T = fieldtype(S, k)
        if T <: Real
            retexpr = :(array[:, $(offset + 1)] .= val)
            offset += 1
        elseif T <: StaticArray
            N = length(T)
            retexpr = :(array[:, ($(offset + 1)):($(offset + N))] .= val)
            offset += N
        else
            offset += varsize(T)
            continue
        end
        push!(expr.args, :(
            if sym == $(QuoteNode(k))
                return @inbounds $retexpr
            end
        ))
    end
    push!(expr.args, :(throw(SetVarError(sym))))
    expr
end


Base.@propagate_inbounds function Base.getindex(
    v::AbstractVars{NTuple{N, T}, A, offset},
    i::Int,
) where {N, T, A, offset}
    # 1 <= i <= N
    array = parent(v)
    if v isa Vars
        return Vars{T, A, offset + (i - 1) * varsize(T)}(array)
    else
        return Grad{T, A, offset + (i - 1) * varsize(T)}(array)
    end
end

Base.@propagate_inbounds function Base.getproperty(
    v::AbstractVars,
    tup_chain::Tuple{S},
) where {S <: Symbol}
    return Base.getproperty(v, tup_chain[1])
end

Base.@propagate_inbounds function Base.getindex(
    v::AbstractVars,
    tup_chain::Tuple{S},
) where {S <: Int}
    return Base.getindex(v, tup_chain[1])
end

Base.@propagate_inbounds function Base.getproperty(
    v::AbstractVars,
    tup_chain::Tuple,
)
    if tup_chain[1] isa Int
        p = Base.getindex(v, tup_chain[1])
    else
        p = Base.getproperty(v, tup_chain[1])
    end
    if tup_chain[2] isa Int
        return Base.getindex(p, tup_chain[2:end])
    else
        return Base.getproperty(p, tup_chain[2:end])
    end
end
Base.@propagate_inbounds function Base.getindex(
    v::AbstractVars,
    tup_chain::Tuple,
)
    if tup_chain[1] isa Int
        p = Base.getindex(v, tup_chain[1])
    else
        p = Base.getproperty(v, tup_chain[1])
    end
    if tup_chain[2] isa Int
        return Base.getindex(p, tup_chain[2:end])
    else
        return Base.getproperty(p, tup_chain[2:end])
    end
end

end # module
