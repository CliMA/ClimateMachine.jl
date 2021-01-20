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

# return `Symbol`s unchanged.
wrap_val(sym) = sym
# wrap integer in `Val`
wrap_val(i::Int) = Val(i)

# We enforce that calls to `varsindex` on
# an `NTuple` must be unrapped in `Val`.
# This is enforced to synchronize failures
# on the CPU and GPU, rather than allowing
# CPU-working and GPU-breaking versions.
# This means that users _must_ wrap `sym`
# in `Val`, which can be done with `wrap_val`
# above.
Base.@propagate_inbounds function varsindex(
    ::Type{S},
    sym::Symbol,
    rest...,
) where {S <: Union{NamedTuple, Tuple}}
    vi = varsindex(fieldtype(S, sym), rest...)
    return varsindex(S, sym)[vi]
end
Base.@propagate_inbounds function varsindex(
    ::Type{S},
    ::Val{i},
    rest...,
) where {i, S <: Union{NamedTuple, Tuple}}
    et = eltype(S)
    offset = (i - 1) * varsize(et)
    vi = varsindex(et, rest...)
    return (vi.start + offset):(vi.stop + offset)
end

Base.@propagate_inbounds function varsindex(
    ::Type{S},
    ::Val{i},
) where {i, S <: SArray}
    return i:i
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
varsize(::Type{SArray{S, T, N} where L}) where {S, T, N} = prod(S.parameters)

include("var_names.jl")

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

"""
    @vars(var1::Type1, var2::Type2)

A convenient syntax for describing a `NamedTuple` type.

# Example
```julia
julia> @vars(a::Float64, b::Float64)
NamedTuple{(:a, :b),Tuple{Float64,Float64}}
```
"""
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
                array[($(offset + 1)):($(offset + N))] .=
                    $T(val).lowertriangle
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
    val::AbstractArray,
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
            retexpr = :(array[:, $(offset + 1)] = val)
            offset += 1
        elseif T <: StaticArray
            N = length(T)
            retexpr = :(
                array[
                    :,
                    # static range is used here to force dispatch to
                    # StaticArrays setindex! because generic setindex! is slow
                    StaticArrays.SUnitRange($(offset + 1), $(offset + N)),
                ] = val
            )
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

export unroll_map, @unroll_map
"""
    @unroll_map(f::F, N::Int, args...) where {F}
    unroll_map(f::F, N::Int, args...) where {F}

Unroll N-expressions and wrap arguments in `Val`.
"""
@generated function unroll_map(f::F, ::Val{N}, args...) where {F, N}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i -> f(Val(i), args...)
    end
end
macro unroll_map(func, N, args...)
    @assert func.head == :(->)
    body = func.args[2]
    pushfirst!(body.args, :(Base.@_inline_meta))
    quote
        $unroll_map($(esc(func)), Val($(esc(N))), $(esc(args))...)
    end
end

export vuntuple
"""
    vuntuple(f::F, N::Int)

Val-Unroll ntuple: wrap `ntuple`
arguments in `Val` for unrolling.
"""
vuntuple(f::F, N::Int) where {F} = ntuple(i -> f(Val(i)), Val(N))

# Inside unroll_map expressions, all indexes `i`
# are wrapped in `Val`, so we must redirect
# these methods:
Base.@propagate_inbounds Base.getindex(t::Tuple, ::Val{i}) where {i} =
    Base.getindex(t, i)
Base.@propagate_inbounds Base.getindex(a::SArray, ::Val{i}) where {i} =
    Base.getindex(a, i)

Base.@propagate_inbounds function Base.getindex(
    v::Vars{NTuple{N, T}, A, offset},
    ::Val{i},
) where {N, T, A, offset, i} # 1 <= i <= N
    return Vars{T, A, offset + (i - 1) * varsize(T)}(parent(v))
end

Base.@propagate_inbounds function Base.getindex(
    v::Grad{NTuple{N, T}, A, offset},
    ::Val{i},
) where {N, T, A, offset, i} # 1 <= i <= N
    return Grad{T, A, offset + (i - 1) * varsize(T)}(parent(v))
end

"""
    getpropertyorindex

An interchangeably and nested-friendly
`getproperty`/`getindex`.
"""
function getpropertyorindex end

# Redirect to Base getproperty/getindex:
Base.@propagate_inbounds getpropertyorindex(t::Tuple, ::Val{i}) where {i} =
    Base.getindex(t, i)
Base.@propagate_inbounds getpropertyorindex(
    a::AbstractArray,
    ::Val{i},
) where {i} = Base.getindex(a, i)
Base.@propagate_inbounds getpropertyorindex(nt::AbstractVars, s::Symbol) =
    Base.getproperty(nt, s)
Base.@propagate_inbounds getpropertyorindex(
    v::AbstractVars,
    ::Val{i},
) where {i} = Base.getindex(v, Val(i))

# Only one element left:
Base.@propagate_inbounds getpropertyorindex(
    v::AbstractVars,
    t::Tuple{A},
) where {A} = getpropertyorindex(v, t[1])
Base.@propagate_inbounds getpropertyorindex(
    a::AbstractArray,
    t::Tuple{A},
) where {A} = getpropertyorindex(a, t[1])

# Peel first element from tuple and recurse:
Base.@propagate_inbounds getpropertyorindex(v::AbstractVars, t::Tuple) =
    getpropertyorindex(getpropertyorindex(v, t[1]), Tuple(t[2:end]))

# Redirect to getpropertyorindex:
Base.@propagate_inbounds Base.getproperty(v::AbstractVars, tup_chain::Tuple) =
    getpropertyorindex(v, tup_chain)
Base.@propagate_inbounds Base.getindex(v::AbstractVars, tup_chain::Tuple) =
    getpropertyorindex(v, tup_chain)

include("flattened_tup_chain.jl")

end # module
