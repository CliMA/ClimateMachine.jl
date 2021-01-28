using LinearAlgebra

export flattened_tup_chain, flattened_named_tuple, flattened_tuple
export FlattenType, FlattenArr, RetainArr

abstract type FlattenType end

"""
    FlattenArr

Flatten arrays in `flattened_tup_chain`
and `flattened_named_tuple`.
"""
struct FlattenArr <: FlattenType end

"""
    RetainArr

Do _not_ flatten arrays in `flattened_tup_chain`
and `flattened_named_tuple`.
"""
struct RetainArr <: FlattenType end

# The Vars instance has many empty entries.
# Keeping all of the keys results in many
# duplicated values. So, it's best we
# "prune" the tree by removing the keys:
flattened_tup_chain(
    ::Type{NamedTuple{(), Tuple{}}},
    ::FlattenType = FlattenArr();
    prefix = (Symbol(),),
) = ()

flattened_tup_chain(
    ::Type{T},
    ::FlattenType;
    prefix = (Symbol(),),
) where {T <: Real} = (prefix,)

flattened_tup_chain(
    ::Type{T},
    ::RetainArr;
    prefix = (Symbol(),),
) where {T <: SArray} = (prefix,)
flattened_tup_chain(
    ::Type{T},
    ::FlattenArr;
    prefix = (Symbol(),),
) where {T <: SArray} = ntuple(i -> (prefix..., i), length(T))

flattened_tup_chain(
    ::Type{T},
    ::RetainArr;
    prefix = (Symbol(),),
) where {T <: SHermitianCompact} = (prefix,)
flattened_tup_chain(
    ::Type{T},
    ::FlattenType;
    prefix = (Symbol(),),
) where {T <: SHermitianCompact} =
    ntuple(i -> (prefix..., i), length(StaticArrays.lowertriangletype(T)))

flattened_tup_chain(
    ::Type{T},
    ::RetainArr;
    prefix = (Symbol(),),
) where {N, TA, T <: Diagonal{N, TA}} = (prefix,)
flattened_tup_chain(
    ::Type{T},
    ::FlattenArr;
    prefix = (Symbol(),),
) where {N, TA, T <: Diagonal{N, TA}} = ntuple(i -> (prefix..., i), length(TA))

flattened_tup_chain(::Type{T}, ::FlattenType; prefix = (Symbol(),)) where {T} =
    (prefix,)

"""
    flattened_tup_chain(::Type{T}) where {T <: Union{NamedTuple,NTuple}}

An array of tuples, containing symbols
and integers for every combination of
each field in the `Vars` array.
"""
function flattened_tup_chain(
    ::Type{T},
    ft::FlattenType = FlattenArr();
    prefix = (Symbol(),),
) where {T <: Union{NamedTuple, NTuple}}
    map(1:fieldcount(T)) do i
        Ti = fieldtype(T, i)
        name = fieldname(T, i)
        sname = name isa Int ? name : Symbol(name)
        flattened_tup_chain(
            Ti,
            ft;
            prefix = prefix == (Symbol(),) ? (sname,) : (prefix..., sname),
        )
    end |>
    Iterators.flatten |>
    collect
end
flattened_tup_chain(
    ::AbstractVars{S},
    ft::FlattenType = FlattenArr(),
) where {S} = flattened_tup_chain(S, ft)

"""
    flattened_named_tuple

A flattened NamedTuple, given a
`Vars` or nested `NamedTuple` instance.

# Example:

```julia
using Test
using ClimateMachine.VariableTemplates
nt = (x = 1, a = (y = 2, z = 3, b = ((a = 1,), (a = 2,), (a = 3,))));
fnt = flattened_named_tuple(nt);
@test keys(fnt) == (:x, :a_y, :a_z, :a_b_1_a, :a_b_2_a, :a_b_3_a)
@test length(fnt) == 6
@test fnt.x == 1
@test fnt.a_y == 2
@test fnt.a_z == 3
@test fnt.a_b_1_a == 1
@test fnt.a_b_2_a == 2
@test fnt.a_b_3_a == 3
```
"""
function flattened_named_tuple end

function flattened_named_tuple(v::AbstractVars, ft::FlattenType = FlattenArr())
    ftc = flattened_tup_chain(v, ft)
    keys_ = Symbol.(join.(ftc, :_))
    vals = map(x -> getproperty(v, wrap_val.(x)), ftc)
    length(keys_) == length(vals) || error("key-value mismatch")
    return (; zip(keys_, vals)...)
end

function flattened_named_tuple(nt::NamedTuple, ft::FlattenType = FlattenArr())
    ftc = flattened_tup_chain(typeof(nt), ft)
    keys_ = Symbol.(join.(ftc, :_))
    vals = flattened_tuple(ft, nt)
    length(keys_) == length(vals) || error("key-value mismatch")
    return (; zip(keys_, vals)...)
end

flattened_tuple(::FlattenArr, a::AbstractArray) = tuple(a...)
flattened_tuple(::RetainArr, a::AbstractArray) = tuple(a)

flattened_tuple(::FlattenArr, a::Diagonal) = tuple(a.diag...)
flattened_tuple(::RetainArr, a::Diagonal) = tuple(a.diag)

flattened_tuple(::FlattenArr, a::SHermitianCompact) =
    tuple(a.lowertriangle...)
flattened_tuple(::RetainArr, a::SHermitianCompact) = tuple(a.lowertriangle)

# when we splat an empty tuple `b` into `flattened_tuple(ft, b...)`
flattened_tuple(::FlattenType) = ()

# for structs
flattened_tuple(::FlattenType, a) = (a,)

# Divide and conquer:
flattened_tuple(ft::FlattenType, a, b...) =
    tuple(flattened_tuple(ft, a)..., flattened_tuple(ft, b...)...)

flattened_tuple(ft::FlattenType, a::Tuple) = flattened_tuple(ft, a...)

flattened_tuple(ft::FlattenType, a::NamedTuple) =
    flattened_tuple(ft, Tuple(a))
