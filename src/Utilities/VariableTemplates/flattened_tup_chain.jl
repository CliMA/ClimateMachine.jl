export flattened_tup_chain, flattened_named_tuple

flattened_tup_chain(::Type{NamedTuple{(), Tuple{}}}; prefix = (Symbol(),)) = ()
flattened_tup_chain(::Type{T}; prefix = (Symbol(),)) where {T <: Real} =
    (prefix,)
flattened_tup_chain(::Type{T}; prefix = (Symbol(),)) where {T <: SArray} =
    (prefix,)
flattened_tup_chain(
    ::Type{T};
    prefix = (Symbol(),),
) where {T <: SHermitianCompact} = (prefix,)
flattened_tup_chain(::Type{T}; prefix = (Symbol(),)) where {T} = (prefix,)

"""
    flattened_tup_chain(::Type{T}) where {T <: Union{NamedTuple,NTuple}}

An array of tuples, containing symbols
and integers for every combination of
each field in the `Vars` array.
"""
function flattened_tup_chain(
    ::Type{T};
    prefix = (Symbol(),),
) where {T <: Union{NamedTuple, NTuple}}
    map(1:fieldcount(T)) do i
        Ti = fieldtype(T, i)
        name = fieldname(T, i)
        sname = name isa Int ? name : Symbol(name)
        flattened_tup_chain(
            Ti,
            prefix = prefix == (Symbol(),) ? (sname,) : (prefix..., sname),
        )
    end |>
    Iterators.flatten |>
    collect
end
flattened_tup_chain(::AbstractVars{S}) where {S} = flattened_tup_chain(S)

"""
    flattened_named_tuple(v::AbstractVars)

A flattened NamedTuple, given a `Vars` instance.

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

function flattened_named_tuple(v::AbstractVars)
    ftc = flattened_tup_chain(v)
    keys_ = Symbol.(join.(ftc, :_))
    vals = map(x -> getproperty(v, x), ftc)
    return (; zip(keys_, vals)...)
end
flattened_named_tuple(v::Nothing) = NamedTuple()

function flattened_named_tuple(nt::NamedTuple)
    ftc = flattened_tup_chain(typeof(nt))
    keys_ = Symbol.(join.(ftc, :_))
    vals = flattened_nt_vals(nt)
    return (; zip(keys_, vals)...)
end

flattened_nt_vals(a::NamedTuple) = flattened_nt_vals(Tuple(a))
flattened_nt_vals(a::NamedTuple{(), Tuple{}}) = (nothing,)
flattened_nt_vals(a) = (a,)
flattened_nt_vals(a::NamedTuple, b...) =
    tuple(flattened_nt_vals(a)..., flattened_nt_vals(b...)...)
flattened_nt_vals(a::NamedTuple{(), Tuple{}}, b...) =
    tuple(nothing, flattened_nt_vals(b...)...)
flattened_nt_vals(a, b...) = tuple(values(a), flattened_nt_vals(b...)...)
flattened_nt_vals(x::Tuple) = flattened_nt_vals(x...)
