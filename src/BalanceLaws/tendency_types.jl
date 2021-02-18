#### Tendency types

# Terminology:
#
# `∂_t Yᵢ + (∇•F₁(Y))ᵢ + (∇•F₂(Y,G)))ᵢ = (S(Y,G))ᵢ`
# `__Tᵤ__   ____T₁____   ______T₂______   ___S___`
#
#  - `Yᵢ` - the i-th prognostic variable
#  - `Y` - the prognostic state (column vector)
#  - `G = ∇Y` - the gradient of the prognostic state (rank 2 tensor)
#  - `F₁` - the first order tendency flux (rank 2 tensor)
#  - `F₂` - the second order tendency flux (rank 2 tensor)
#  - `Tᵤ` - the explicit time derivative (column vector)
#  - `T₁` - the first order flux divergence (column vector)
#  - `T₂` - the second order flux divergence (column vector)
#  - `S` - the non-conservative source (column vector)

export PrognosticVariable,
    AbstractMomentum, AbstractEnergy, Moisture, Precipitation, AbstractTracers

export FirstOrder, SecondOrder
export AbstractTendencyType, Flux, Source
export TendencyDef
export eq_tends, prognostic_vars, fluxes, sources

"""
    PrognosticVariable

Subtypes are used for specifying
each prognostic variable.
"""
abstract type PrognosticVariable end

abstract type AbstractMomentum <: PrognosticVariable end
abstract type AbstractEnergy <: PrognosticVariable end
abstract type Moisture <: PrognosticVariable end
abstract type Precipitation <: PrognosticVariable end
abstract type AbstractTracers{N} <: PrognosticVariable end


"""
    AbstractOrder

Subtypes are used for dispatching
on the flux order.
"""
abstract type AbstractOrder end

"""
    FirstOrder

A type for dispatching on first order fluxes
"""
struct FirstOrder <: AbstractOrder end

"""
    SecondOrder

A type for dispatching on second order fluxes
"""
struct SecondOrder <: AbstractOrder end

"""
    AbstractTendencyType

Subtypes are used for specifying a
tuple of tendencies to be accumulated.
"""
abstract type AbstractTendencyType end

"""
    Flux{O <: AbstractOrder}

A type for dispatching on flux tendency types
where `O` is an abstract order ([`FirstOrder`](@ref)
or [`SecondOrder`](@ref)).
"""
struct Flux{O <: AbstractOrder} <: AbstractTendencyType end

"""
    Source

A type for dispatching on source tendency types
"""
struct Source <: AbstractTendencyType end

"""
    TendencyDef

Subtypes are used for specifying
each tendency definition.
"""
abstract type TendencyDef{TT <: AbstractTendencyType, PV <: PrognosticVariable} end

"""
    eq_tends(::PrognosticVariable, ::BalanceLaw, ::AbstractTendencyType)

A tuple of `TendencyDef`s given
 - `PrognosticVariable` prognostic variable
 - `AbstractTendencyType` tendency type
 - `BalanceLaw` balance law

i.e., a tuple of `TendencyDef`s corresponding
to `F₁`, `F₂`, **or** `S` for a single
prognostic variable in:

    `∂_t Yᵢ + (∇•F₁(Y))ᵢ + (∇•F₂(Y,G)))ᵢ = (S(Y,G))ᵢ`
"""
function eq_tends end

"""
    prognostic_vars(::BalanceLaw)

A tuple of `PrognosticVariable`s given
the `BalanceLaw`.

i.e., a tuple of `PrognosticVariable`s
corresponding to the column-vector `Yᵢ` in:

    `∂_t Yᵢ + (∇•F₁(Y))ᵢ + (∇•F₂(Y,G)))ᵢ = (S(Y,G))ᵢ`
"""
prognostic_vars(::BalanceLaw) = ()

"""
    var, name = get_prog_state(state::Union{Vars, Grad}, pv::PrognosticVariable)

Returns a tuple of two elements. `var` is a `Vars` or `Grad`
object, and `name` is a Symbol. They should be linked such that
`getproperty(var, name)` returns the corresponding prognostic
variable type `pv`.

# Example

```julia
get_prog_state(state, ::Moisture) = (state.moisture, :ρq_tot)
var, name = get_prog_state(state, Moisture())
@test getproperty(var, name) == state.moisture.ρq_tot
```
"""
function get_prog_state end

"""
    sources(bl::BalanceLaw)

A tuple of `TendencyDef{Source}`s
given the `BalanceLaw`.

i.e., a tuple of `TendencyDef{Source}`s
corresponding to the column-vector `S` in:

    `∂_t Yᵢ + (∇•F₁(Y))ᵢ + (∇•F₂(Y,G)))ᵢ = (S(Y,G))ᵢ`
"""
function sources(bl::BalanceLaw)
    tend = eq_tends.(prognostic_vars(bl), Ref(bl), Ref(Source()))
    tend = filter(x -> x ≠ nothing, tend)
    return Tuple(Iterators.flatten(tend))
end

"""
    fluxes(bl::BalanceLaw, order::O) where {O <: AbstractOrder}

A tuple of `TendencyDef{Flux{O}}`s
given the `BalanceLaw` and the `order::O`.

i.e., a tuple of `TendencyDef{Flux{O}}`s
corresponding to the column-vector `F₁`
or `F₂` given the flux order `order::O` in:

    `∂_t Yᵢ + (∇•F₁(Y))ᵢ + (∇•F₂(Y,G)))ᵢ = (S(Y,G))ᵢ`
"""
function fluxes(bl::BalanceLaw, order::O) where {O <: AbstractOrder}
    tend = eq_tends.(prognostic_vars(bl), Ref(bl), Ref(Flux{O}()))
    tend = filter(x -> x ≠ nothing, tend)
    return Tuple(Iterators.flatten(tend))
end

"""
    precompute(bl, args, ::AbstractTendencyType)

A nested NamedTuple of precomputed (cached) values
and or objects. This is useful for "expensive"
point-wise quantities that are used in multiple
tendency terms. For example, computing a quantity
that requires iteration.
"""
precompute(bl, args, ::AbstractTendencyType) = NamedTuple()
