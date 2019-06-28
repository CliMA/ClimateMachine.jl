"""
    BalanceLaw

An abstract type representing a PDE balance law of the form

elements for balance laws of the form

```math
q_{,t} + Σ_{i=1,...d} F_{i,i} = s
```

Subtypes `L` should define the following methods:
- `vars_aux(::L)`: a tuple of symbols containing the auxiliary variables
- `vars_state(::L)`: a tuple of symbols containing the state variables
- `vars_state_for_transform(::L)`: a tuple of symbols containing the state variables which are passed to the `transform!` function.
- `vars_transform(::L)`: a tuple of symbols containing the transformed variables of which gradients are computed
- `vars_diffusive(::L)`: a tuple of symbols containing the diffusive variables
- `flux!(::L, flux::Grad, state::State, diffstate::State, auxstate::State, t::Real)`
- `gradtransform!(::L, transformstate::State, state::State, auxstate::State, t::Real)`
- `diffusive!(::L, diffstate::State, ∇transformstate::Grad, auxstate::State, t::Real)`
- `source!(::L, source::State, state::State, auxstate::State, t::Real)`
- `wavespeed(::L, nM, state::State, aux::State, t::Real)`
- `boundarycondition!(::L, stateP::State, diffP::State, auxP::State, normalM, stateM::State, diffM::State, auxM::State, bctype, t)`
- `init_aux!(::L, aux::State, coords, args...)`
- `init_state!(::L, state::State, aux::State, coords, args...)`

"""
abstract type BalanceLaw end # PDE part

num_aux(m::BalanceLaw) = length(vars_aux(m)) 
num_state(m::BalanceLaw) = length(vars_state(m)) # nstate
num_gradtransform(m::BalanceLaw) = length(vars_gradtransform(m))  # number_gradient_states
num_diffusive(m::BalanceLaw) = length(vars_diffusive(m)) # number_viscous_states

has_diffusive(m::BalanceLaw) = num_diffusive(m) > 0

num_state_for_gradtransform(bl::BalanceLaw) = num_state(m)

# function stubs
function vars_aux end
function vars_state end
function vars_state_for_gradtransform end
function vars_gradtransform end
function vars_diffusive end

function flux! end
function gradtransform! end
function diffusive! end
function source! end 
function wavespeed end
function boundarycondition! end
function init_aux! end
function init_state! end



# TODO: allow aliases and vector values
struct State{vars, A<:StaticVector}
  arr::A
end
State{vars}(arr::A) where {vars,A<:StaticVector} = State{vars,A}(arr)

struct GetFieldException <: Exception
  sym::Symbol
end



Base.propertynames(s::State{vars}) where {vars} = vars
@generated function Base.getproperty(s::State{vars}, sym::Symbol) where {vars}
  expr = quote
      Base.@_inline_meta
  end
  i = 0
  for var in vars
    push!(expr.args, quote
      if sym == $(QuoteNode(var))
        return getfield(s,:arr)[$(i+=1)]
      end
    end)
  end
  push!(expr.args, :(throw(GetFieldException(sym))))
  expr
end
@generated function Base.setproperty!(s::State{vars}, sym::Symbol, val) where {vars}
  expr = quote
    Base.@_inline_meta
  end
  i = 0
  for var in vars
    push!(expr.args, quote
      if sym == $(QuoteNode(var))
        return getfield(s,:arr)[$(i+=1)] = val
      end
    end)
  end
  push!(expr.args, :(throw(GetFieldException(sym))))
  expr
end
Base.eltype(s::State) = eltype(getfield(s,:arr))

struct Grad{vars, A<:StaticMatrix}
  arr::A
end
Grad{vars}(arr::A) where {vars,A<:StaticMatrix} = Grad{vars,A}(arr)

Base.propertynames(s::Grad{vars}) where {vars} = vars
@generated function Base.getproperty(∇s::Grad{vars}, sym::Symbol) where {vars}
  expr = quote
    Base.@_inline_meta
  end
  i = 0
  for var in vars
    push!(expr.args, quote
      if sym == $(QuoteNode(var))
        return getfield(∇s,:arr)[:, $(i+=1)]
      end
    end)
  end
  push!(expr.args, :(throw(GetFieldException(sym))))
  expr
end
@generated function Base.setproperty!(∇s::Grad{vars}, sym::Symbol, val) where {vars}
  expr = quote
    Base.@_inline_meta
  end
  i = 0
  for var in vars
    push!(expr.args, quote
      if sym == $(QuoteNode(var))
        return getfield(∇s,:arr)[:, $(i+=1)] = val
      end
    end)
  end
  push!(expr.args, :(throw(GetFieldException(sym))))
  expr
end

Base.eltype(∇s::Grad) = eltype(getfield(∇s,:arr))
