"""
    BalanceLaw

An abstract type representing a PDE balance law of the form

elements for balance laws of the form

```math
q_{,t} + Σ_{i=1,...d} F_{i,i} = s
```

Subtypes `L` should define the following methods:
- `dimension(::L)` the number of dimensions
- `varmap_aux(::L)`: a tuple of symbols containing the auxiliary variables
- `varmap_state(::L)`: a tuple of symbols containing the state variables
- `varmap_state_for_transform(::L)`: a tuple of symbols containing the state variables which are passed to the `transform!` function.
- `varmap_transform(::L)`: a tuple of symbols containing the transformed variables of which gradients are computed
- `varmap_diffusive(::L)`: a tuple of symbols containing the diffusive variables
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

has_diffusive(m::BalanceLaw) = num_diffusive(m) > 0

# function stubs
function num_aux end
function num_state end
function num_gradtransform end
function num_diffusive end

function num_state_for_gradtransform end

function dimension end
function varmap_aux end
function varmap_state end
function varmap_gradtransform end
function varmap_diffusive end

function varmap_state_for_gradtransform end

function flux! end
function gradtransform! end
function diffusive! end
function source! end 
function wavespeed end
function boundarycondition! end
function init_aux! end
function init_state! end



# TODO: allow aliases and vector values
struct State{A<:StaticVector,V}
  arr::A
  varmap::V
end

Base.propertynames(s::State) = propertynames(s.varmap)
@inline function Base.getproperty(s::State, sym::Symbol)
  i = getfield(s.varmap, sym)
  if i isa Integer
    return getfield(s,:arr)[i]
  else
    return getfield(s,:arr)[SVector(i...)]
  end
end
@inline function Base.setproperty!(s::State, sym::Symbol, val)
  i = getfield(s.varmap, sym)
  if i isa Integer
    return getfield(s,:arr)[i] = val
  else
    return getfield(s,:arr)[SVector(i...)] = val
  end
end


struct Grad{A<:StaticMatrix,V}
  arr::A
  varmap::V
end

Base.propertynames(s::Grad) = propertynames(s.varmap)
@inline function Base.getproperty(∇s::Grad, sym::Symbol)
  i = getfield(∇s.varmap, sym)
  if i isa Integer
    return getfield(∇s,:arr)[:,i]
  else
    return getfield(∇s,:arr)[:,SVector(i...)]
  end
end
@inline function Base.setproperty!(∇s::Grad, sym::Symbol, val)
  i = getfield(∇s.varmap, sym)
  if i isa Integer
    return getfield(∇s,:arr)[:,i] = val
  else
    return getfield(∇s,:arr)[:,SVector(i...)] = val
  end
end
