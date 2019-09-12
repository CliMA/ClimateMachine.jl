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
- `vars_gradient(::L)`: a tuple of symbols containing the transformed variables of which gradients are computed
- `vars_diffusive(::L)`: a tuple of symbols containing the diffusive variables
- `flux_nondiffusive!(::L, flux::Grad, state::State, auxstate::State, t::Real)`
- `flux_diffusive!(::L, flux::Grad, state::State, diffstate::State, auxstate::State, t::Real)`
- `gradvariables!(::L, transformstate::State, state::State, auxstate::State, t::Real)`
- `diffusive!(::L, diffstate::State, ∇transformstate::Grad, auxstate::State, t::Real)`
- `source!(::L, source::State, state::State, auxstate::State, t::Real)`
- `wavespeed(::L, nM, state::State, aux::State, t::Real)`
- `boundary_state!(::GradNumericalPenalty, ::L, stateP::State, auxP::State, normalM, stateM::State, auxM::State, bctype, t)`
- `boundary_state!(::NumericalFluxNonDiffusive, ::L, stateP::State, auxP::State, normalM, stateM::State, auxM::State, bctype, t)`
- `boundary_state!(::NumericalFluxDiffusive, ::L, stateP::State, diffP::State, auxP::State, normalM, stateM::State, diffM::State, auxM::State, bctype, t)`
- `init_aux!(::L, aux::State, coords, args...)`
- `init_state!(::L, state::State, aux::State, coords, args...)`

"""
abstract type BalanceLaw end # PDE part

# function stubs
function vars_state end
function vars_aux end
function vars_gradient end
function vars_diffusive end
vars_integrals(::BalanceLaw, T) = @vars()
# init_ode_param(::DGModel, ::BalanceLaw) = nothing # Defined in DGmodel.jl

num_aux(m::BalanceLaw, T) = varsize(vars_aux(m,T)) 
num_state(m::BalanceLaw, T) = varsize(vars_state(m,T)) # nstate
num_gradient(m::BalanceLaw, T) = varsize(vars_gradient(m,T))  # number_gradient_states
num_diffusive(m::BalanceLaw, T) = varsize(vars_diffusive(m,T)) # number_viscous_states
num_integrals(m::BalanceLaw, T) = varsize(vars_integrals(m,T))

function update_aux! end
function integrate_aux! end
function flux_nondiffusive! end
function flux_diffusive! end
function gradvariables! end
function diffusive! end
function source! end 
function wavespeed end
function boundary_state! end
function init_aux! end
function init_state! end
