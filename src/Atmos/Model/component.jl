#### Abstract component

export Component
export property
export Dirichlet, Neumann
export ScalarField, VectorField

abstract type ComponentNames end
property(::ComponentNames) = nothing

abstract type AbstractField end
struct ScalarField <: AbstractField end
struct VectorField <: AbstractField end

abstract type BoundaryConditions{AbstractField} end
struct Dirichlet{AF,T,F} <: BoundaryConditions{AF}
  val::T
  fun::F
end
Dirichlet{F}(args...) where F = Dirichlet{F,typeof.(args)...}(args...)

struct Neumann{AF,T,F} <: BoundaryConditions{AF}
  val::T
  fun::F
end
Neumann{F}(args...) where F = Neumann{F,typeof.(args)...}(args...)

abstract type AbstractComponent end

"""
    Component{Name,I,B,S,FD,FND} <: AbstractComponent

Constructor:

  Component(Mass)
"""
struct Component{Name,I,B,RS,S,FD,FND} <: AbstractComponent
  """ Initial conditions """
  ics::I
  """ Boundary conditions """
  bcs::B
  """ Reference state """
  refstate::RS
  """ Source terms """
  sources::S
  """ Diffusive fluxes """
  fluxes_diffusive::FD
  """ Non-diffusive fluxes """
  fluxes_non_diffusive::FND
end
Component{Name}(args...) where Name = Component{Name,typeof.(args)...}(args...)

Component(Name;
          ics=nothing,
          bcs=nothing,
          refstate=nothing,
          sources=nothing,
          fluxes_diffusive=nothing,
          fluxes_non_diffusive=nothing) =
  Component{Name}(ics,
                  bcs,
                  refstate,
                  sources,
                  fluxes_diffusive,
                  fluxes_non_diffusive)

vars_state(c::Component    , T) = @vars()
vars_gradient(c::Component , T) = @vars()
vars_diffusive(c::Component, T) = @vars()
vars_integrals(c::Component, T) = @vars()
vars_aux(c::Component      , T) = @vars()

"""
    flux!(c::Component, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)

Computes flux `F` in:

```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
"""
flux!(c::Component, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing
update_aux!(c::Component, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing
integrate_aux!(c::Component, integ::Vars, state::Vars, aux::Vars) = nothing
init_aux!(c::Component, aux::Vars, geom::LocalGeometry) = nothing

"""
    source!(c::Component, source::Vars, state::Vars, aux::Vars, t::Real)

Computes flux `S(Y)` in:

```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
"""
source!(c::Component, source::Vars, state::Vars, aux::Vars, t::Real) = nothing
atmos_source!(c::Component, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real) = nothing

#### Initial conditions
""" Default initial state """
function atmos_init_state!(c::Component{Name}, state::Vars, aux::Vars, coords, t) where Name
  # TODO: Unroll loop
  for f in fieldnames(vars_state(c, eltype(state)))
    localstate = getproperty(state, propname(Name))
    setproperty!(localstate, f, eltype(state)(0))
  end
end

""" When a custom initial state kernel is passed """
function atmos_init_state!(c::Component{N,I,B,RS,S,FD,FND}, state::Vars, aux::Vars, coords, t) where {N,I<:Function,B,RS,S,FD,FND}
  c.ics(state, aux, coords, t)
end

#### Boundary conditions
""" No flux BCs (stateP = stateM) """
function atmos_boundarycondition!(c::Component{Name,I,NoFluxBC,RS,S,FD,FND}, m::AtmosModel,
                                  stateP::Vars, diffP::Vars, auxP::Vars, nM,
                                  stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) where {Name,I,RS,S,FD,FND}
  # TODO: Unroll loop
  for f in fieldnames(vars_state(c, eltype(stateP)))
    localstateP = getproperty(stateP, propname(Name))
    localstateM = getproperty(stateM, propname(Name))
    setproperty!(localstateP, f, getproperty(localstateM,f))
  end
end

"""
    InitStateBC <: BoundaryCondition

Set the value at the boundary to match the `init_state!` function.
This is mainly useful for cases where the problem has an explicit solution.
"""
function atmos_boundarycondition!(c::Component{Name,I,InitStateBC,RS,S,FD,FND}, m::AtmosModel,
                                  stateP::Vars, diffP::Vars, auxP::Vars, nM,
                                  stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) where {Name,I,RS,S,FD,FND}
  atmos_init_state!(c, stateP, auxP, auxP.coord, t)
  m.init_state(stateP, auxP, auxP.coord, t)
end
