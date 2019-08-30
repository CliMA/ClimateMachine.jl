#### Mass component in atmosphere model

export Mass

struct Mass <: ComponentNames end
propname(::Type{Mass}) = :mass

vars_state(c::Component{Mass}, T) = @vars(ρ::T)
# vars_gradient(c::Component{Mass}, T) = @vars()
# vars_diffusive(c::Component{Mass}, T) = @vars()
# vars_integrals(c::Component{Mass}, T) = @vars()
# vars_aux(c::Component{Mass}, T) = @vars()

# update_aux!(c::Component{Mass}, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing
# integrate_aux!(c::Component{Mass}, integ::Vars, state::Vars, aux::Vars) = nothing
# init_aux!(c::Component{Mass}, aux::Vars, geom::LocalGeometry) = nothing

"""
    flux!(c::Component{Mass}, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)

Computes flux `F` in:

```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
"""
function flux!(c::Component{Mass}, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  println("calling flux")
  flux.mass.ρ   = state.ρu
end

# """
#     source!(c::Component{Mass}, source::Vars, state::Vars, aux::Vars, t::Real)

# Computes flux `S(Y)` in:

# ```
# ∂Y
# -- = - ∇ • F + S(Y)
# ∂t
# ```
# """
# function source!(c::Component{Mass}, source::Vars, state::Vars, aux::Vars, t::Real)
# end

# function atmos_boundarycondition!(c::Component{Mass}, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)
#     stateP.mass.ρ = stateM.mass.ρ
# end

# """ When a custom initial state kernel is passed """
# function atmos_init_state!(c::Component{Mass,F,B,RS,S,FD,FND}, state::Vars, aux::Vars, coords, t) where {F<:Function,B,RS,S,FD,FND}
#   c.ics(state, aux, coords, t)
# end

""" Default initial state """
function atmos_init_state!(c::Component{Mass,Nothing,B,RS,S,FD,FND}, state::Vars, aux::Vars, coords, t) where {B,RS,S,FD,FND}
  state.mass.ρ = eltype(state)(0)
end
