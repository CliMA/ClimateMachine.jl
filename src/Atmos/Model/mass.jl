#### Mass component in atmosphere model

export Mass

struct Mass <: ComponentNames end
propname(::Type{Mass}) = :mass

vars_state(c::Component{Mass}, T) = @vars(ρ::T)

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
  flux.mass.ρ   = state.ρu
end

