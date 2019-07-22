abstract type Force
end


struct Gravity <: Force
end

using CLIMA.PlanetParameters: grav

function source!(m::Gravity, source::Vars, state::Vars, aux::Vars, t::Real)
  T = eltype(state)
  source.ρu -= SVector(0,0,state.ρ * T(grav))
end
