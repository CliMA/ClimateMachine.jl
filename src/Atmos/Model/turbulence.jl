#### Turbulence closures
abstract type TurbulenceClosure
end

vars_state(::TurbulenceClosure, T) = Tuple{}
vars_gradient(::TurbulenceClosure, T) = Tuple{}
vars_diffusive(::TurbulenceClosure, T) = Tuple{}
vars_aux(::TurbulenceClosure, T) = Tuple{}

"""
    ConstantViscosityWithDivergence <: TurbulenceClosure

Turbulence with constant dynamic viscosity (`ρν`). Divergence terms are included in the momentum flux tensor.
"""
struct ConstantViscosityWithDivergence <: TurbulenceClosure
  ρν::Float64
end
dynamic_viscosity_tensor(m::ConstantViscosityWithDivergence, S, state::Vars, aux::Vars, t::Real) = m.ρν
function scaled_momentum_flux_tensor(m::ConstantViscosityWithDivergence, ρν, S)
  trS = S[1] + S[2] + S[3]  
  I = SVector(1,1,1,0,0,0)
  return (-2*ρν) .* S .+ (2*ρν/3)*trS .* I
end

"""
    SmagorinskyLilly <: TurbulenceClosure

"""
struct SmagorinskyLilly <: TurbulenceClosure
  C_smag::Float64 # 0.15 
  Δ::Float64 # equivalent grid scale (can we get rid of this?)
end
function dynamic_viscosity_tensor(m::SmagorinskyLilly, S, state::Vars, aux::Vars, t::Real) 
  # strain rate tensor norm
  # NOTE: factor of 2 scaling
  # normS = norm(2S)
  normS = sqrt(2*(S[1]^2 + S[2]^2 + S[3]^2 + 2*(S[4]^2 + S[5]^2 + S[6]^2))) 
  return state.ρ * normS * T(m.C_smag * m.Δ)^2
end
function scaled_momentum_flux_tensor(m::SmagorinskyLilly, ρν, S)
  (-2*ρν) .* S 
end
