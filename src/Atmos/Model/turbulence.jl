#### Turbulence closures
abstract type TurbulenceClosure
end

struct ConstantViscosityWithDivergence <: TurbulenceClosure
  ν::Float64 # turbulent kinematic viscosity
end
kinematic_viscosity_tensor(t::ConstantViscosityWithDivergence, normS) = t.ν
function momentum_flux_tensor(t::ConstantViscosityWithDivergence, ν, S)
  (-2*ν) .* S .+ (2ν/3)*tr(S)*I
end

struct SmagorinskyLilly <: TurbulenceClosure
  C_smag::Float64 # 0.15 
  Δ::Float64 # equivalent grid scale (can we get rid of this?)
end
kinematic_viscosity_tensor(t::SmagorinskyLilly, normS) = normS * T(m.C_smag * m.Δ)^2
momentum_flux_tensor(t::SmagorinskyLilly, ν, S) = (-2*ν) .* S 
