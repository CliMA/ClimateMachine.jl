#### Turbulence closures
abstract type TurbulenceClosure
end

struct ConstantViscosity <: TurbulenceClosure
  ν::Float64
end
kinematic_viscosity_tensor(t::ConstantViscosity, normS) = t.ν

struct SmagorinskyLilly <: TurbulenceClosure
  C_smag::Float64 # 0.15 
  Δ::Float64 # equivalent grid scale (can we get rid of this?)
end
kinematic_viscosity_tensor(t::SmagorinskyLilly, normS) = normS * T(m.C_smag * m.Δ)^2