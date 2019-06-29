#### Turbulence closures
abstract type TurbulenceClosure
end

struct ConstantViscosity <: TurbulenceClosure
  μ::Float64
end

vars_gradtransform(::ConstantViscosity) = (:u, :v, :w)
function gradtransform!(m::ConstantViscosity, transformstate::State, state::State, auxstate::State, t::Real)
    ρinv = 1 / state.ρ
    transformstate.u = ρinv * state.ρu
    transformstate.v = ρinv * state.ρv
    transformstate.w = ρinv * state.ρw
end
  
vars_diffusive(::ConstantViscosity) = (:τ11, :τ22, :τ33, :τ12, :τ13, :τ23)
function diffusive!(m::ConstantViscosity, diffusive::State, ∇transform::Grad, state::State, auxstate::State, t::Real)
  T = eltype(diffusive)
  μ = T(m.μ)
  
  dudx, dudy, dudz = ∇transform.u
  dvdx, dvdy, dvdz = ∇transform.v
  dwdx, dwdy, dwdz = ∇transform.w

  # strains
  ϵ11 = dudx
  ϵ22 = dvdy
  ϵ33 = dwdz
  ϵ12 = (dudy + dvdx) / 2
  ϵ13 = (dudz + dwdx) / 2
  ϵ23 = (dvdz + dwdy) / 2

  # deviatoric stresses
  diffusive.τ11 = 2μ * (ϵ11 - (ϵ11 + ϵ22 + ϵ33) / 3)
  diffusive.τ22 = 2μ * (ϵ22 - (ϵ11 + ϵ22 + ϵ33) / 3)
  diffusive.τ33 = 2μ * (ϵ33 - (ϵ11 + ϵ22 + ϵ33) / 3)
  diffusive.τ12 = 2μ * ϵ12
  diffusive.τ13 = 2μ * ϵ13
  diffusive.τ23 = 2μ * ϵ23
end

function flux!(m::ConstantViscosity, flux::Grad, state::State, diffusive::State, auxstate::State, t::Real)
    flux.ρu -= SVector(diffusive.τ11, diffusive.τ12, diffusive.τ13)
    flux.ρv -= SVector(diffusive.τ12, diffusive.τ22, diffusive.τ23)
    flux.ρw -= SVector(diffusive.τ13, diffusive.τ23, diffusive.τ33)

    # energy dissipation
    flux.ρe -= SVector(u * diffusive.τ11 + v * diffusive.τ12 + w * diffusive.τ13,
                       u * diffusive.τ12 + v * diffusive.τ22 + w * diffusive.τ23,
                       u * diffusive.τ13 + v * diffusive.τ23 + w * diffusive.τ33)
end
function preodefun!(m::ConstantViscosity, auxstate::State, state::State, t::Real)
end





struct SmagorinskyLilly <: TurbulenceClosure
  C_smag_Δ::Float64 # 0.15 * equivalent grid scale (can we get rid of this?)
end

vars_gradtransform(::SmagorinskyLilly) = (:u, :v, :w, :ρqt, :T)
function gradtransform!(m::SmagorinskyLilly, transformstate::State, state::State, auxstate::State, t::Real)
    ρinv = 1 / state.ρ
    transformstate.u = ρinv * state.ρu
    transformstate.v = ρinv * state.ρv
    transformstate.w = ρinv * state.ρw

    transformstate.ρqt = state.ρqt
    transformstate.T = aux.T
end

vars_diffusive(::SmagorinskyLilly) = (:τ11, :τ22, :τ33, :τ12, :τ13, :τ23,
                                      :α1, :α2, :α3,
                                      :β1, :β2, :β3)
function diffusive!(m::SmagorinskyLilly, diffusive::State, ∇transform::Grad, state::State, auxstate::State, t::Real)
  T = eltype(diffusive)
  
  dudx, dudy, dudz = ∇transform.u
  dvdx, dvdy, dvdz = ∇transform.v
  dwdx, dwdy, dwdz = ∇transform.w
  
  # strains
  S11 = dudx
  S22 = dvdy
  S33 = dwdz
  S12 = (dudy + dvdx) / 2
  S13 = (dudz + dwdx) / 2
  S23 = (dvdz + dwdy) / 2

  SijSij = S11^2 + S22^2 + S33^2 + 2S12^2 + 2S13^2 + 2S23^2

  # Multiply stress tensor by viscosity coefficient:
  ν_e = sqrt(2 * SijSij) * T(m.C_smag_Δ)^2

  # deviatoric stresses
  diffusive.τ11 = 2ν_e * (S11 - (S11 + S22 + S33) / 3)
  diffusive.τ22 = 2ν_e * (S22 - (S11 + S22 + S33) / 3)
  diffusive.τ33 = 2ν_e * (S33 - (S11 + S22 + S33) / 3)
  diffusive.τ12 = 2ν_e * S12
  diffusive.τ13 = 2ν_e * S13
  diffusive.τ23 = 2ν_e * S23

  # TODO: better names/symbols?
  diffusive.k_conductivity_1 = T(cp_d / Prandtl_t) * ν_e * ∇transform.T[1]
  diffusive.k_conductivity_1 = T(cp_d / Prandtl_t) * ν_e * ∇transform.T[2]
  diffusive.k_conductivity_1 = T(cp_d / Prandtl_t) * ν_e * ∇transform.T[3]
  
  # TODO: we need a way to do this for all 
  # Viscous contributions to mass flux terms
  diffusive.β1 = (ν_e / T(Prandtl_t)) * ∇transform.ρqt[1]
  diffusive.β2 = (ν_e / T(Prandtl_t)) * ∇transform.ρqt[2]
  diffusive.β3 = (ν_e / T(Prandtl_t)) * ∇transform.ρqt[3]
end

function flux!(m::SmagorinskyLilly, flux::Grad, state::State, diffusive::State, auxstate::State, t::Real)
    T = eltype(flux)

    D_e = diffusive.ν_e / T(Prandtl_t)

    flux.ρu -= SVector(diffusive.τ11, diffusive.τ12, diffusive.τ13)
    flux.ρv -= SVector(diffusive.τ12, diffusive.τ22, diffusive.τ23)
    flux.ρw -= SVector(diffusive.τ13, diffusive.τ23, diffusive.τ33)

    # energy dissipation
    flux.ρe -= SVector(u * diffusive.τ11 + v * diffusive.τ12 + w * diffusive.τ13 + diffusive.α1,
                       u * diffusive.τ12 + v * diffusive.τ22 + w * diffusive.τ23 + diffusive.α2,
                       u * diffusive.τ13 + v * diffusive.τ23 + w * diffusive.τ33 + diffusive.α3)

    # Viscous contributions to mass flux terms
    # should this apply to any tracer variable?
    # if so, we need a better way to handle this
    flux.ρqt -= SVector(diffusive.β1, diffusive.β2, diffusive.β3)
end
function preodefun!(m::SmagorinskyLilly, auxstate::State, state::State, t::Real)
end