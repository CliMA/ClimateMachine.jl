using CLIMA.VariableTemplates

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient, vars_diffusive,
flux!, source!, wavespeed, boundarycondition!, gradvariables!, diffusive!,
init_aux!, init_state!, init_ode_param, init_ode_state


struct MMSModel{dim} <: BalanceLaw
end

vars_aux(::MMSModel,T) = NamedTuple{(:x,:y,:z),NTuple{3,T}}
vars_state(::MMSModel, T) = NamedTuple{(:ρ, :ρu, :ρv, :ρw, :ρe),NTuple{5,T}}
vars_gradient(::MMSModel, T) = NamedTuple{(:u, :v, :w),NTuple{3,T}}
vars_diffusive(::MMSModel, T) = NamedTuple{(:τ11, :τ22, :τ33, :τ12, :τ13, :τ23),NTuple{6,T}}

function flux!(::MMSModel, flux::Grad, state::Vars, diffusive::Vars, auxstate::Vars, t::Real)
  # preflux
  T = eltype(flux)
  γ = T(γ_exact)
  ρinv = 1 / state.ρ
  u, v, w = ρinv * state.ρu, ρinv * state.ρv, ρinv * state.ρw
  P = (γ-1)*(state.ρe - ρinv * (state.ρu^2 + state.ρv^2 + state.ρw^2) / 2)

  # invisc terms
  flux.ρ  = SVector(state.ρu          , state.ρv          , state.ρw)
  flux.ρu = SVector(u * state.ρu  + P , v * state.ρu      , w * state.ρu)
  flux.ρv = SVector(u * state.ρv      , v * state.ρv + P  , w * state.ρv)
  flux.ρw = SVector(u * state.ρw      , v * state.ρw      , w * state.ρw + P)
  flux.ρe = SVector(u * (state.ρe + P), v * (state.ρe + P), w * (state.ρe + P))

  # viscous terms
  flux.ρu -= SVector(diffusive.τ11, diffusive.τ12, diffusive.τ13)
  flux.ρv -= SVector(diffusive.τ12, diffusive.τ22, diffusive.τ23)
  flux.ρw -= SVector(diffusive.τ13, diffusive.τ23, diffusive.τ33)

  flux.ρe -= SVector(u * diffusive.τ11 + v * diffusive.τ12 + w * diffusive.τ13,
                     u * diffusive.τ12 + v * diffusive.τ22 + w * diffusive.τ23,
                     u * diffusive.τ13 + v * diffusive.τ23 + w * diffusive.τ33)
end

function gradvariables!(::MMSModel, transformstate::Vars, state::Vars, auxstate::Vars, t::Real)
  ρinv = 1 / state.ρ
  transformstate.u = ρinv * state.ρu
  transformstate.v = ρinv * state.ρv
  transformstate.w = ρinv * state.ρw
end

function diffusive!(::MMSModel, diffusive::Vars, ∇transform::Grad, state::Vars, auxstate::Vars, t::Real)
  T = eltype(diffusive)
  μ = T(μ_exact)
  
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

function source!(::MMSModel{dim}, source::Vars, state::Vars, aux::Vars, t::Real) where {dim}
  source.ρ  = Sρ_g(t, aux.x, aux.y, aux.z, Val(dim))
  source.ρu = SU_g(t, aux.x, aux.y, aux.z, Val(dim))
  source.ρv = SV_g(t, aux.x, aux.y, aux.z, Val(dim))
  source.ρw = SW_g(t, aux.x, aux.y, aux.z, Val(dim))
  source.ρe = SE_g(t, aux.x, aux.y, aux.z, Val(dim))
end

function wavespeed(::MMSModel, nM, state::Vars, aux::Vars, t::Real)
  T = eltype(state)
  γ = T(γ_exact)
  ρinv = 1 / state.ρ
  u, v, w = ρinv * state.ρu, ρinv * state.ρv, ρinv * state.ρw
  P = (γ-1)*(state.ρe - ρinv * (state.ρu^2 + state.ρv^2 + state.ρw^2) / 2)
  return abs(nM[1] * u + nM[2] * v + nM[3] * w) + sqrt(ρinv * γ * P)
end

function boundarycondition!(bl::MMSModel, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)
  init_state!(bl, stateP, auxP, (auxM.x, auxM.y, auxM.z), t)
end

function init_aux!(::MMSModel, aux::Vars, (x,y,z))
  aux.x = x
  aux.y = y
  aux.z = z


end

function init_state!(bl::MMSModel{dim}, state::Vars, aux::Vars, (x,y,z), t) where {dim}
  state.ρ = ρ_g(t, x, y, z, Val(dim))
  state.ρu = U_g(t, x, y, z, Val(dim))
  state.ρv = V_g(t, x, y, z, Val(dim))
  state.ρw = W_g(t, x, y, z, Val(dim))
  state.ρe = E_g(t, x, y, z, Val(dim))
end
