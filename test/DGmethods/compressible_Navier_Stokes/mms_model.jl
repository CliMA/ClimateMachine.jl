import CLIMA.DGmethods: BalanceLaw, dimension, vars_aux, vars_state, vars_state_for_gradtransform, vars_gradtransform, vars_diffusive,
flux!, source!, wavespeed, boundarycondition!, gradtransform!, diffusive!,
init_aux!, init_state!, init_ode_param, init_ode_state


struct MMSModel{dim} <: BalanceLaw
end

dimension(::MMSModel{dim}) where {dim} = dim
vars_aux(::MMSModel) = (:x,:y,:z)
vars_state(::MMSModel) = (:ρ, :ρu, :ρv, :ρw, :ρe)
vars_state_for_gradtransform(::MMSModel) = (:ρ, :ρu, :ρv, :ρw)
vars_gradtransform(::MMSModel) = (:u, :v, :w)
vars_diffusive(::MMSModel) = (:τ11, :τ22, :τ33, :τ12, :τ13, :τ23)

function flux!(::MMSModel, flux::Grad, state::State, diffusive::State, auxstate::State, t::Real)
  # preflux
  γ = γ_exact  
  ρinv = 1 / state.ρ
  u, v, w = ρinv * state.ρu, ρinv * state.ρv, ρinv * state.ρw
  P = (γ-1)*(state.ρe - ρinv * (state.ρu^2 + state.ρv^2 + state.ρw^2) / 2)

  # invisc terms
  flux.ρ  = (state.ρu          , state.ρv          , state.ρw)
  flux.ρu = (u * state.ρu  + P , v * state.ρu      , w * state.ρu)
  flux.ρv = (u * state.ρv      , v * state.ρv + P  , w * state.ρv)
  flux.ρw = (u * state.ρw      , v * state.ρw      , w * state.ρw + P)
  flux.ρe = (u * (state.ρe + P), v * (state.ρe + P), w * (state.ρe + P))

  # viscous terms
  flux.ρu -= SVector(diffusive.τ11, diffusive.τ12, diffusive.τ13)
  flux.ρv -= SVector(diffusive.τ12, diffusive.τ22, diffusive.τ23)
  flux.ρw -= SVector(diffusive.τ13, diffusive.τ23, diffusive.τ33)

  flux.ρe -= SVector(u * diffusive.τ11 + v * diffusive.τ12 + w * diffusive.τ13,
                     u * diffusive.τ12 + v * diffusive.τ22 + w * diffusive.τ23,
                     u * diffusive.τ13 + v * diffusive.τ23 + w * diffusive.τ33)
end

function gradtransform!(::MMSModel, transformstate::State, state::State, auxstate::State, t::Real)
  ρinv = 1 / state.ρ
  transformstate.u = ρinv * state.ρu
  transformstate.v = ρinv * state.ρv
  transformstate.w = ρinv * state.ρw
end

function diffusive!(::MMSModel, diffusive::State, ∇transform::Grad, state::State, auxstate::State, t::Real)
  μ = μ_exact
  
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

function source!(::MMSModel{dim}, source::State, state::State, aux::State, t::Real) where {dim}
  source.ρ  = Sρ_g(t, aux.x, aux.y, aux.z, Val(dim))
  source.ρu = SU_g(t, aux.x, aux.y, aux.z, Val(dim))
  source.ρv = SV_g(t, aux.x, aux.y, aux.z, Val(dim))
  source.ρw = SW_g(t, aux.x, aux.y, aux.z, Val(dim))
  source.ρe = SE_g(t, aux.x, aux.y, aux.z, Val(dim))
end

function wavespeed(::MMSModel, nM, state::State, aux::State, t::Real)
  γ = γ_exact
  ρinv = 1 / state.ρ
  u, v, w = ρinv * state.ρu, ρinv * state.ρv, ρinv * state.ρw
  P = (γ-1)*(state.ρe - ρinv * (state.ρu^2 + state.ρv^2 + state.ρw^2) / 2)

  return abs(nM[1] * u + nM[2] * v + nM[3] * w) + sqrt(ρinv * γ * P)
end

function boundarycondition!(bl::MMSModel, stateP::State, diffP::State, auxP::State, nM, stateM::State, diffM::State, auxM::State, bctype, t)
  init_state!(bl, stateP, auxP, (auxM.x, auxM.y, auxM.z), t)
end

function init_aux!(::MMSModel, aux::State, (x,y,z))
  aux.x = x
  aux.y = y
  aux.z = z
end

function init_state!(bl::MMSModel{dim}, state::State, aux::State, (x,y,z), t) where {dim}
  state.ρ = ρ_g(t, x, y, z, Val(dim))
  state.ρu = U_g(t, x, y, z, Val(dim))
  state.ρv = V_g(t, x, y, z, Val(dim))
  state.ρw = W_g(t, x, y, z, Val(dim))
  state.ρe = E_g(t, x, y, z, Val(dim))
end
