
import CLIMA.DGmethods: BalanceLaw, dimension, vars_aux, vars_state, num_state_for_gradtransform,  vars_gradtransform, vars_diffusive,
  flux!, source!, wavespeed, boundarycondition!, gradtransform!, diffusive!,
  init_aux!, init_state!, preodefun!


  function state(m::AtmosModel, T)
    NamedTuple{(:ρ, :ρu, :ρe, :turbulence, :moisture, :radiation), 
    Tuple{T, SVector{3,T}, T, state(m.turbulence,T), state(m.moisture,T), state(m.radiation, T)}}
  end
  
  state(m::ConstViscosity, T) = NamedTuple{}
  state(m::EquilMoist, T) = NamedTuple{(:ρqt,), Tuple{T}}
  state(m::NoRadiation, T) = NamedTuple{}
  
  model = AtmosModel(ConstViscosity(), EquilMoist(), NoRadiation())
  
  st = state(model, Float64)
  
  v = Vars{st}(zeros(MVector{6,Float64}))
  g = Grad{st}(zeros(MMatrix{3,6,Float64}))

  
  
struct AtmosModel{T,M,R} <: BalanceLaw
  turbulence::T
  moisture::M
  radiation::R
end

using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0

const γ_exact = 7 // 5

vars_state(m::AtmosModel) = (:ρ, :ρu, :ρv, :ρw, :ρe, vars_state(m.moisture)...)
num_state_for_gradtransform(::AtmosModel) = length(vars_state(m)) #TODO: need to handle this better

vars_gradtransform(::AtmosModel) = (vars_gradtransform(m.turbulence)...,)
vars_diffusive(::AtmosModel) = (vars_diffusive(m.turbulence)..., 
                                vars_diffusive(m.moisture)..., 
                                vars_diffusive(m.radiation)...)

vars_aux(::AtmosModel) = (vars_aux(m.turbulence)...,
                          vars_aux(m.moisture)...,
                          vars_aux(m.radiation)...)

# Navier-Stokes flux terms
function flux!(m::AtmosModel, flux::Grad, state::State, diffusive::State, auxstate::State, t::Real)
  # preflux
  ρinv = 1 / state.ρ
  u, v, w = ρinv * state.ρu, ρinv * state.ρv, ρinv * state.ρw

  P = pressure(m.moisture, state, diffusive, auxstate, t)

  # invisc terms
  flux.ρ  = SVector(state.ρu          , state.ρv          , state.ρw)
  flux.ρu = SVector(u * state.ρu  + P , v * state.ρu      , w * state.ρu)
  flux.ρv = SVector(u * state.ρv      , v * state.ρv + P  , w * state.ρv)
  flux.ρw = SVector(u * state.ρw      , v * state.ρw      , w * state.ρw + P)
  flux.ρe = SVector(u * (state.ρe + P), v * (state.ρe + P), w * (state.ρe + P))

  # viscous terms
  flux!(m.turbulence, flux, state, diffusive, auxstate, t)

  # flux for moisture components
  flux!(m.moisture, flux, state, diffusive, auxstate, t)

  # flux for radiation components
  flux!(m.radiation, flux, state, diffusive, auxstate, t)
end


function source!(::AtmosModel, source::State, state::State, aux::State, t::Real)
  T = eltype(state)
  source.ρw -= state.ρ * T(grav)
end

function wavespeed(::AtmosModel, nM, state::State, aux::State, t::Real)
  ρinv = 1 / state.ρ
  u, v, w = ρinv * state.ρu, ρinv * state.ρv, ρinv * state.ρw 
  P = pressure(m.moisture, state, diffusive, auxstate, t)
  return abs(nM[1] * u + nM[2] * v + nM[3] * w) + sqrt(ρinv * γ * P)
end


function gradtransform!(::AtmosModel, transformstate::State, state::State, auxstate::State, t::Real)
  ρinv = 1 / state.ρ
  transformstate.u = ρinv * state.ρu
  transformstate.v = ρinv * state.ρv
  transformstate.w = ρinv * state.ρw
end

function diffusive!(m::AtmosModel, diffusive::State, ∇transform::Grad, state::State, auxstate::State, t::Real)
  diffusive!(m.turbulence, diffusive, ∇transform, state, auxstate, t)
end



function preodefun!(m::AtmosModel, auxstate::State, state::State, t::Real)
  # map
  preodefun_elem!(m.moisture, auxstate, state, t)
  preodefun!(m.radiation, auxstate, state, t)
end


# utility function for specifying the flux terms on tracers
@inline function flux_tracer!(sym::Symbol, flux::Grad, state::State)
  # preflux
  ρinv = 1 / state.ρ
  u, v, w = ρinv * state.ρu, ρinv * state.ρv, ρinv * state.ρw
 
  # invisc terms
  setfield!(flux, sym, SVector(u * getfield(state, sym), 
                               v * getfield(state, sym) , 
                               w * getfield(state, sym)))
end

MMS = AtmosModel(ConstantViscocity(?), DryModel(), NoRadiation())

DYCOMS = AtmosModel(SmagorinskyLilly(coef), 
    EquilMoist(), StevensRadiation())


function boundarycondition!(bl::MMSModel, stateP::State, diffP::State, auxP::State, nM, stateM::State, diffM::State, auxM::State, bctype, t)
  init_state!(bl, stateP, auxP, (auxM.x, auxM.y, auxM.z), t)
end

function init_auxvars!(::MMSModel, aux::State, (x,y,z))
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
