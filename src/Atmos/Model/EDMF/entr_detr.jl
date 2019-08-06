#### Entrainment-Detrainment Model
abstract type EntrDetrModel{T} end

export ConstantEntrDetr, BOverW2

vars_state(    ::EntrDetrModel, T) = @vars()
vars_gradient( ::EntrDetrModel, T) = @vars()
vars_diffusive(::EntrDetrModel, T) = @vars()
vars_aux(      ::EntrDetrModel, T) = @vars()

function update_aux!(       edmf::EDMF{N}, m::EntrDetrModel, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function gradvariables!(    edmf::EDMF{N}, m::EntrDetrModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N; end
function flux_diffusive!(   edmf::EDMF{N}, m::EntrDetrModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_nondiffusive!(edmf::EDMF{N}, m::EntrDetrModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_advective!(   edmf::EDMF{N}, m::EntrDetrModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function source!(           edmf::EDMF{N}, m::EntrDetrModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end

"""
    ConstantEntrDetr{T} <: EntrDetrModel{T}

Constant entrainment-detrainment model.
"""
struct ConstantEntrDetr{T} <: EntrDetrModel{T}
  c_ε::T
  c_δ_0::T
  δ_B::T
  denom_limiter::T
end

"""
    ConstantEntrDetr(::Type{T}) where T

Default values for constant entrainment-detrainment model.
"""
function ConstantEntrDetr(::Type{T}) where T
  c_ε = T(0.1)
  c_δ_0 = T(0.1)
  δ_B = T(0.004)
  denom_limiter = T(1e-2)
  return ConstantEntrDetr{T}(c_ε, c_δ_0, δ_B, denom_limiter)
end

function vars_aux(m::ConstantEntrDetr, T, N)
  @vars begin
    δ_model::SVector{N,T}
    εδ_ρa::SVector{N,T}
    εδ_ρu::SVector{N,T}
    εδ_ρe::SVector{N,T}
    εδ_q_tot::SVector{N,T}
    εδ_tke::SVector{N,T}
  end
end

@inline function safe_divide(m::ConstantEntrDetr, a, b)
  return a/(max(b, m.denom_limiter))
end

function update_aux!(edmf::EDMF{N,PrognosticAreaFrac,M,TKE,SM,ML,ED,P,B}, m::ConstantEntrDetr{T}, state::Vars, diffusive::Vars, ∇transform::Grad, aux::Vars, t::Real) where {N,M,TKE,SM,ML,ED,P,B,T}

  id = idomains(N)

  @inbounds for ϕ in (:ρa, :ρu, :ρe, :q_tot, :tke, :cv_q_tot_q_tot, :cv_e_int_e_int, :cv_e_int_q_tot)
    @inbounds for i in (id.en, id.up...)
      ts = thermo_state(edmf, state, aux, i)
      q_pt = PhasePartition(ts)

      ρa_i = getfield(state.turbconv, :ρa)[i]
      ϕ_i = ϕ==:ρu ? getfield(state.turbconv, ϕ)[i][3] : getfield(state.turbconv, ϕ)[i]
      w_i = getfield(state.turbconv, :ρu)[i][3]
      ϕ_env = getfield(state.turbconv, ϕ)[id.en]
      ϕ_env = getfield(state.turbconv, ϕ)[id.en]
      w_env = getfield(state.turbconv, :ρu)[id.en][3]
      B_i = aux.turbconv.buoyancy[i]

      ε_i = m.c_ε
      δ_i = m.c_δ_0
      aux.turbconv.δ_model[i] = δ_i

      δ_i_ϕ_i = δ_i*ϕ_i
      ε_i_ϕ_e = ε_i*ϕ_env
      ρaw_k = ρa_i * abs(w_i)
      if ϕ==:ρa
        aux.turbconv.εδ_ρa = ρaw_k * (-δ_i + ε_i)
      elseif ϕ==:ρu
        aux.turbconv.εδ_ρu = ρaw_k * (-δ_i_ϕ_i + ε_i_ϕ_e)
      elseif ϕ==:ρe
        aux.turbconv.εδ_ρe = ρaw_k * (-δ_i_ϕ_i + ε_i_ϕ_e)
      elseif ϕ==:q_tot
        aux.turbconv.εδ_q_tot = ρaw_k * (-δ_i_ϕ_i + ε_i_ϕ_e)
      elseif ϕ==:tke
        aux.turbconv.εδ_tke = ρaw_k * (-δ_i_ϕ_i + ε_i*(ϕ_env + 1/2*(w_env-w_i)^2) )
      elseif ϕ==:cv_q_tot_q_tot
        # TODO: Add computations for cv_q_tot_q_tot
      elseif ϕ==:cv_e_int_e_int
        # TODO: Add computations for cv_e_int_e_int
      elseif ϕ==:cv_e_int_q_tot
        # TODO: Add computations for cv_e_int_q_tot
      else
        error("Bad field in "*@__FILE__)
      end
    end
  end
end

"""
    BOverW2{N, T} <: EntrDetrModel{N}

Yair's entrainment-detrainment model.
"""
struct BOverW2{T} <: EntrDetrModel{T}
  c_ε::T
  δ_B::T
  c_δ_0::T
  denom_limiter::T
end

"""
    BOverW2(::Type{T}) where T

Default values for Yair's entrainment-detrainment model.
"""
function BOverW2(::Type{T}) where T
  c_ε = T(0.12)
  δ_B = T(0.004)
  c_δ_0 = T(0.12)
  denom_limiter = T(1e-2)
  return BOverW2{T}(c_ε, c_δ_0, δ_B, denom_limiter)
end

function vars_aux(m::BOverW2, T, N)
  @vars begin
    δ_model::SVector{N,T}
    εδ_ρa::SVector{N,T}
    εδ_ρu::SVector{N,T}
    εδ_ρe::SVector{N,T}
    εδ_q_tot::SVector{N,T}
    εδ_tke::SVector{N,T}
  end
end

@inline function safe_divide(m::BOverW2, a, b)
  return a/(max(b, m.denom_limiter))
end

function update_aux!(edmf::EDMF{N,PrognosticAreaFrac,M,TKE,SM,ML,ED,P,B}, m::BOverW2{T}, state::Vars, diffusive::Vars, ∇transform::Grad, aux::Vars, t::Real) where {N,M,TKE,SM,ML,ED,P,B,T}

  id = idomains(N)

  for ϕ in (:ρa, :ρu, :ρe, :q_tot, :tke, :cv_q_tot_q_tot, :cv_e_int_e_int, :cv_e_int_q_tot)
    for i in (id.en, id.up...)
      ts = thermo_state(edmf, state, aux, i)
      q_pt = PhasePartition(ts)

      ρa_i = getfield(state.turbconv, :ρa)[i]
      ϕ_i = ϕ==:ρu ? getfield(state.turbconv, ϕ)[i][3] : getfield(state.turbconv, ϕ)[i]
      w_i = getfield(state.turbconv, :ρu)[i][3]
      ϕ_env = getfield(state.turbconv, ϕ)[id.en]
      ϕ_env = getfield(state.turbconv, ϕ)[id.en]
      w_env = getfield(state.turbconv, :ρu)[id.en][3]
      B_i = aux.turbconv.buoyancy[i]

      ε_i = m.c_ε*safe_divide(m, max(B_i, 0), w_i^2)
      δ_i = m.c_δ_0*safe_divide(m, abs(min(B_i, 0)), w_i^2) + m.δ_B*T(q_pt.liq > T(0))
      aux.turbconv.δ_model[i] = δ_i

      δ_i_ϕ_i = δ_i*ϕ_i
      ε_i_ϕ_e = ε_i*ϕ_env
      ρaw_k = ρa_i * abs(w_i)
      if ϕ==:ρa
        aux.turbconv.εδ_ρa = ρaw_k * (-δ_i + ε_i)
      elseif ϕ==:ρu
        aux.turbconv.εδ_ρu = ρaw_k * (-δ_i_ϕ_i + ε_i_ϕ_e)
      elseif ϕ==:ρe
        aux.turbconv.εδ_ρe = ρaw_k * (-δ_i_ϕ_i + ε_i_ϕ_e)
      elseif ϕ==:q_tot
        aux.turbconv.εδ_q_tot = ρaw_k * (-δ_i_ϕ_i + ε_i_ϕ_e)
      elseif ϕ==:tke
        aux.turbconv.εδ_tke = ρaw_k * (-δ_i_ϕ_i + ε_i*(ϕ_env + 1/2*(w_env-w_i)^2) )
      elseif ϕ==:cv_q_tot_q_tot
        # TODO: Add computations for cv_q_tot_q_tot
      elseif ϕ==:cv_e_int_e_int
        # TODO: Add computations for cv_e_int_e_int
      elseif ϕ==:cv_e_int_q_tot
        # TODO: Add computations for cv_e_int_q_tot
      else
        error("Bad field in "*@__FILE__)
      end
    end
  end
end

