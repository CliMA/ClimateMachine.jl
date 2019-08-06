#### Area fraction Model
abstract type AreaFractionModel{T} end

export ConstantAreaFrac, PrognosticAreaFrac

vars_state(    ::AreaFractionModel, T, N) = @vars()
vars_gradient( ::AreaFractionModel, T, N) = @vars()
vars_diffusive(::AreaFractionModel, T, N) = @vars()
vars_aux(      ::AreaFractionModel, T, N) = @vars()

function update_aux!(       edmf::EDMF{N}, m::AreaFractionModel, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function gradvariables!(    edmf::EDMF{N}, m::AreaFractionModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N; end
function flux_diffusive!(   edmf::EDMF{N}, m::AreaFractionModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_nondiffusive!(edmf::EDMF{N}, m::AreaFractionModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_advective!(   edmf::EDMF{N}, m::AreaFractionModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function source!(           edmf::EDMF{N}, m::AreaFractionModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function boundarycondition!(edmf::EDMF{N}, m::AreaFractionModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end

"""
    ConstantAreaFrac{T} <: AreaFractionModel{T}

Constant area fraction model. Area fraction is prescribed
using this model.
"""
struct ConstantAreaFrac{N, T} <: AreaFractionModel{T}
  ρa::NTuple{N, T}
end

"""
    PrognosticAreaFrac{T} <: AreaFractionModel{T}

Prognostic area fraction model. Area fraction is solved for
using this model.
"""
struct PrognosticAreaFrac{T} <: AreaFractionModel{T} end

vars_state(m::PrognosticAreaFrac, T, N) = @vars(ρa::SVector{N,T})

function flux_advective!(edmf::EDMF{N}, m::PrognosticAreaFrac, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
    ρinv = 1/state.turbconv.area_frac.ρ[i]
    flux.turbconv.area_frac.ρa[i] = ρinv*state.turbconv.area_frac.ρa[i]*state.turbconv.area_frac.ρu[i]
  end
end

function source!(edmf::EDMF{N}, m::PrognosticAreaFrac, source::Vars, state::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  DT = eltype(state)
  @inbounds for i in id.up
    source.turbconv.area_frac.ρa[i] = aux.turbconv.entr_detr.εδ_ρa[i]
  end
  source.turbconv.area_frac.ρa[id.en] = sum([DT(0) - aux.turbconv.entr_detr.εδ_ρa[i] for i in id.up])
end

function diagnose_env!(edmf::EDMF{N}, m::PrognosticAreaFrac, state::Vars, aux::Vars) where N
  id = idomains(N)
  state.turbconv.area_frac.ρa[id.en] = state.ρa  - sum([state.turbconv.area_frac.ρa[i] for i in id.up])
end
