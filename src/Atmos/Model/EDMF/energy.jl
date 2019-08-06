#### Energy Model
abstract type AbstractEnergyModel{T} end

struct EnergyModel{T} <: AbstractEnergyModel{T} end
EnergyModel(::Type{T}) where T = EnergyModel{T}()

export EnergyModel

vars_state(    ::AbstractEnergyModel, T, N) = @vars()
vars_gradient( ::AbstractEnergyModel, T, N) = @vars()
vars_diffusive(::AbstractEnergyModel, T, N) = @vars()
vars_aux(      ::AbstractEnergyModel, T, N) = @vars(ρe_int::SVector{N,T})

function update_aux!(       edmf::EDMF{N}, m::AbstractEnergyModel, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function gradvariables!(    edmf::EDMF{N}, m::AbstractEnergyModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N; end
function flux_diffusive!(   edmf::EDMF{N}, m::AbstractEnergyModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_nondiffusive!(edmf::EDMF{N}, m::AbstractEnergyModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_advective!(   edmf::EDMF{N}, m::AbstractEnergyModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function source!(           edmf::EDMF{N}, m::AbstractEnergyModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function boundarycondition!(edmf::EDMF{N}, m::AbstractEnergyModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end

function flux_advective!(edmf::EDMF{N}, m::EnergyModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
    ρinv = 1/state.ρ
    flux.turbconv.energy.ρe[i] = state.turbconv.momentum.ρu[i] * ρinv*state.turbconv.energy.ρe[i]
  end
end

function flux_pressure!(edmf::EDMF{N}, m::EnergyModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
    ρinv = 1/state.ρ
    p = pressure(edmf, state, aux, i)
    flux.turbconv.energy.ρe[i] += ρinv*state.turbconv.momentum.ρu*p
  end
end

function diagnose_env!(edmf::EDMF{N}, m::EnergyModel, state::Vars, aux::Vars) where N
  id = idomains(N)
  state.turbconv.energy.ρe[id.en] = state.ρe  - sum([state.turbconv.energy.ρe[i] for i in id.up])
end
