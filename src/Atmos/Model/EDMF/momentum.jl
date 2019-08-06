#### Momentum Model
abstract type AbstractMomentumModel{T} end

struct MomentumModel{T} <: AbstractMomentumModel{T} end
MomentumModel(::Type{T}) where T = MomentumModel{T}()

export MomentumModel

vars_state(    ::AbstractMomentumModel, T, N) = @vars(ρu::SMatrix{3,N+1,T})
vars_gradient( ::AbstractMomentumModel, T, N) = @vars(ρu::SMatrix{3,N+1,T})
vars_diffusive(::AbstractMomentumModel, T, N) = @vars()
vars_aux(      ::AbstractMomentumModel, T, N) = @vars(ρu::SMatrix{3,N+1,T})

function gradvariables!(    edmf::EDMF{N}, m::AbstractMomentumModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N; end
function update_aux!(       edmf::EDMF{N}, m::AbstractMomentumModel, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_diffusive!(   edmf::EDMF{N}, m::AbstractMomentumModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_nondiffusive!(edmf::EDMF{N}, m::AbstractMomentumModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_advective!(   edmf::EDMF{N}, m::AbstractMomentumModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function source!(           edmf::EDMF{N}, m::AbstractMomentumModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function boundarycondition!(edmf::EDMF{N}, m::AbstractMomentumModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end

function diagnose_env!(edmf::EDMF{N}, m::MomentumModel, state::Vars, aux::Vars) where N
  id = idomains(N)
  state.turbconv.momentum.ρu[id.en] = state.ρu .- sum([state.turbconv.momentum.ρu[i] for i in id.up])
end

function flux_advective!(edmf::EDMF{N}, m::MomentumModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
    ρinv = 1/state.ρ
    flux.turbconv.momentum.ρu[i] = ρinv*state.turbconv.momentum.ρu[i] .* state.turbconv.momentum.ρu[i]'
  end
end

function flux_pressure!(edmf::EDMF{N}, m::MomentumModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
    ρinv = 1/state.ρ
    p = pressure(edmf, state, aux, i)
    flux.turbconv.momentum.ρu[i] += p*I
  end
end

function flux_nondiffusive!(edmf::EDMF{N}, m::MomentumModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
  end
end

function flux_diffusive!(edmf::EDMF{N}, m::MomentumModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
  end
end

function gradvariables!(edmf::EDMF{N}, m::MomentumModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
  end
end
