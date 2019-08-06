#### Turbulent Kinetic Energy Model
abstract type AbstractTKEModel{T} end
struct TKEModel{T} <: AbstractTKEModel{T} end
TKEModel(::Type{T}) where T = TKEModel{T}()

export TKEModel

vars_state(    ::AbstractTKEModel, T, N) = @vars()
vars_gradient( ::AbstractTKEModel, T, N) = @vars()
vars_diffusive(::AbstractTKEModel, T, N) = @vars()
vars_aux(      ::AbstractTKEModel, T, N) = @vars(tke::SVector{N,T})
function update_aux!(       edmf::EDMF{N}, m::AbstractTKEModel, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function gradvariables!(    edmf::EDMF{N}, m::AbstractTKEModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N; end
function flux_diffusive!(   edmf::EDMF{N}, m::AbstractTKEModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_nondiffusive!(edmf::EDMF{N}, m::AbstractTKEModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function flux_advective!(   edmf::EDMF{N}, m::AbstractTKEModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function source!(           edmf::EDMF{N}, m::AbstractTKEModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function boundarycondition!(edmf::EDMF{N}, m::AbstractTKEModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end

function diagnose_env!(edmf::EDMF{N}, m::TKEModel, state::Vars, aux::Vars) where N
  id = idomains(N)
  state.turbconv.tke[id.en] = state.tke  - sum([state.turbconv.tke[i] for i in id.up])
end

function flux_advective!(edmf::EDMF{N}, m::TKEModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
  end
end

function flux_pressure!(edmf::EDMF{N}, m::TKEModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
  end
end

function flux_nondiffusive!(edmf::EDMF{N}, m::TKEModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
  end
end

function flux_diffusive!(edmf::EDMF{N}, m::TKEModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
  end
end

function gradvariables!(edmf::EDMF{N}, m::TKEModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  @inbounds for i in id.up
  end
end
