#### Buoyancy Model
abstract type AbstractBuoyancyModel{T} end
struct BuoyancyModel{T} <: AbstractBuoyancyModel{T} end
BuoyancyModel(::Type{T}) where T = BuoyancyModel{T}()

export BuoyancyModel

vars_state(    ::BuoyancyModel, T) = @vars()
vars_gradient( ::BuoyancyModel, T) = @vars()
vars_diffusive(::BuoyancyModel, T) = @vars()
vars_aux(      ::BuoyancyModel, T, N) = @vars(buoyancy::SVector{N,T})

function update_aux!(   edmf::EDMF{N}, m::BuoyancyModel, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function gradvariables!(edmf::EDMF{N}, m::BuoyancyModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N; end

function update_aux!(edmf::EDMF{N,A,M,TKE,SM,ML,ED,P,BuoyancyModel}, state::Vars, diffusive::Vars, ∇transform::Grad, aux::Vars, t::Real) where {N,A,M,TKE,SM,ML,ED,P}
  id = idomains(N)
  DT = eltype(state.ρ)

  @inbounds for i in (id.en, id.up...)
    ts = thermo_state(edmf, state, aux, i)
    R_m = gas_constant_air(ts)
    T = air_temperature(ts)
    α_i = R_m*T/aux.pressure
    α_0 = 1/state.ρ
    aux.turbconv.buoyancy[i] = grav*DT((big(α_i) - big(α_0))/α_0)
  end
  S = big(DT(0))
  @inbounds for i in (id.en, id.up...)
    a_i = (1/state.ρ)*state.turbconv.ρa[i]
    B_i = aux.turbconv.buoyancy[i]
    S += big(B_i)*big(a_i)
  end
  aux.buoyancy = DT(S)
  @inbounds for i in (id.en, id.up...)
    aux.turbconv.buoyancy[i] = DT(aux.turbconv.buoyancy[i] - S)
  end

  S = big(DT(0))
  @inbounds for i in (id.en, id.up...)
    a_i = (1/state.ρ)*state.turbconv.ρa[i]
    S += big(aux.turbconv.buoyancy[i])*big(a_i)
  end
  aux.buoyancy = DT(S)

end
