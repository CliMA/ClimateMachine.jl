#### Mixing Length Model
abstract type MixingLengthModel{T} end

using ..PlanetParameters
export ConstantMixingLength, DynamicMixingLength

vars_state(    ::MixingLengthModel, T) = @vars()
vars_gradient( ::MixingLengthModel, T) = @vars()
vars_diffusive(::MixingLengthModel, T) = @vars()
vars_aux(      ::MixingLengthModel, T, N) = @vars(l_mix::SVector{N,T})

function update_aux!(   edmf::EDMF{N}, m::MixingLengthModel, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end
function gradvariables!(edmf::EDMF{N}, m::MixingLengthModel, transform::Vars, state::Vars, aux::Vars, t::Real) where N; end

"""
    ConstantMixingLength{N, T} <: MixingLengthModel{N}

Constant mixing length model.
"""
struct ConstantMixingLength{T} <: MixingLengthModel{T}
  l_mix::T
end
ConstantMixingLength(::Type{T}) where T = ConstantMixingLength{T}(100)

function update_aux!(edmf::EDMF{N}, m::ConstantMixingLength, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N
  id = idomains(N)
  for i in (id.en, id.up...)
    aux.turbconv.l_mix[i] = m.l_mix
  end
end

"""
    DynamicMixingLength{T} <: MixingLengthModel{T}

Dynamic mixing length model.
"""
struct DynamicMixingLength{T} <: MixingLengthModel{T}
  Prandtl_neutral::T
  c_ε::T
  c_K::T
  ω_1::T
  ω_2::T
  c_1::Tuple{T, T}
  c_2::Tuple{T, T}
  denom_limiter::T
end

"""
    DynamicMixingLength(::Type{T}) where T

Default values for dynamic mixing length.
"""
function DynamicMixingLength(::Type{T}) where T
  Prandtl_neutral = T(0.74)
  c_ε = T(0.12)
  c_K = T(0.1)
  ω_1 = T(40/13)
  ω_2 = ω_1+1
  c_1 = NTuple{2, T}([-100.0, 0.2])
  c_2 = NTuple{2, T}([2.7, -1.0])
  denom_limiter =  T(1e-2)
  return DynamicMixingLength{T}(Prandtl_neutral, c_ε, c_K, ω_1, ω_2, c_1, c_2, denom_limiter)
end

vars_aux(m::DynamicMixingLength, T, N) = @vars(θ_virt::SVector{N,T},
                                               S_squared::SVector{N,T})
vars_gradient(m::DynamicMixingLength, T, N) = @vars(θ_virt::SVector{N,T})

function ϕ_m(m, ξ, obukhov_length)
  a_L, b_L = obukhov_length<0 ? m.c_1 : m.c_2
  return (1+a_L*ξ)^(-b_L)
end

function update_aux!(edmf::EDMF{N}, m::DynamicMixingLength, ∇transform::Grad, state::Vars, aux::Vars) where N
  id = idomains(N)
  L = Vector(undef, 3)
  # obukhov_length, ustar, windspeed = compute_surface_fluxes(m, state, aux)
  z = aux.coordinates.z

  ∇u = ∇transform.u
  aux.turbconv.mix_len.S_squared[id.en] = ∇u[1,3]^2 + ∇u[2,3]^2 + ∇u[3,3]^2

  buoyancy_freq = grav*∇transform.turbconv.mix_len.θ_virt[id.en]/aux.turbconv.mix_len.θ_virt[id.en]
  L[1] = sqrt(c_w*state.turbconv.tke.tke[id.en])/buoyancy_freq
  ξ = z/obukhov_length
  κ_star = ustar/sqrt(state.turbconv.tke.tke[id.en])
  L[2] = k_Karman*z/(m.c_K*κ_star*ϕ_m(m, ξ, obukhov_length))
  R_g = ∇transform.turbconv.buoyancy.buoyancy[id.en]/aux.turbconv.mix_len.S_squared[id.en]
  Pr_z = obukhov_length <0 ? m.Prandtl_neutral : m.Prandtl_neutral*(1+m.ω_2*R_g - sqrt(-4*R_g+(1+m.ω_2*R_g)^2))/(2*R_g)
  discriminant = aux.turbconv.mix_len.S_squared[id.en] - ∇transform.turbconv.buoyancy.buoyancy[id.en]/Pr_z
  L[3] = sqrt(m.c_ε/m.c_K)*sqrt(state.turbconv.tke.tke[id.en])*1/sqrt(max(discriminant, m.denom_limiter))
  l_mix = sum([L[j]*exp(-L[j]) for j in 1:3])/sum([exp(-L[j]) for j in 1:3])

  aux.turbconv.l_mix[id.en] = l_mix
end
