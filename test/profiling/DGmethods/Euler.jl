using CUDAnative
using CuArrays
CuArrays.allowscalar(false)
try
  isdefined(NumericalFluxes, :rusanov!)
catch
  include("../../src/DGmethods/NumericalFluxes.jl")
end

include("DGBalanceLawProfileHarness.jl")

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const statenames = ("ρ", "U", "V", "W", "E")
const γ_exact = 7 // 5

# preflux computation
@inline function preflux(Q, _...)
  γ::eltype(Q) = γ_exact
  @inbounds ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
  ρinv = 1 / ρ
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  ((γ-1)*(E - ρinv * (U^2 + V^2 + W^2) / 2), u, v, w, ρinv)
end

# max eigenvalue
@inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv)
  γ::eltype(Q) = γ_exact
  @inbounds abs(n[1] * u + n[2] * v + n[3] * w) + sqrt(ρinv * γ * P)
end

# physical flux function
eulerflux!(F, Q, QV, aux, t) =
eulerflux!(F, Q, QV, aux, t, preflux(Q)...)

@inline function eulerflux!(F, Q, QV, aux, t, P, u, v, w, ρinv)
  @inbounds begin
    ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]

    F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
    F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
    F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
    F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
    F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
  end
end

numerical_flux!(x...) = NumericalFluxes.rusanov!(x..., eulerflux!, wavespeed,
                                                 preflux)

let
  DFloat = Float64
  dim = 2
  nelem = 100
  N = 4
  cpu_dg = DGProfiler(Array, DFloat, dim, nelem, N, _nstate, eulerflux!,
                      numerical_flux!; stateoffset = ((_E, 20), (_ρ, 1)))
  gpu_dg = DGProfiler(CuArray, DFloat, dim, nelem, N, _nstate, eulerflux!,
                      numerical_flux!; stateoffset = ((_E, 20), (_ρ, 1)))
  volumerhs!(cpu_dg)
  volumerhs!(gpu_dg)
  facerhs!(cpu_dg)
  facerhs!(gpu_dg)
  gpu_rhs = Array(gpu_dg.rhs)
  cpu_rhs = Array(cpu_dg.rhs)
  @show gpu_rhs ≈ cpu_rhs
  nothing
end
