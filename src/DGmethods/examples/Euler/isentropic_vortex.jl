# Standard isentropic vortex test case.  For a more complete description of
# the setup see for Example 3 of:
#
# @article{ZHOU2003159,
#   author = {Y.C. Zhou and G.W. Wei},
#   title = {High resolution conjugate filters for the simulation of flows},
#   journal = {Journal of Computational Physics},
#   volume = {189},
#   number = {1},
#   pages = {159--179},
#   year = {2003},
#   doi = {10.1016/S0021-9991(03)00206-7},
#   url = {https://doi.org/10.1016/S0021-9991(03)00206-7},
# }
#
# This version runs the isentropic vortex within the entire moist thermodynamics
# framework

const halfperiod = 5

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)

const _nCstate = 5
const _ϕ, _ϕx, _ϕy, _ϕz, _T = 1:_nCstate

using CLIMA.PlanetParameters: cp_d, cv_d
using CLIMA.MoistThermodynamics: saturation_adjustment, air_pressure,
                                 phase_partitioning_eq, internal_energy

# physical flux function
function eulerflux!(F, Q, gradQstate, X, Cstate)
  DFloat = eltype(Q)

  q_t::DFloat = 0
  ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
  ϕ = Cstate[_ϕ]

  ρinv = 1 / ρ
  u, v, w = ρinv * U, ρinv * V, ρinv * W

  e_int = ρinv * E - (u^2 + v^2 + w^2) / 2 - ϕ

  T = saturation_adjustment(e_int, ρ, q_t)
  P = air_pressure(T, ρ, q_t)

  F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
  F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
  F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
  F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
  F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
end

# initial condition
function isentropicvortex!(Q, t, x, y, z)
  DFloat = eltype(Q)

  γ::DFloat    = cp_d / cv_d
  uinf::DFloat = 1
  vinf::DFloat = 1
  Tinf::DFloat = 1
  λ::DFloat    = 5

  xs = x - uinf*t
  ys = y - vinf*t

  # make the function periodic
  xtn = floor((xs+halfperiod)/(2halfperiod))
  ytn = floor((ys+halfperiod)/(2halfperiod))
  xp = xs - xtn*2*halfperiod
  yp = ys - ytn*2*halfperiod

  rsq = xp^2 + yp^2

  u = uinf - λ*(1//2)*exp(1-rsq)*yp/π
  v = vinf + λ*(1//2)*exp(1-rsq)*xp/π
  w = zero(DFloat)

  ρ = (Tinf - ((γ-1)*λ^2*exp(2*(1-rsq))/(γ*16*π*π)))^(1/(γ-1))
  p = ρ^γ
  U = ρ*u
  V = ρ*v
  W = ρ*w
  E = p/(γ-1) + ρ * internal_energy(0) + (1//2)*ρ*(u^2 + v^2 + w^2)

  Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
end
