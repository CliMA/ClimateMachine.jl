"""
    Microphysics

Microphysics parameterization functions (for example condensation, 
evaporation, autoconversion rates for cloud water and rain water)
TODO: Snow, ice, and 2-moment formulation
"""
module Microphysics

using ..MoistThermodynamics
using ...PlanetParameters

# rates of conversion between microphysics categories
export qv2qli, ql2qr, qr2qv

# rain fall speed
export terminal_velocity
 
# conversion from specific humidity to mixing ratio (TODO - temporary)
export q2r

"""
    qv2qli(qt, ql, qi, T, ρ[, timescale=1])

Return the ql and qi source terms due to condesation/evaporation (for ql)
and sublimation/resublimation (for qi)
The source terms are obtained assuming a relaxation to equilibrum with a 
constant timescale. The timescale is an optional parameter.
"""
function qv2qli(qt, ql, qi, T, ρ, timescale=1)
  dqldt = - (ql - liquid_fraction(T) * saturation_excess(T, ρ, qt, ql, qi)) /
          timescale
  dqidt = - (qi - (1-liquid_fracion(T)) * saturation_excess(T, ρ, qt, ql, qi)) /
          timescale
  return dqldt, dqidt
end

"""
    ql2qr(ql[, timescale=1, ql_0=1e-4])

Return the qr source term due to collisions between cloud droplets
(autoconversion) parametrized following Kessler (TODO - add reference).
Contrary to the Kessler paper the process is not assumed to be instanteneous.
The assumed timescale and autoconverion threshold ql_0 are optional parameters.
"""
function ql2qr(ql, timescale=1, ql_0=1e-4)

  dqrdt = max(0, ql - ql_0) / timescale
  return dqrdt
end

"""
    q2r(q_, qt)

Convert specific humidity to mixing ratio
"""
function q2r(q_, qt)
    return q_ / (1. - qt)
end

"""
    terminal_velocity(qt, qr, ρ, ρ_ground)

Return rain terminal velocity.
TODO - add citation
"""
function terminal_velocity(qt, qr, ρ, ρ_ground)
    rr = q2r(qr, qt)
    return 14.34 * ρ_ground^0.5 * ρ^-0.3654 * rr^0.1346
end

"""
    qr2qv(qt, ql, qi, T, ρ, p)

Return rain evaportaion rate.
TODO - add citation
"""
function qr2qv(qt, ql, qi, T, ρ, p)
  qv_star = saturation_shum(T, ρ, ql, qi)
  qv = qt - ql - qi

  rr = q2r(qr, qt)
  rv = q2r(qv, qt)
  rsat = q2r(qsat, qt)

  C = 1.6 + 124.9 * (1e-3 * ρ * rr)^0.2046 # ventilation factor

  return (1 - qt) * (1. - rv/rsat) * C * (1e-3 * ρ * rr)^0.525 /
          ρ / (540 + 2.55 * 1e5 / (p * rsat))
end

end #module Microphysics.jl
