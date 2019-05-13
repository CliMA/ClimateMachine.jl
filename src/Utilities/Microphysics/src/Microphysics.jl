"""
    Microphysics

Microphysics parameterization functions (for example condensation,
evaporation, autoconversion rates for cloud water and rain water)
TODO: Snow, ice, and 2-moment formulation
"""
module Microphysics

using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters

using Printf

# rates of conversion between microphysics categories
export qv2qli, ql2qr, qr2qv

# rain fall speed
export terminal_velocity

# conversion from specific humidity to mixing ratio (TODO - temporary)
export q2r

"""
    qv2qli(PhasePartition, T, ρ[, timescale=1])

Return the ql and qi source terms due to condesation/evaporation (for ql)
and sublimation/resublimation (for qi)
The source terms are obtained assuming a relaxation to equilibrum with a
constant timescale. The timescale is an optional parameter.
"""
#function qv2qli(q::PhasePartition, T::DT, ρ::DT, timescale::DT=DT(1)) where {DT<:Real} TODO
function qv2qli(q::PhasePartition, T::DT, ρ::DT, x::DT, z::DT, timescale::DT=DT(1)) where {DT<:Real}
    dqldt = - (
        q.liq - liquid_fraction_equil(T) * saturation_excess(T, ρ, q)
      ) / timescale

    dqidt = - (
        q.ice - (DT(1) - liquid_fraction_equil(T)) * saturation_excess(T, ρ, q)
      ) / timescale

  #if q.liq != 0 && x ==0
  #  if z > 1.3
  #      @printf("z = %1.1f, q_l = %.16e, dqldt = %.16e \n",
  #           z, q.liq, dqldt)
  #  end
  #  if z == 0
  #      @printf("  \n")
  #  end
  #end

  return (dqldt, dqidt)
end

"""
    ql2qr(PhasePartition[, timescale=1, ql_0=1e-4])

Return the qr source term due to collisions between cloud droplets
(autoconversion) parametrized following Kessler (TODO - add reference).
Contrary to the Kessler paper the process is not assumed to be instanteneous.
The assumed timescale and autoconverion threshold ql_0 are optional parameters.
"""
function ql2qr(q::PhasePartition, timescale::DT=DT(1), ql_0::DT=DT(1e-4)) where {DT<:Real}

  dqrdt = max(DT(0), q.liq - ql_0) / timescale

  return dqrdt
end

"""
    q2r(q_, qt)

Convert specific humidity to mixing ratio
"""
function q2r(q_::DT, qt::DT) where {DT<:Real}
    return q_ / (DT(1) - qt)
end

"""
    terminal_velocity(qt, qr, ρ, ρ_ground)

Return rain terminal velocity.
TODO - add citation
"""
function terminal_velocity(qt::DT, qr::DT, ρ::DT, ρ_ground::DT) where {DT<:Real}
    rr = q2r(qr, qt)
    return DT(14.34) * ρ_ground^DT(0.5) * ρ^-DT(0.3654) * rr^DT(0.1346)
end

"""
    qr2qv(qt, PhasePartition, T, ρ, p)

Return rain evaportaion rate.
TODO - add citation
"""
function qr2qv(q::PhasePartition, T::DT, ρ::DT, p::DT) where {DT<:Real}
  qv_star = saturation_shum(T, ρ, q.liq, q.ice)
  qv = q.tot - q.liq - q.ice

  rr = q2r(qr, q.tot)
  rv = q2r(qv, q.tot)
  rsat = q2r(qsat, q.tot)

  C = DT(1.6) + DT(124.9) * (DT(1e-3) * ρ * rr)^DT(0.2046) # ventilation factor

  return (DT(1) - q.tot) * (DT(1) - rv/rsat) * C * (DT(1e-3) * ρ * rr)^DT(0.525) /
          ρ / (DT(540) + DT(2.55) * DT(1e5) / (p * rsat))
end

end #module Microphysics.jl
