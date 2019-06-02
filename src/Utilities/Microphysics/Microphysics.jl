"""
    Microphysics

Microphysics parameterization functions (for example condensation,
evaporation, autoconversion rates for cloud water and rain water)
TODO: Snow, ice, and 2-moment formulation
"""
module Microphysics

using ..MoistThermodynamics
using ..PlanetParameters

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
function qv2qli(q_sa::PhasePartition, q::PhasePartition,
                timescale::DT=DT(1)) where {DT<:Real}

    dqldt = (q_sa.liq - q.liq) / timescale
    dqidt = (q_sa.ice - q.ice) / timescale

  return (dqldt, dqidt)
end

"""
    ql2qr(PhasePartition[, timescale=1, ql_0=1e-4])

Return the qr source term due to collisions between cloud droplets
(autoconversion) parametrized following Kessler (TODO - add reference).
Contrary to the Kessler paper the process is not assumed to be instanteneous.
The assumed timescale and autoconverion threshold ql_0 are optional parameters.
"""
function ql2qr(ql::DT, timescale::DT=DT(1), ql_0::DT=DT(5e-4)) where {DT<:Real}

  dqrdt = max(DT(0), ql - ql_0) / timescale

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
function terminal_velocity(qt::DT, qr::DT, ρ::DT,
                           ρ_ground::DT) where {DT<:Real}
    rr  = q2r(qr, qt)
    vel::DT = 0

    if (qr > DT(1e-8)) # TODO - assert positive definite elswhere
      vel = DT(14.34) * ρ_ground^DT(0.5) * ρ^-DT(0.3654) * rr^DT(0.1346)
      #vel = 1e-2  #TODO - tmp
    end

    return vel
end

"""
    qr2qv(qt, PhasePartition, T, ρ, p)

Return rain evaportaion rate.
TODO - add citation
"""
function qr2qv(q::PhasePartition, T::DT, ρ::DT, p::DT, qr::DT) where {DT<:Real}

  ret::DT = 0

  if (qr > 0) # TODO - assert positive definite elswhere

    qv_sat = saturation_shum(T, ρ, q)
    qv = q.tot - q.liq - q.ice

    rr = q2r(qr, q.tot)
    rv = q2r(qv, q.tot)
    rv_sat = q2r(qv_sat, q.tot)

    # ventilation factor
    C = DT(1.6) + DT(124.9) * (DT(1e-3) * ρ * rr)^DT(0.2046)

    ret = (DT(1) - q.tot) * (DT(1) - rv/rv_sat) * C *
          (DT(1e-3) * ρ * rr)^DT(0.525) /
          ρ / (DT(540) + DT(2.55) * DT(1e5) / (p * rv_sat))
  end

  return ret

end

end #module Microphysics.jl
