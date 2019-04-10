"""
As far as I can tell there are 3 types of source terms
  - point-wise source terms
  - diffusion source terms
  - divergence flux source terms
My examples here are only pointwise
(this is what my microphysics needs and this is what I know). 
I'll add the other two types later.

"0") equilibrum thermodynamics, no rain
----------------------------------------
nmoist = 1 (qt)
Source terms:
  - none

1) non-equilibrum thermodynamics, no rain
------------------------------------------
nmoist = 3 (qt, ql, qi)
Source 
  - qt -> ql and qi 
    relaxation to equilibrum with some timescale and phase partitioning 
    between ql and qi

2) non-equilibrum thermodynamics, rain
--------------------------------------
nmoist = 4 (qt, ql, qi, qr)
  - qt -> ql and qi 
    relaxation to equilibrum with some timescale and phase partitioning 
    between ql and qi
  - ql -> qr (creation of rain, aka autoconversion)
  - qr -> qt (rain evaporation)
  - qr velocity = model_velocity + rain_terminal_velocity
"""

# TODO - I couldn't compile the code due to some MPI issues
# So I can't really check if my functions work. Hope to fix it soon!
using CLIMA.MoistThermodynamics
using CLIMA.Planetparameters

#TODO - All those function should be in the MoistThermodynamic
# or CloudPhysics modules
function qt_cond_evap(qt, ql, qi, T, ρ)
"""
Calulate the ql and qi source terms due to condesation and evaporation.
Assumed relaxation to equilibrum with a constant timescale.
"""
  timescale = 1

  dqldt = - (ql - liquid_fraction(T) * saturation_excess(T, ρ, qt, ql, qi))
          / timescale
  dqidt = - (qi - (1-liquid_fracion(T)) * saturation_excess(T, ρ, qt, ql, qi))
          / timescale
  return dqldt, dqidt
end

function autoconversion(qt, ql, qi, T, ρ)
"""
Calculate the qr source term due to collisions between cloud droplets.
Its parametrized here using the Kessler autoconversion threshold.
The threshold is expressed in terms of saturation excess.
The process is not assumed to be instanteneous,
instead a very short timescale is used.
"""
  timescale = 1
  excess_tr = 1.02

  dqrdt = max(0, ql - excess_tr * saturation_excess(T, ρ, qt, ql, qi))
          / timescale
  return dqrdt
end

function q2r(q_, qt)
"""
Convert specific humidity to mixing ratio
"""
    return q_ / (1. - qt)
end

function terminal_velocity(qt, qr, ρ, ρ_ground)
"""
Calcuate rain terminal velocity 
"""
    rr = q2r(qr, qt)
    return 14.34 * ρ_ground**0.5 * ρ**-0.3654 * rr**0.1346
end

function rain_evaporation_rate(qt, qv, ql, qi, T, ρ)
"""
TODO - add paper reference
"""
  qv_star = saturation_shum(T, ρ, ql, qi)

  rr = q2r(qr, qt)
  rv = q2r(qv, qt)
  rsat = q2r(qsat, qt)

  C = 1.6 + 124.9 * (1e-3 * rho * rr)**0.2046 # ventilation factor

  dqrdt = (1 - qt) * (1. - rv/rsat) * C * (1e-3 * ρ * rr)**0.525
          / ρ / (540 + 2.55 * 1e5 / (p0 * rsat))
end

function sourcefun_no_rain!(source, Qstate, X, Cstate)
"""
microphysics source terms in non-equilibrum thermodynamics, no rain
"""
  # get moist variables from state vector 
  # TODO - are those multiplied by ρ in state vector?
  ρ = Qstate[ρ]
  qt = Qstate[qt] / ρ
  ql = Qstate[ql] / ρ
  qi = Qstate[qi] / ρ
  qv = qt - ql - qi
  U = Qstate[U]
  V = Qstate[V]
  W = Qstate[W]
  height = #TODO

  # TODO - is this the way to get the internal and not total energy from state?
  # and should we put it into the MoistThermodynamics?
  e_int = Qstate[e] - ρ * gravity * height - .5 (U^2 + V^2 + W^2) / ρ

  T = air_temperature(e_int, qt, ql, qi)

  cevap_ql_rate, cevap_qi_rate = qt_cond_evap(qt, ql, qi, T, ρ)
  total_ql_rate = cevap_ql_rate
  total_qi_rate = cevap_qi_rate

  #TODO qt, qr, ql and qi can't be negative
  # so in practice for all of the above it should be something like
  total_ql_rate = min(ql, total_ql_rate * model_dt) / model_dt

end

function sourcefun_yes_rain!(source, Qstate, X, Cstate)
"""
microphysics sourceterms in non-equilibrum thermodynamics with rain
"""
  # get moist variables from state vector 
  # TODO - are those multiplied by ρ in state vector?
  ρ = Qstate[ρ]
  qt = Qstate[qt] / ρ
  ql = Qstate[ql] / ρ
  qi = Qstate[qi] / ρ
  qv = qt - ql - qi
  qr = Qstate[qr] / ρ
  U = Qstate[U]
  V = Qstate[V]
  W = Qstate[W]
  height = #TODO

  # TODO - is this the way to get the internal and not total energy from state?
  # and should we put it into the MoistThermodynamics?
  e_int = Qstate[e] - ρ * gravity * height - .5 (U^2 + V^2 + W^2) / ρ

  T = air_temperature(e_int, qt, ql, qi)

  cevap_ql_rate, cevap_qi_rate = qt_cond_evap(qt, ql, qi, T, ρ)
  acnv_qr_rate = autoconversion(qt, ql, qi, T, ρ)
  evap_qr_rate = rain_evaporation_rate(qt, qv, ql, qi, T, ρ)

  total_qt_rate = - acnv_qr_rate + evap_qr_rate
  total_ql_rate = cevap_ql_rate - acnv_rate
  total_qi_rate = cevap_qi_rate
  total_qr_rate = - total_qt_rate

  #TODO qt, qr, ql and qi can't be negative
  # so in practice for all of the above it should be something like
  total_qr_rate = min(ql, total_qr_rate * model_dt) / model_dt
  # TODO - what to do with other rates if the total rate is smaller than ql
end

function fluxfun!(flux, Qstate, gradQstate, X, Cstate)
"""
advection of rain
"""
  ρ = Qstate[ρ]
  qt = Qstate[qt] / ρ
  qr = Qstate[qr] / ρ
  U = Qstate[U]
  V = Qstate[V]
  W = Qstate[W]
  height = #TODO

  ρ_ground = ρ[height = surface] #TODO - how to do it?

  v_term = terminal_velocity(qt, qr, ρ, ρ_ground)
  # W = W - v_term for qr

end


function bcfun!(Qstate, gradQstate, bcid, normals, X, Cstate)
end
function gradbcfun!(Qstate, bcid, normals, X, Cstate)
end
function cfun!(Qstate)
end

# spacedisc = data needed for evaluating the right-hand side function
spacedisc = DGBalanceLawDiscretization(grid,
                                       nstate=neuler + nmoist,
                                       fluxfun! = ,
                                       sourcefun! = ,
                                       bcfun! = ,
                                       gradbcfun! = ,
                                       cfun! = ,
                                      )
