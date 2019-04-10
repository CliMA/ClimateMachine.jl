# Thermodynamic states
export ThermodynamicState, InternalEnergySHumEquil,
  InternalEnergySHumNonEquil, LiquidPotTempSHumEquil


"""
    ThermodynamicState{DT}

A thermodynamic state, which can be initialized for
various thermodynamic formulations (via its sub-types).
All `ThermodynamicState`'s have access to functions to
compute all other thermodynamic properties.
"""
abstract type ThermodynamicState{DT} end

"""
    InternalEnergySHumEquil{DT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic equilibrium (therefore, saturation adjustment
is needed).

# Constructors

    InternalEnergySHumEquil(e_int, q_tot, ρ)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct InternalEnergySHumEquil{DT} <: ThermodynamicState{DT}
  "internal energy"
  e_int::DT
  "total specific humidity"
  q_tot::DT
  "density of air (potentially moist)"
  density::DT
  "temperature: computed via [`saturation_adjustment`](@ref)"
  T::DT
end
InternalEnergySHumEquil(e_int, q_tot, ρ) = 
  InternalEnergySHumEquil(e_int, q_tot, ρ, saturation_adjustment(e_int, ρ, q_tot))


"""
    InternalEnergySHumNonEquil{DT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic non-equilibrium (therefore, temperature can
be computed directly).

# Constructors

    InternalEnergySHumNonEquil(e_int, q_tot, q_liq, q_ice, ρ)

# Fields

$(DocStringExtensions.FIELDS)

"""
struct InternalEnergySHumNonEquil{DT} <: ThermodynamicState{DT}
  "internal energy"
  e_int::DT
  "total specific humidity"
  q_tot::DT
  "specific humidity of liquid"
  q_liq::DT
  "specific humidity of ice"
  q_ice::DT
  "density of air (potentially moist)"
  density::DT
end

"""
    LiquidPotTempSHumEquil{DT} <: ThermodynamicState{DT}

A thermodynamic state assuming thermodynamic equilibrium (therefore, saturation adjustment
is needed).

# Constructors

    LiquidPotTempSHumEquil(θ_liq, q_tot, ρ, p)

# Fields

$(DocStringExtensions.FIELDS)

"""
struct LiquidPotTempSHumEquil{DT} <: ThermodynamicState{DT}
  "liquid potential temperature"
  θ_liq::DT
  "total specific humidity"
  q_tot::DT
  "density of air (potentially moist)"
  density::DT
  "pressure"
  pressure::DT
  "temperature: computed via [`saturation_adjustment`](@ref)"
  T::DT
end

function LiquidPotTempSHumEquil(θ_liq, q_tot, ρ, p)
    T = saturation_adjustment_q_t_θ_l(θ_liq, q_tot, ρ, p)
    LiquidPotTempSHumEquil(θ_liq, q_tot, ρ, p, T)
end
