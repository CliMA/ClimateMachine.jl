# Thermodynamic states
export ThermodynamicState, PhaseEquil,
  PhaseNonEquil, LiquidIcePotTempSHumEquil


"""
    ThermodynamicState{DT}

A thermodynamic state, which can be initialized for
various thermodynamic formulations (via its sub-types).
All `ThermodynamicState`'s have access to functions to
compute all other thermodynamic properties.
"""
abstract type ThermodynamicState{DT} end

"""
    PhaseEquil{DT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic equilibrium (therefore, saturation adjustment
is needed).

# Constructors

    PhaseEquil(e_int, q_tot, ρ)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseEquil{DT} <: ThermodynamicState{DT}
  "internal energy"
  e_int::DT
  "total specific humidity"
  q_tot::DT
  "density of air (potentially moist)"
  density::DT
  "temperature: computed via [`saturation_adjustment`](@ref)"
  T::DT
end
PhaseEquil(e_int, q_tot, ρ) =
  PhaseEquil(e_int, q_tot, ρ, saturation_adjustment(e_int, ρ, q_tot))

"""
    LiquidIcePotTempSHumEquil(θ_liq_ice, q_tot, ρ, p)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from liquid-ice potential temperature.

 - `θ_liq_ice` - liquid-ice potential temperature
 - `q_tot` - total specific humidity
 - `ρ` - density
 - `p` - pressure
"""
function LiquidIcePotTempSHumEquil(θ_liq_ice, q_tot, ρ, p)
    T = saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, q_tot, ρ, p)
    q_liq, q_ice = phase_partitioning_eq(T, ρ, q_tot)
    e_int = internal_energy(T, q_tot, q_liq, q_ice)
    PhaseEquil(e_int, q_tot, ρ, T)
end


"""
    PhaseNonEquil{DT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic non-equilibrium (therefore, temperature can
be computed directly).

# Constructors

    PhaseNonEquil(e_int, q_tot, q_liq, q_ice, ρ)

# Fields

$(DocStringExtensions.FIELDS)

"""
struct PhaseNonEquil{DT} <: ThermodynamicState{DT}
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

