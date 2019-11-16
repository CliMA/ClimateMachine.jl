export PhasePartition
# Thermodynamic states
export ThermodynamicState,
       PhaseDry,
       PhaseEquil,
       PhaseNonEquil,
       TemperatureSHumEquil,
       LiquidIcePotTempSHumEquil,
       LiquidIcePotTempSHumNonEquil,
       LiquidIcePotTempSHumNonEquil_given_pressure

"""
    PhasePartition

Represents the mass fractions of the moist air mixture.

# Constructors

    PhasePartition(q_tot::Real[, q_liq::Real[, q_ice::Real]])
    PhasePartition(ts::ThermodynamicState)

See also [`PhasePartition_equil`](@ref)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhasePartition{FT<:Real}
  "total specific humidity"
  tot::FT
  "liquid water specific humidity (default: `0`)"
  liq::FT
  "ice specific humidity (default: `0`)"
  ice::FT
end

PhasePartition(q_tot::FT,q_liq::FT) where {FT<:Real} =
  PhasePartition(q_tot, q_liq, zero(FT))
PhasePartition(q_tot::FT) where {FT<:Real} =
  PhasePartition(q_tot, zero(FT), zero(FT))


"""
    ThermodynamicState{FT}

A thermodynamic state, which can be initialized for
various thermodynamic formulations (via its sub-types).
All `ThermodynamicState`'s have access to functions to
compute all other thermodynamic properties.
"""
abstract type ThermodynamicState{FT} end

"""
    PhaseEquil{FT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic equilibrium (therefore, saturation adjustment
may be needed).

# Constructors

    PhaseEquil(e_int, q_tot, ρ)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseEquil{FT} <: ThermodynamicState{FT}
  "internal energy"
  e_int::FT
  "total specific humidity"
  q_tot::FT
  "density of air (potentially moist)"
  ρ::FT
  "temperature: computed via [`saturation_adjustment`](@ref)"
  T::FT
end
PhaseEquil(e_int::FT, q_tot::FT, ρ::FT) where {FT<:Real} =
  PhaseEquil(e_int, q_tot, ρ, saturation_adjustment(e_int, ρ, q_tot))

"""
    PhaseDry{FT} <: ThermodynamicState

A dry thermodynamic state (`q_tot = 0`).

# Constructors

    PhaseDry(e_int, ρ)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseDry{FT} <: ThermodynamicState{FT}
  "internal energy"
  e_int::FT
  "density of dry air"
  ρ::FT
end

"""
    LiquidIcePotTempSHumEquil(θ_liq_ice, q_tot, ρ)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` - liquid-ice potential temperature
 - `q_tot` - total specific humidity
 - `ρ` - density
"""
function LiquidIcePotTempSHumEquil(θ_liq_ice::FT, q_tot::FT, ρ::FT) where {FT<:Real}
    T = saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, q_tot, ρ)
    q_pt = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy(T, q_pt)
    return PhaseEquil(e_int, q_tot, ρ, T)
end

"""
    TemperatureSHumEquil(T, q_tot, p)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from temperature.

 - `T` - temperature
 - `q_tot` - total specific humidity
 - `p` - pressure
"""
function TemperatureSHumEquil(T::FT, q_tot::FT, p::FT) where {FT<:Real}
    ρ = air_density(T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy(T, q)
    PhaseEquil(e_int, q_tot, ρ, T)
end

"""
   	 PhaseNonEquil{FT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic non-equilibrium (therefore, temperature can
be computed directly).

# Constructors

    PhaseNonEquil(e_int, q::PhasePartition, ρ)

# Fields

$(DocStringExtensions.FIELDS)

"""
struct PhaseNonEquil{FT} <: ThermodynamicState{FT}
  "internal energy"
  e_int::FT
  "phase partition"
  q::PhasePartition{FT}
  "density of air (potentially moist)"
  ρ::FT
end

"""
    LiquidIcePotTempSHumNonEquil(θ_liq_ice, q_pt, ρ)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` - liquid-ice potential temperature
 - `q_pt` - phase partition
 - `ρ` - density
"""
function LiquidIcePotTempSHumNonEquil(θ_liq_ice::FT, q_pt::PhasePartition{FT}, ρ::FT) where {FT<:Real}
    T = air_temperature_from_liquid_ice_pottemp_non_linear(θ_liq_ice, ρ, q_pt)
    e_int = internal_energy(T, q_pt)
    PhaseNonEquil(e_int, q_pt, ρ)
end

"""
    LiquidIcePotTempSHumNonEquil_given_pressure(θ_liq_ice, q_pt, p)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` - liquid-ice potential temperature
 - `q_pt` - phase partition
 - `p` - pressure
"""
function LiquidIcePotTempSHumNonEquil_given_pressure(θ_liq_ice::FT, q_pt::PhasePartition{FT}, p::FT) where {FT<:Real}
    T = air_temperature_from_liquid_ice_pottemp_given_pressure(θ_liq_ice, p, q_pt)
    ρ = air_density(T, p, q_pt)
    e_int = internal_energy(T, q_pt)
    PhaseNonEquil(e_int, q_pt, ρ)
end
