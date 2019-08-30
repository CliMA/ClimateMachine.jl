export PhasePartition
# Thermodynamic states
export ThermodynamicState,
       PhaseDry,
       PhaseEquil,
       PhaseNonEquil,
       TemperatureSHumEquil,
       LiquidIcePotTempSHumEquil,
       LiquidIcePotTempSHumEquil_no_ρ,
       LiquidIcePotTempSHumNonEquil_no_ρ

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
struct PhasePartition{DT<:Real}
  "total specific humidity"
  tot::DT
  "liquid water specific humidity (default: `0`)"
  liq::DT
  "ice specific humidity (default: `0`)"
  ice::DT
end

PhasePartition(q_tot::DT,q_liq::DT) where {DT<:Real} =
  PhasePartition(q_tot, q_liq, zero(DT))
PhasePartition(q_tot::DT) where {DT<:Real} =
  PhasePartition(q_tot, zero(DT), zero(DT))


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
may be needed).

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
  ρ::DT
  "temperature: computed via [`saturation_adjustment`](@ref)"
  T::DT
end
PhaseEquil(e_int::DT, q_tot::DT, ρ::DT) where {DT<:Real} =
  PhaseEquil(e_int, q_tot, ρ, saturation_adjustment(e_int, ρ, q_tot))

"""
    PhaseDry{DT} <: ThermodynamicState

A dry thermodynamic state (`q_tot = 0`).

# Constructors

    PhaseDry(e_int, ρ)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseDry{DT} <: ThermodynamicState{DT}
  "internal energy"
  e_int::DT
  "density of dry air"
  ρ::DT
end

"""
    LiquidIcePotTempSHumEquil(θ_liq_ice, q_tot, ρ, p)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` - liquid-ice potential temperature
 - `q_tot` - total specific humidity
 - `ρ` - density
 - `p` - pressure
"""
function LiquidIcePotTempSHumEquil(θ_liq_ice::DT, q_tot::DT, ρ::DT, p::DT) where {DT<:Real}
    T = saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, q_tot, ρ, p)
    q = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy(T, q)
    PhaseEquil(e_int, q_tot, ρ, T)
end

"""
    LiquidIcePotTempSHumEquil_no_ρ(θ_liq_ice, q_tot, p)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` - liquid-ice potential temperature
 - `q_tot` - total specific humidity
 - `p` - pressure
"""
function LiquidIcePotTempSHumEquil_no_ρ(θ_liq_ice::DT, q_tot::DT, p::DT) where {DT<:Real}
    q_pt_dry = PhasePartition(q_tot)
    T_dry = θ_liq_ice * exner(p, q_pt_dry)
    ρ_dry = air_density(T_dry, p, q_pt_dry)
    T = saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, q_tot, ρ_dry, p)
    ρ = air_density(T, p, q_pt_dry)
    q_pt = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy(T, q_pt)
    PhaseEquil(e_int, q_tot, ρ, T)
end

"""
    LiquidIcePotTempSHumEquil_no_ρ(θ_liq_ice, q_tot, p)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` - liquid-ice potential temperature
 - `q_pt` - phase partition
 - `p` - pressure
"""
function LiquidIcePotTempSHumEquil_no_ρ(θ_liq_ice::DT, q_pt::PhasePartition{DT}, p::DT) where {DT<:Real}
    T = θ_liq_ice * exner(p, q_pt)
    ρ = air_density(T, p, q_pt)
    e_int = internal_energy(T, q_pt)
    PhaseEquil(e_int, q_pt.tot, ρ, T)
end

"""
    TemperatureSHumEquil(T, q_tot, p)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from temperature.

 - `T` - temperature
 - `q_tot` - total specific humidity
 - `p` - pressure
"""
function TemperatureSHumEquil(T::DT, q_tot::DT, p::DT) where {DT<:Real}
    ρ = air_density(T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy(T, q)
    PhaseEquil(e_int, q_tot, ρ, T)
end

"""
   	 PhaseNonEquil{DT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic non-equilibrium (therefore, temperature can
be computed directly).

# Constructors

    PhaseNonEquil(e_int, q::PhasePartition, ρ)

# Fields

$(DocStringExtensions.FIELDS)

"""
struct PhaseNonEquil{DT} <: ThermodynamicState{DT}
  "internal energy"
  e_int::DT
  "phase partition"
  q::PhasePartition{DT}
  "density of air (potentially moist)"
  ρ::DT
end

"""
    LiquidIcePotTempSHumNonEquil_no_ρ(θ_liq_ice, q_tot, p)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` - liquid-ice potential temperature
 - `q_pt` - phase partition
 - `p` - pressure
"""
function LiquidIcePotTempSHumNonEquil_no_ρ(θ_liq_ice::DT, q_pt::PhasePartition{DT}, p::DT) where {DT<:Real}
    T = θ_liq_ice * exner(p, q_pt)
    ρ = air_density(T, p, q_pt)
    e_int = internal_energy(T, q_pt)
    PhaseNonEquil(e_int, q_pt, ρ)
end
