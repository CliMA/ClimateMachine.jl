export PhasePartition
# Thermodynamic states
export ThermodynamicState,
       PhaseDry,
       PhaseEquil,
       PhaseNonEquil,
       TemperatureSHumEquil,
       LiquidIcePotTempSHumEquil,
       LiquidIcePotTempSHumEquil_given_pressure,
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

    PhaseEquil(e_int, ρ, q_tot)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseEquil{FT} <: ThermodynamicState{FT}
  "internal energy"
  e_int::FT
  "density of air (potentially moist)"
  ρ::FT
  "total specific humidity"
  q_tot::FT
  "temperature: computed via [`saturation_adjustment`](@ref)"
  T::FT
end
function PhaseEquil(e_int::FT,
                    ρ::FT,
                    q_tot::FT,
                    tol::FT=FT(1e-1),
                    maxiter::Int=5,
                    sat_adjust::F=saturation_adjustment) where {FT<:Real,F}
    return PhaseEquil{FT}(e_int, ρ, q_tot, sat_adjust(e_int, ρ, q_tot, tol, maxiter))
end

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
    LiquidIcePotTempSHumEquil(θ_liq_ice, ρ, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` liquid-ice potential temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
 - `tol` tolerance for saturation adjustment
 - `maxiter` maximum iterations for saturation adjustment
"""
function LiquidIcePotTempSHumEquil(θ_liq_ice::FT,
                                   ρ::FT,
                                   q_tot::FT,
                                   tol::FT=FT(1e-1),
                                   maxiter::Int=30
                                   ) where {FT<:Real}
    T = saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, ρ, q_tot, tol, maxiter)
    q_pt = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy(T, q_pt)
    return PhaseEquil{FT}(e_int, ρ, q_tot, T)
end

"""
    LiquidIcePotTempSHumEquil_given_pressure(θ_liq_ice, p, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` liquid-ice potential temperature
 - `p` pressure
 - `q_tot` total specific humidity
 - `tol` tolerance for saturation adjustment
 - `maxiter` maximum iterations for saturation adjustment
"""
function LiquidIcePotTempSHumEquil_given_pressure(θ_liq_ice::FT,
                                                  p::FT,
                                                  q_tot::FT,
                                                  tol::FT=FT(1e-1),
                                                  maxiter::Int=30) where {FT<:Real}
    T = saturation_adjustment_q_tot_θ_liq_ice_given_pressure(θ_liq_ice, p, q_tot, tol, maxiter)
    ρ = air_density(T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy(T, q)
    return PhaseEquil{FT}(e_int, ρ, q.tot, T)
end

"""
    TemperatureSHumEquil(T, p, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from temperature.

 - `T` temperature
 - `p` pressure
 - `q_tot` total specific humidity
"""
function TemperatureSHumEquil(T::FT, p::FT, q_tot::FT) where {FT<:Real}
    ρ = air_density(T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy(T, q)
    return PhaseEquil{FT}(e_int, ρ, q_tot, T)
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
  "density of air (potentially moist)"
  ρ::FT
  "phase partition"
  q::PhasePartition{FT}
end

"""
    LiquidIcePotTempSHumNonEquil(θ_liq_ice, ρ, q_pt)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` liquid-ice potential temperature
 - `ρ` (moist-)air density
 - `q_pt` phase partition
and, optionally
 - `tol` tolerance for non-linear equation solve
 - `maxiter` maximum iterations for non-linear equation solve
"""
function LiquidIcePotTempSHumNonEquil(θ_liq_ice::FT,
                                      ρ::FT,
                                      q_pt::PhasePartition{FT},
                                      tol::FT=FT(1e-1),
                                      maxiter::Int=5
                                      ) where {FT<:Real}
    T = air_temperature_from_liquid_ice_pottemp_non_linear(θ_liq_ice, ρ, tol, maxiter, q_pt)
    e_int = internal_energy(T, q_pt)
    return PhaseNonEquil{FT}(e_int, ρ, q_pt)
end

"""
    LiquidIcePotTempSHumNonEquil_given_pressure(θ_liq_ice, p, q_pt)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` liquid-ice potential temperature
 - `p` pressure
 - `q_pt` phase partition
"""
function LiquidIcePotTempSHumNonEquil_given_pressure(θ_liq_ice::FT, p::FT, q_pt::PhasePartition{FT}) where {FT<:Real}
    T = air_temperature_from_liquid_ice_pottemp_given_pressure(θ_liq_ice, p, q_pt)
    ρ = air_density(T, p, q_pt)
    e_int = internal_energy(T, q_pt)
    return PhaseNonEquil{FT}(e_int, ρ, q_pt)
end

"""
    fixed_lapse_rate_ref_state(z::FT,
                               T_surface::FT,
                               T_min::FT) where {FT<:AbstractFloat}

Fixed lapse rate hydrostatic reference state
"""
function fixed_lapse_rate_ref_state(z::FT,
                                    T_surface::FT,
                                    T_min::FT) where {FT<:AbstractFloat}
  Γ = FT(grav)/FT(cp_d)
  z_tropopause = (T_surface - T_min) / Γ
  H_min = FT(R_d) * T_min / FT(grav)
  T = max(T_surface - Γ*z, T_min)
  p = FT(MSLP)*(T / T_surface)^(FT(grav)/(FT(R_d)*Γ))
  T == T_min && (p = p * exp(-(z-z_tropopause)/FT(H_min)))
  ρ = p / (FT(R_d) * T)
  return T,p,ρ
end

"""
    tested_convergence_range(FT, n)

A range of input arguments to thermodynamic state constructors
 - `e_int` internal energy
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
 - `q_pt` phase partition
 - `T` air temperature
 - `θ_liq_ice` liquid-ice potential temperature
that are tested for convergence in saturation adjustment.

Note that the output vectors are of size ``n*n_RH``, and they
should span the input arguments to all of the constructors.
"""
function tested_convergence_range(FT, n)
  n_RS1 = 10
  n_RS2 = 20
  n_RS = n_RS1+n_RS2
  z_range  = range(FT(0), stop=FT(2.5e4), length=n)
  relative_sat1 = range(FT(0), stop=FT(1), length=n_RS1)
  relative_sat2 = range(FT(1), stop=FT(1.02), length=n_RS2)
  relative_sat = [relative_sat1...,relative_sat2...]
  T_min = FT(150)
  T_surface = FT(350)

  args = fixed_lapse_rate_ref_state.(z_range, Ref(T_surface), Ref(T_min))
  T,p,ρ = getindex.(args, 1),
          getindex.(args, 2),
          getindex.(args, 3)

  p  = collect(Iterators.flatten([p  for RS in 1:n_RS]))
  ρ  = collect(Iterators.flatten([ρ  for RS in 1:n_RS]))
  T  = collect(Iterators.flatten([T  for RS in 1:n_RS]))
  relative_sat = collect(Iterators.flatten([relative_sat for RS in 1:n]))

  # Additional variables
  q_sat = q_vap_saturation.(T, ρ)
  q_tot = min.(relative_sat .*q_sat, FT(1))
  q_pt = PhasePartition_equil.(T, ρ, q_tot)
  e_int = internal_energy.(T, q_pt)
  θ_liq_ice = liquid_ice_pottemp.(T, ρ, q_pt)
  return e_int, ρ, q_tot, q_pt, T, p, θ_liq_ice
end
