# Atmospheric equation of state
export air_pressure,
    air_temperature, air_density, specific_volume, soundspeed_air
export total_specific_humidity

# Energies
export total_energy, internal_energy, internal_energy_sat

# Specific heats and gas constants of moist air
export cp_m, cv_m, gas_constant_air, gas_constants

# Latent heats
export latent_heat_vapor,
    latent_heat_sublim, latent_heat_fusion, latent_heat_liq_ice

# Saturation vapor pressures and specific humidities over liquid and ice
export Liquid, Ice
export saturation_vapor_pressure, q_vap_saturation_generic, q_vap_saturation
export saturation_excess

# Functions used in thermodynamic equilibrium among phases (liquid and ice
# determined diagnostically from total water specific humidity)

export liquid_fraction, PhasePartition_equil

# Auxiliary functions, e.g., for diagnostic purposes
export dry_pottemp,
    dry_pottemp_given_pressure, virtual_pottemp, exner, exner_given_pressure
export liquid_ice_pottemp,
    liquid_ice_pottemp_given_pressure, liquid_ice_pottemp_sat, relative_humidity
export air_temperature_from_liquid_ice_pottemp,
    air_temperature_from_liquid_ice_pottemp_given_pressure
export air_temperature_from_liquid_ice_pottemp_non_linear
export vapor_specific_humidity

"""
    gas_constant_air(param_set, [q::PhasePartition])

The specific gas constant of moist air given
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function gas_constant_air(param_set::APS, q::PhasePartition{FT}) where {FT}
    _R_d::FT = R_d(param_set)
    _molmass_ratio::FT = molmass_ratio(param_set)
    return _R_d *
           (1 + (_molmass_ratio - 1) * q.tot - _molmass_ratio * (q.liq + q.ice))
end

gas_constant_air(param_set::APS, ::Type{FT}) where {FT} =
    gas_constant_air(param_set, q_pt_0(FT))

"""
    gas_constant_air(ts::ThermodynamicState)

The specific gas constant of moist air given
a thermodynamic state `ts`.
"""
gas_constant_air(ts::ThermodynamicState) =
    gas_constant_air(ts.param_set, PhasePartition(ts))
gas_constant_air(ts::PhaseDry{FT}) where {FT <: Real} = FT(R_d(ts.param_set))

"""
    vapor_specific_humidity(q::PhasePartition{FT})

The vapor specific humidity, given a `PhasePartition` `q`.
"""
vapor_specific_humidity(q::PhasePartition) = q.tot - q.liq - q.ice
vapor_specific_humidity(ts::ThermodynamicState) =
    vapor_specific_humidity(PhasePartition(ts))

"""
    air_pressure(param_set, T, ρ[, q::PhasePartition])

The air pressure from the equation of state
(ideal gas law) where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` air temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function air_pressure(
    param_set::APS,
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    return gas_constant_air(param_set, q) * ρ * T
end

"""
    air_pressure(ts::ThermodynamicState)

The air pressure from the equation of state
(ideal gas law), given a thermodynamic state `ts`.
"""
air_pressure(ts::ThermodynamicState) = air_pressure(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
)


"""
    air_density(param_set, T, p[, q::PhasePartition])

The (moist-)air density from the equation of state
(ideal gas law) where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` air temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function air_density(
    param_set::APS,
    T::FT,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    return p / (gas_constant_air(param_set, q) * T)
end

"""
    air_density(ts::ThermodynamicState)

The (moist-)air density, given a thermodynamic state `ts`.
"""
air_density(ts::ThermodynamicState) = ts.ρ

"""
    specific_volume(ts::ThermodynamicState)

The (moist-)air specific volume, given a thermodynamic state `ts`.
"""
specific_volume(ts::ThermodynamicState) = 1 / air_density(ts)

"""
    total_specific_humidity(ts::ThermodynamicState)

Total specific humidity, given a thermodynamic state `ts`.
"""
total_specific_humidity(ts::ThermodynamicState) = ts.q_tot
total_specific_humidity(ts::PhaseDry{FT}) where {FT} = FT(0)
total_specific_humidity(ts::PhaseNonEquil) = ts.q.tot

"""
    cp_m(param_set, [q::PhasePartition])

The isobaric specific heat capacity of moist air given
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
and, optionally
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function cp_m(
    param_set::APS,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    _cp_d::FT = cp_d(param_set)
    _cp_v::FT = cp_v(param_set)
    _cp_l::FT = cp_l(param_set)
    _cp_i::FT = cp_i(param_set)
    return _cp_d +
           (_cp_v - _cp_d) * q.tot +
           (_cp_l - _cp_v) * q.liq +
           (_cp_i - _cp_v) * q.ice
end

cp_m(param_set::APS, ::Type{FT}) where {FT <: Real} =
    cp_m(param_set, q_pt_0(FT))

"""
    cp_m(ts::ThermodynamicState)

The isobaric specific heat capacity of moist air, given a thermodynamic state `ts`.
"""
cp_m(ts::ThermodynamicState) = cp_m(ts.param_set, PhasePartition(ts))
cp_m(ts::PhaseDry{FT}) where {FT <: Real} = FT(cp_d(ts.param_set))

"""
    cv_m(param_set, [q::PhasePartition])

The isochoric specific heat capacity of moist air given
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
and, optionally
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function cv_m(param_set::APS, q::PhasePartition{FT}) where {FT <: Real}
    _cv_d::FT = cv_d(param_set)
    _cv_v::FT = cv_v(param_set)
    _cv_l::FT = cv_l(param_set)
    _cv_i::FT = cv_i(param_set)
    return _cv_d +
           (_cv_v - _cv_d) * q.tot +
           (_cv_l - _cv_v) * q.liq +
           (_cv_i - _cv_v) * q.ice
end

cv_m(param_set::APS, ::Type{FT}) where {FT <: Real} =
    cv_m(param_set, q_pt_0(FT))

"""
    cv_m(ts::ThermodynamicState)

The isochoric specific heat capacity of moist air, given a thermodynamic state `ts`.
"""
cv_m(ts::ThermodynamicState) = cv_m(ts.param_set, PhasePartition(ts))
cv_m(ts::PhaseDry{FT}) where {FT <: Real} = FT(cv_d(ts.param_set))


"""
    (R_m, cp_m, cv_m, γ_m) = gas_constants(param_set, [q::PhasePartition])

Wrapper to compute all gas constants at once, where optionally,
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.

The function returns a tuple of
 - `R_m` [`gas_constant_air`](@ref)
 - `cp_m` [`cp_m`](@ref)
 - `cv_m` [`cv_m`](@ref)
 - `γ_m = cp_m/cv_m`
"""
function gas_constants(
    param_set::APS,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    R_gas = gas_constant_air(param_set, q)
    cp = cp_m(param_set, q)
    cv = cv_m(param_set, q)
    γ = cp / cv
    return (R_gas, cp, cv, γ)
end

"""
    (R_m, cp_m, cv_m, γ_m) = gas_constants(ts::ThermodynamicState)

Wrapper to compute all gas constants at once, given a thermodynamic state `ts`.

The function returns a tuple of
 - `R_m` [`gas_constant_air`](@ref)
 - `cp_m` [`cp_m`](@ref)
 - `cv_m` [`cv_m`](@ref)
 - `γ_m = cp_m/cv_m`

"""
gas_constants(ts::ThermodynamicState) =
    gas_constants(ts.param_set, PhasePartition(ts))

"""
    air_temperature(param_set, e_int, q::PhasePartition)

The air temperature, where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `e_int` internal energy per unit mass
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function air_temperature(
    param_set::APS,
    e_int::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    _T_0::FT = T_0(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    return _T_0 +
           (
        e_int - (q.tot - q.liq) * _e_int_v0 + q.ice * (_e_int_v0 + _e_int_i0)
    ) / cv_m(param_set, q)
end

"""
    air_temperature(ts::ThermodynamicState)

The air temperature, given a thermodynamic state `ts`.
"""
air_temperature(ts::ThermodynamicState) =
    air_temperature(ts.param_set, internal_energy(ts), PhasePartition(ts))
air_temperature(ts::PhaseEquil) = ts.T


"""
    internal_energy(param_set, T[, q::PhasePartition])

The internal energy per unit mass, given a thermodynamic state `ts` or

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function internal_energy(
    param_set::APS,
    T::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    _T_0::FT = T_0(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    return cv_m(param_set, q) * (T - _T_0) + (q.tot - q.liq) * _e_int_v0 -
           q.ice * (_e_int_v0 + _e_int_i0)
end

"""
    internal_energy(ts::ThermodynamicState)

The internal energy per unit mass, given a thermodynamic state `ts`.
"""
internal_energy(ts::ThermodynamicState) = ts.e_int

"""
    internal_energy(ρ::FT, ρe::FT, ρu::AbstractVector{FT}, e_pot::FT)

The internal energy per unit mass, given
 - `ρ` (moist-)air density
 - `ρe` total energy per unit volume
 - `ρu` momentum vector
 - `e_pot` potential energy (e.g., gravitational) per unit mass
"""
@inline function internal_energy(
    ρ::FT,
    ρe::FT,
    ρu::AbstractVector{FT},
    e_pot::FT,
) where {FT <: Real}
    ρinv = 1 / ρ
    ρe_kin = ρinv * sum(abs2, ρu) / 2
    ρe_pot = ρ * e_pot
    ρe_int = ρe - ρe_kin - ρe_pot
    e_int = ρinv * ρe_int
    return e_int
end

"""
    internal_energy_sat(param_set, T, ρ, q_tot)

The internal energy per unit mass in thermodynamic equilibrium at saturation where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
"""
function internal_energy_sat(
    param_set::APS,
    T::FT,
    ρ::FT,
    q_tot::FT,
) where {FT <: Real}
    return internal_energy(
        param_set,
        T,
        PhasePartition_equil(param_set, T, ρ, q_tot),
    )
end

"""
    internal_energy_sat(ts::ThermodynamicState)

The internal energy per unit mass in
thermodynamic equilibrium at saturation,
given a thermodynamic state `ts`.
"""
internal_energy_sat(ts::ThermodynamicState) = internal_energy_sat(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    total_specific_humidity(ts),
)


"""
    total_energy(param_set, e_kin, e_pot, T[, q::PhasePartition])

The total energy per unit mass, given

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `e_kin` kinetic energy per unit mass
 - `e_pot` potential energy per unit mass
 - `T` temperature
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.

"""
function total_energy(
    param_set::APS,
    e_kin::FT,
    e_pot::FT,
    T::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    return e_kin + e_pot + internal_energy(param_set, T, q)
end

"""
    total_energy(e_kin, e_pot, ts::ThermodynamicState)

The total energy per unit mass
given a thermodynamic state `ts`.
"""
function total_energy(
    e_kin::FT,
    e_pot::FT,
    ts::ThermodynamicState{FT},
) where {FT <: Real}
    return internal_energy(ts) + e_kin + e_pot
end

"""
    soundspeed_air(param_set, T[, q::PhasePartition])

The speed of sound in unstratified air, where
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function soundspeed_air(
    param_set::APS,
    T::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    γ = cp_m(param_set, q) / cv_m(param_set, q)
    R_m = gas_constant_air(param_set, q)
    return sqrt(γ * R_m * T)
end

"""
    soundspeed_air(ts::ThermodynamicState)

The speed of sound in unstratified air given a thermodynamic state `ts`.
"""
soundspeed_air(ts::ThermodynamicState) =
    soundspeed_air(ts.param_set, air_temperature(ts), PhasePartition(ts))


"""
    latent_heat_vapor(param_set, T::FT) where {FT<:Real}

The specific latent heat of vaporization where
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
"""
function latent_heat_vapor(param_set::APS, T::FT) where {FT <: Real}
    _cp_l::FT = cp_l(param_set)
    _cp_v::FT = cp_v(param_set)
    _LH_v0::FT = LH_v0(param_set)
    return latent_heat_generic(param_set, T, _LH_v0, _cp_v - _cp_l)
end

"""
    latent_heat_vapor(ts::ThermodynamicState)

The specific latent heat of vaporization
given a thermodynamic state `ts`.
"""
latent_heat_vapor(ts::ThermodynamicState) =
    latent_heat_vapor(ts.param_set, air_temperature(ts))

"""
    latent_heat_sublim(param_set, T::FT) where {FT<:Real}

The specific latent heat of sublimation where
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
"""
function latent_heat_sublim(param_set::APS, T::FT) where {FT <: Real}
    _LH_s0::FT = LH_s0(param_set)
    _cp_v::FT = cp_v(param_set)
    _cp_i::FT = cp_i(param_set)
    return latent_heat_generic(param_set, T, _LH_s0, _cp_v - _cp_i)
end

"""
    latent_heat_sublim(ts::ThermodynamicState)

The specific latent heat of sublimation
given a thermodynamic state `ts`.
"""
latent_heat_sublim(ts::ThermodynamicState) =
    latent_heat_sublim(ts.param_set, air_temperature(ts))

"""
    latent_heat_fusion(param_set, T::FT) where {FT<:Real}

The specific latent heat of fusion where
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
"""
function latent_heat_fusion(param_set::APS, T::FT) where {FT <: Real}
    _LH_f0::FT = LH_f0(param_set)
    _cp_l::FT = cp_l(param_set)
    _cp_i::FT = cp_i(param_set)
    return latent_heat_generic(param_set, T, _LH_f0, _cp_l - _cp_i)
end

"""
    latent_heat_fusion(ts::ThermodynamicState)

The specific latent heat of fusion
given a thermodynamic state `ts`.
"""
latent_heat_fusion(ts::ThermodynamicState) =
    latent_heat_fusion(ts.param_set, air_temperature(ts))

"""
    latent_heat_generic(param_set, T::FT, LH_0::FT, Δcp::FT) where {FT<:Real}

The specific latent heat of a generic phase transition between
two phases, computed using Kirchhoff's relation with constant
isobaric specific heat capacities of the two phases, given

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `LH_0` latent heat of the phase transition at `T_0`
 - `Δcp` difference between the isobaric specific heat capacities
         (heat capacity in the higher-temperature phase minus that
         in the lower-temperature phase).
"""
function latent_heat_generic(
    param_set::APS,
    T::FT,
    LH_0::FT,
    Δcp::FT,
) where {FT <: Real}
    _T_0::FT = T_0(param_set)
    return LH_0 + Δcp * (T - _T_0)
end


"""
    Phase

A condensed phase, to dispatch over
[`saturation_vapor_pressure`](@ref) and
[`q_vap_saturation_generic`](@ref).
"""
abstract type Phase end

"""
    Liquid <: Phase

A liquid phase, to dispatch over
[`saturation_vapor_pressure`](@ref) and
[`q_vap_saturation_generic`](@ref).
"""
struct Liquid <: Phase end

"""
    Ice <: Phase

An ice phase, to dispatch over
[`saturation_vapor_pressure`](@ref) and
[`q_vap_saturation_generic`](@ref).
"""
struct Ice <: Phase end

"""
    saturation_vapor_pressure(param_set, T, Liquid())

Return the saturation vapor pressure over a plane liquid surface given
 - `T` temperature
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details

    saturation_vapor_pressure(param_set, T, Ice())

Return the saturation vapor pressure over a plane ice surface given
 - `T` temperature
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details

    saturation_vapor_pressure(param_set, T, LH_0, Δcp)

Compute the saturation vapor pressure over a plane surface by integration
of the Clausius-Clapeyron relation.

The Clausius-Clapeyron relation

    dlog(p_v_sat)/dT = [LH_0 + Δcp * (T-T_0)]/(R_v*T^2)

is integrated from the triple point temperature `T_triple`, using
Kirchhoff's relation

    L = LH_0 + Δcp * (T - T_0)

for the specific latent heat `L` with constant isobaric specific
heats of the phases. The linear dependence of the specific latent heat
on temperature `T` allows analytic integration of the Clausius-Clapeyron
relation to obtain the saturation vapor pressure `p_v_sat` as a function of
the triple point pressure `press_triple`.

"""
function saturation_vapor_pressure(
    param_set::APS,
    T::FT,
    ::Liquid,
) where {FT <: Real}
    _LH_v0::FT = LH_v0(param_set)
    _cp_v::FT = cp_v(param_set)
    _cp_l::FT = cp_l(param_set)
    return saturation_vapor_pressure(param_set, T, _LH_v0, _cp_v - _cp_l)
end

function saturation_vapor_pressure(
    ts::ThermodynamicState{FT},
    ::Liquid,
) where {FT <: Real}
    _LH_v0::FT = LH_v0(ts.param_set)
    _cp_v::FT = cp_v(ts.param_set)
    _cp_l::FT = cp_l(ts.param_set)
    return saturation_vapor_pressure(
        ts.param_set,
        air_temperature(ts),
        _LH_v0,
        _cp_v - _cp_l,
    )

end

function saturation_vapor_pressure(
    param_set::APS,
    T::FT,
    ::Ice,
) where {FT <: Real}
    _LH_s0::FT = LH_s0(param_set)
    _cp_v::FT = cp_v(param_set)
    _cp_i::FT = cp_i(param_set)
    return saturation_vapor_pressure(param_set, T, _LH_s0, _cp_v - _cp_i)
end

function saturation_vapor_pressure(
    ts::ThermodynamicState{FT},
    ::Ice,
) where {FT <: Real}
    _LH_s0::FT = LH_s0(ts.param_set)
    _cp_v::FT = cp_v(ts.param_set)
    _cp_i::FT = cp_i(ts.param_set)
    return saturation_vapor_pressure(
        ts.param_set,
        air_temperature(ts),
        _LH_s0,
        _cp_v - _cp_i,
    )
end

function saturation_vapor_pressure(
    param_set::APS,
    T::FT,
    LH_0::FT,
    Δcp::FT,
) where {FT <: Real}
    _press_triple::FT = press_triple(param_set)
    _R_v::FT = R_v(param_set)
    _T_triple::FT = T_triple(param_set)
    _T_0::FT = T_0(param_set)

    return _press_triple *
           (T / _T_triple)^(Δcp / _R_v) *
           exp((LH_0 - Δcp * _T_0) / _R_v * (1 / _T_triple - 1 / T))

end

"""
    q_vap_saturation_generic(param_set, T, ρ[; phase=Liquid()])

Compute the saturation specific humidity over a plane surface of condensate, given

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `Liquid()` indicating condensate is liquid
 - `Ice()` indicating condensate is ice
"""
function q_vap_saturation_generic(
    param_set::APS,
    T::FT,
    ρ::FT;
    phase::Phase = Liquid(),
) where {FT <: Real}
    p_v_sat = saturation_vapor_pressure(param_set, T, phase)
    return q_vap_saturation_from_pressure(param_set, T, ρ, p_v_sat)
end

"""
    q_vap_saturation(param_set, T, ρ[, q::PhasePartition])

Compute the saturation specific humidity, given

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref)

If the `PhasePartition` `q` is given, the saturation specific humidity is that of a
mixture of liquid and ice, computed in a thermodynamically consistent way from the
weighted sum of the latent heats of the respective phase transitions (Pressel et al.,
JAMES, 2015). That is, the saturation vapor pressure and from it the saturation specific
humidity are computed from a weighted mean of the latent heats of vaporization and
sublimation, with the weights given by the fractions of condensates `q.liq/(q.liq + q.ice)`
and `q.ice/(q.liq + q.ice)` that are liquid and ice, respectively.

If the `PhasePartition` `q` is not given, or has zero liquid and ice specific humidities,
the saturation specific humidity is that over a mixture of liquid and ice, with the
fraction of liquid given by temperature dependent `liquid_fraction(T)` and the
fraction of ice by the complement `1 - liquid_fraction(T)`.
"""
function q_vap_saturation(
    param_set::APS,
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}

    _LH_v0::FT = LH_v0(param_set)
    _LH_s0::FT = LH_s0(param_set)
    _cp_v::FT = cp_v(param_set)
    _cp_l::FT = cp_l(param_set)
    _cp_i::FT = cp_i(param_set)
    # get phase partitioning
    _liquid_frac = liquid_fraction(param_set, T, q)
    _ice_frac = 1 - _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0 = _liquid_frac * _LH_v0 + _ice_frac * _LH_s0
    Δcp = _liquid_frac * (_cp_v - _cp_l) + _ice_frac * (_cp_v - _cp_i)

    # saturation vapor pressure over possible mixture of liquid and ice
    p_v_sat = saturation_vapor_pressure(param_set, T, LH_0, Δcp)

    return q_vap_saturation_from_pressure(param_set, T, ρ, p_v_sat)

end

"""
    q_vap_saturation(ts::ThermodynamicState)

Compute the saturation specific humidity, given a thermodynamic state `ts`.
"""
q_vap_saturation(ts::ThermodynamicState) = q_vap_saturation(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
)

"""
    q_vap_saturation_from_pressure(param_set, T, ρ, p_v_sat)

Compute the saturation specific humidity, given

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature,
 - `ρ` (moist-)air density
 - `p_v_sat` saturation vapor pressure
"""
function q_vap_saturation_from_pressure(
    param_set::APS,
    T::FT,
    ρ::FT,
    p_v_sat::FT,
) where {FT <: Real}
    _R_v::FT = R_v(param_set)
    return p_v_sat / (ρ * _R_v * T)
end

"""
    saturation_excess(param_set, T, ρ, q::PhasePartition)

The saturation excess in equilibrium where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
 - `q` [`PhasePartition`](@ref)

The saturation excess is the difference between the total specific humidity `q.tot`
and the saturation specific humidity in equilibrium, and it is defined to be
nonzero only if this difference is positive.
"""
function saturation_excess(
    param_set::APS,
    T::FT,
    ρ::FT,
    q::PhasePartition{FT},
) where {FT <: Real}
    return max(0, q.tot - q_vap_saturation(param_set, T, ρ, q))
end

"""
    saturation_excess(ts::ThermodynamicState)

Compute the saturation excess in equilibrium,
given a thermodynamic state `ts`.
"""
saturation_excess(ts::ThermodynamicState) = saturation_excess(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
)

"""
    liquid_fraction(param_set, T[, q::PhasePartition])

The fraction of condensate that is liquid where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `q` [`PhasePartition`](@ref)

If `q.liq` or `q.ice` are nonzero, the liquid fraction is computed from
them.

Otherwise, phase equilibrium is assumed so that the fraction of liquid
is a function that is 1 above `T_freeze` and goes to zero below `T_freeze`.
"""
function liquid_fraction(
    param_set::APS,
    T::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    _T_freeze::FT = T_freeze(param_set)
    q_c = q.liq + q.ice     # condensate specific humidity
    if q_c > 0
        return q.liq / q_c
    else
        # For now: Heaviside function for partitioning into liquid and ice: all liquid
        # for T > T_freeze; all ice for T <= T_freeze
        return FT(T > _T_freeze)
    end
end

"""
    liquid_fraction(ts::ThermodynamicState)

The fraction of condensate that is liquid given a thermodynamic state `ts`.
"""
liquid_fraction(ts::ThermodynamicState) =
    liquid_fraction(ts.param_set, air_temperature(ts), PhasePartition(ts))

"""
    PhasePartition_equil(param_set, T, ρ, q_tot)

Partition the phases in equilibrium, returning a [`PhasePartition`](@ref) object using the
[`liquid_fraction`](@ref) function where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity

The residual `q.tot - q.liq - q.ice` is the vapor specific humidity.
"""
function PhasePartition_equil(
    param_set::APS,
    T::FT,
    ρ::FT,
    q_tot::FT,
) where {FT <: Real}
    _liquid_frac = liquid_fraction(param_set, T)                      # fraction of condensate that is liquid
    q_c = saturation_excess(param_set, T, ρ, PhasePartition(q_tot)) # condensate specific humidity
    q_liq = _liquid_frac * q_c                             # liquid specific humidity
    q_ice = (1 - _liquid_frac) * q_c                       # ice specific humidity

    return PhasePartition(q_tot, q_liq, q_ice)
end

PhasePartition_equil(ts::PhaseNonEquil) = PhasePartition_equil(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    total_specific_humidity(ts),
)

PhasePartition(ts::PhaseDry{FT}) where {FT <: Real} = q_pt_0(FT)
PhasePartition(ts::PhaseEquil) = PhasePartition_equil(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    total_specific_humidity(ts),
)
PhasePartition(ts::PhaseNonEquil) = ts.q

function ∂e_int_∂T(
    param_set::APS,
    T::FT,
    e_int::FT,
    ρ::FT,
    q_tot::FT,
) where {FT <: Real}
    _LH_v0::FT = LH_v0(param_set)
    _LH_s0::FT = LH_s0(param_set)
    _R_v::FT = R_v(param_set)
    _T_0::FT = T_0(param_set)
    _cv_v::FT = cv_v(param_set)
    _cv_l::FT = cv_l(param_set)
    _cv_i::FT = cv_i(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)

    cvm = cv_m(param_set, PhasePartition_equil(param_set, T, ρ, q_tot))
    q_vap_sat = q_vap_saturation(param_set, T, ρ)
    λ = liquid_fraction(param_set, T)
    L = λ * _LH_v0 + (1 - λ) * _LH_s0
    ∂q_vap_sat_∂T = q_vap_sat * L / (_R_v * T^2)
    dcvm_dq_vap = _cv_v - λ * _cv_l - (1 - λ) * _cv_i
    return cvm +
           (_e_int_v0 + (1 - λ) * _e_int_i0 + (T - _T_0) * dcvm_dq_vap) *
           ∂q_vap_sat_∂T
end

"""
    saturation_adjustment(param_set, e_int, ρ, q_tot)

Compute the temperature that is consistent with

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `e_int` internal energy
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
 - `tol` absolute tolerance for saturation adjustment iterations. Can be one of:
    - `SolutionTolerance()` to stop when `|x_2 - x_1| < tol`
    - `ResidualTolerance()` to stop when `|f(x)| < tol`
 - `maxiter` maximum iterations for non-linear equation solve

by finding the root of

``e_int - internal_energy_sat(param_set, T, ρ, q_tot) = 0``

using Newtons method with analytic gradients.

See also [`saturation_adjustment`](@ref).
"""
function saturation_adjustment(
    param_set::APS,
    e_int::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int,
    tol::AbstractTolerance,
) where {FT <: Real}
    _T_min::FT = T_min(param_set)

    T_1 = max(_T_min, air_temperature(param_set, e_int, PhasePartition(q_tot))) # Assume all vapor
    q_v_sat = q_vap_saturation(param_set, T_1, ρ)
    unsaturated = q_tot <= q_v_sat
    if unsaturated && T_1 > _T_min
        return T_1
    else
        sol = find_zero(
            T -> internal_energy_sat(param_set, T, ρ, q_tot) - e_int,
            NewtonsMethod(T_1, T_ -> ∂e_int_∂T(param_set, T_, e_int, ρ, q_tot)),
            CompactSolution(),
            tol,
            maxiter,
        )
        if !sol.converged
            error("saturation_adjustment did not converge")
        end
        return sol.root
    end
end

"""
    ΔT_min(::Type{FT})

Minimum interval for saturation adjustment using Secant method
"""
@inline ΔT_min(::Type{FT}) where {FT} = FT(3)

"""
    ΔT_max(::Type{FT})

Maximum interval for saturation adjustment using Secant method
"""
@inline ΔT_max(::Type{FT}) where {FT} = FT(10)

"""
    bound_upper_temperature(T_1::FT, T_2::FT) where {FT<:Real}

Bounds the upper temperature, `T_2`, for
saturation adjustment using Secant method
"""
@inline function bound_upper_temperature(T_1::FT, T_2::FT) where {FT <: Real}
    T_2 = max(T_1 + ΔT_min(FT), T_2)
    return min(T_1 + ΔT_max(FT), T_2)
end

"""
    saturation_adjustment_SecantMethod(param_set, e_int, ρ, q_tot)

Compute the temperature `T` that is consistent with

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `e_int` internal energy
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
 - `tol` absolute tolerance for saturation adjustment iterations. Can be one of:
    - `SolutionTolerance()` to stop when `|x_2 - x_1| < tol`
    - `ResidualTolerance()` to stop when `|f(x)| < tol`
 - `maxiter` maximum iterations for non-linear equation solve

by finding the root of

``e_int - internal_energy_sat(param_set, T, ρ, q_tot) = 0``

See also [`saturation_adjustment_q_tot_θ_liq_ice`](@ref).
"""
function saturation_adjustment_SecantMethod(
    param_set::APS,
    e_int::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int,
    tol::AbstractTolerance,
) where {FT <: Real}
    _T_min::FT = T_min(param_set)
    T_1 = max(_T_min, air_temperature(param_set, e_int, PhasePartition(q_tot))) # Assume all vapor
    q_v_sat = q_vap_saturation(param_set, T_1, ρ)
    unsaturated = q_tot <= q_v_sat
    if unsaturated && T_1 > _T_min
        return T_1
    else
        # FIXME here: need to revisit bounds for saturation adjustment to guarantee bracketing of zero.
        T_2 = air_temperature(
            param_set,
            e_int,
            PhasePartition(q_tot, FT(0), q_tot),
        ) # Assume all ice
        T_2 = bound_upper_temperature(T_1, T_2)
        sol = find_zero(
            T -> internal_energy_sat(param_set, T, ρ, q_tot) - e_int,
            SecantMethod(T_1, T_2),
            CompactSolution(),
            tol,
            maxiter,
        )
        if !sol.converged
            error("saturation_adjustment_SecantMethod did not converge")
        end
        return sol.root
    end
end

"""
    saturation_adjustment_q_tot_θ_liq_ice(param_set, θ_liq_ice, ρ, q_tot)

Compute the temperature `T` that is consistent with

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `θ_liq_ice` liquid-ice potential temperature
 - `q_tot` total specific humidity
 - `ρ` (moist-)air density
 - `tol` absolute tolerance for saturation adjustment iterations. Can be one of:
    - `SolutionTolerance()` to stop when `|x_2 - x_1| < tol`
    - `ResidualTolerance()` to stop when `|f(x)| < tol`
 - `maxiter` maximum iterations for non-linear equation solve

by finding the root of

``
  θ_{liq_ice} - liquid_ice_pottemp_sat(param_set, T, ρ, q_tot) = 0
``

See also [`saturation_adjustment`](@ref).
"""
function saturation_adjustment_q_tot_θ_liq_ice(
    param_set::APS,
    θ_liq_ice::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int,
    tol::AbstractTolerance,
) where {FT <: Real}
    _T_min::FT = T_min(param_set)
    T_1 = max(
        _T_min,
        air_temperature_from_liquid_ice_pottemp(
            param_set,
            θ_liq_ice,
            ρ,
            PhasePartition(q_tot),
        ),
    ) # Assume all vapor
    q_v_sat = q_vap_saturation(param_set, T_1, ρ)
    unsaturated = q_tot <= q_v_sat
    if unsaturated && T_1 > _T_min
        return T_1
    else
        T_2 = air_temperature_from_liquid_ice_pottemp(
            param_set,
            θ_liq_ice,
            ρ,
            PhasePartition(q_tot, FT(0), q_tot),
        ) # Assume all ice
        T_2 = bound_upper_temperature(T_1, T_2)
        sol = find_zero(
            T -> liquid_ice_pottemp_sat(param_set, T, ρ, q_tot) - θ_liq_ice,
            SecantMethod(T_1, T_2),
            CompactSolution(),
            tol,
            maxiter,
        )
        if !sol.converged
            error("saturation_adjustment_q_tot_θ_liq_ice did not converge")
        end
        return sol.root
    end
end

"""
    saturation_adjustment_q_tot_θ_liq_ice_given_pressure(param_set, θ_liq_ice, p, q_tot)

Compute the temperature `T` that is consistent with

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `θ_liq_ice` liquid-ice potential temperature
 - `q_tot` total specific humidity
 - `p` pressure
 - `tol` absolute tolerance for saturation adjustment iterations. Can be one of:
    - `SolutionTolerance()` to stop when `|x_2 - x_1| < tol`
    - `ResidualTolerance()` to stop when `|f(x)| < tol`
 - `maxiter` maximum iterations for non-linear equation solve

by finding the root of

``
  θ_{liq_ice} - liquid_ice_pottemp_sat(param_set, T, air_density(param_set, T, p, PhasePartition(q_tot)), q_tot) = 0
``

See also [`saturation_adjustment`](@ref).
"""
function saturation_adjustment_q_tot_θ_liq_ice_given_pressure(
    param_set::APS,
    θ_liq_ice::FT,
    p::FT,
    q_tot::FT,
    maxiter::Int,
    tol::AbstractTolerance,
) where {FT <: Real}
    _T_min::FT = T_min(param_set)
    T_1 = air_temperature_from_liquid_ice_pottemp_given_pressure(
        param_set,
        θ_liq_ice,
        p,
        PhasePartition(q_tot),
    ) # Assume all vapor
    ρ = air_density(param_set, T_1, p, PhasePartition(q_tot))
    q_v_sat = q_vap_saturation(param_set, T_1, ρ)
    unsaturated = q_tot <= q_v_sat
    if unsaturated && T_1 > _T_min
        return T_1
    else
        T_2 = air_temperature_from_liquid_ice_pottemp(
            param_set,
            θ_liq_ice,
            p,
            PhasePartition(q_tot, FT(0), q_tot),
        ) # Assume all ice
        T_2 = bound_upper_temperature(T_1, T_2)
        sol = find_zero(
            T ->
                liquid_ice_pottemp_sat(
                    param_set,
                    T,
                    air_density(param_set, T, p, PhasePartition(q_tot)),
                    q_tot,
                ) - θ_liq_ice,
            SecantMethod(T_1, T_2),
            CompactSolution(),
            tol,
            maxiter,
        )
        if !sol.converged
            error("saturation_adjustment_q_tot_θ_liq_ice_given_pressure did not converge")
        end
        return sol.root
    end
end

"""
    latent_heat_liq_ice(param_set, q::PhasePartition{FT})

Effective latent heat of condensate (weighted sum of liquid and ice),
with specific latent heat evaluated at reference temperature `T_0` given
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function latent_heat_liq_ice(
    param_set::APS,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    _LH_v0::FT = LH_v0(param_set)
    _LH_s0::FT = LH_s0(param_set)
    return _LH_v0 * q.liq + _LH_s0 * q.ice
end


"""
    liquid_ice_pottemp_given_pressure(param_set, T, p, q::PhasePartition)

The liquid-ice potential temperature where
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function liquid_ice_pottemp_given_pressure(
    param_set::APS,
    T::FT,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    # liquid-ice potential temperature, approximating latent heats
    # of phase transitions as constants
    return dry_pottemp_given_pressure(param_set, T, p, q) *
           (1 - latent_heat_liq_ice(param_set, q) / (cp_m(param_set, q) * T))
end


"""
    liquid_ice_pottemp(param_set, T, ρ, q::PhasePartition)

The liquid-ice potential temperature where
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function liquid_ice_pottemp(
    param_set::APS,
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    return liquid_ice_pottemp_given_pressure(
        param_set,
        T,
        air_pressure(param_set, T, ρ, q),
        q,
    )
end

"""
    liquid_ice_pottemp(ts::ThermodynamicState)

The liquid-ice potential temperature,
given a thermodynamic state `ts`.
"""
liquid_ice_pottemp(ts::ThermodynamicState) = liquid_ice_pottemp(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
)

"""
    dry_pottemp(param_set, T, ρ[, q::PhasePartition])

The dry potential temperature where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
 """
function dry_pottemp(
    param_set::APS,
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    return T / exner(param_set, T, ρ, q)
end

"""
    dry_pottemp_given_pressure(param_set, T, p[, q::PhasePartition])

The dry potential temperature where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
 """
function dry_pottemp_given_pressure(
    param_set::APS,
    T::FT,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    return T / exner_given_pressure(param_set, p, q)
end

"""
    dry_pottemp(ts::ThermodynamicState)

The dry potential temperature, given a thermodynamic state `ts`.
"""
dry_pottemp(ts::ThermodynamicState) = dry_pottemp(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
)

"""
    air_temperature_from_liquid_ice_pottemp(param_set, θ_liq_ice, ρ, q::PhasePartition)

The temperature given
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `θ_liq_ice` liquid-ice potential temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function air_temperature_from_liquid_ice_pottemp(
    param_set::APS,
    θ_liq_ice::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}

    _MSLP::FT = MSLP(param_set)
    cvm = cv_m(param_set, q)
    cpm = cp_m(param_set, q)
    R_m = gas_constant_air(param_set, q)
    κ = 1 - cvm / cpm
    T_u = (ρ * R_m * θ_liq_ice / _MSLP)^(R_m / cvm) * θ_liq_ice
    T_1 = latent_heat_liq_ice(param_set, q) / cvm
    T_2 = -κ / (2 * T_u) * (latent_heat_liq_ice(param_set, q) / cvm)^2
    return T_u + T_1 + T_2
end

"""
    air_temperature_from_liquid_ice_pottemp_non_linear(param_set, θ_liq_ice, ρ, q::PhasePartition)

Computes temperature `T` given

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `θ_liq_ice` liquid-ice potential temperature
 - `ρ` (moist-)air density
 - `tol` absolute tolerance for non-linear equation iterations. Can be one of:
    - `SolutionTolerance()` to stop when `|x_2 - x_1| < tol`
    - `ResidualTolerance()` to stop when `|f(x)| < tol`
 - `maxiter` maximum iterations for non-linear equation solve
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air,

by finding the root of
``
  T - air_temperature_from_liquid_ice_pottemp_given_pressure(param_set, θ_liq_ice, air_pressure(param_set, T, ρ, q), q) = 0
``
"""
function air_temperature_from_liquid_ice_pottemp_non_linear(
    param_set::APS,
    θ_liq_ice::FT,
    ρ::FT,
    maxiter::Int,
    tol::AbstractTolerance,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    _T_min::FT = T_min(param_set)
    _T_max::FT = T_max(param_set)
    sol = find_zero(
        T ->
            T - air_temperature_from_liquid_ice_pottemp_given_pressure(
                param_set,
                θ_liq_ice,
                air_pressure(param_set, T, ρ, q),
                q,
            ),
        SecantMethod(_T_min, _T_max),
        CompactSolution(),
        tol,
        maxiter,
    )
    if !sol.converged
        error("air_temperature_from_liquid_ice_pottemp_non_linear did not converge")
    end
    return sol.root
end

"""
    air_temperature_from_liquid_ice_pottemp_given_pressure(param_set, θ_liq_ice, p[, q::PhasePartition])

The air temperature where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `θ_liq_ice` liquid-ice potential temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function air_temperature_from_liquid_ice_pottemp_given_pressure(
    param_set::APS,
    θ_liq_ice::FT,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    return θ_liq_ice * exner_given_pressure(param_set, p, q) +
           latent_heat_liq_ice(param_set, q) / cp_m(param_set, q)
end

"""
    virtual_pottemp(param_set, T, ρ[, q::PhasePartition])

The virtual temperature where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function virtual_pottemp(
    param_set::APS,
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    _R_d::FT = R_d(param_set)
    return gas_constant_air(param_set, q) / _R_d *
           dry_pottemp(param_set, T, ρ, q)
end

"""
    virtual_pottemp(ts::ThermodynamicState)

The virtual potential temperature,
given a thermodynamic state `ts`.
"""
virtual_pottemp(ts::ThermodynamicState) = virtual_pottemp(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
)

"""
    liquid_ice_pottemp_sat(param_set, T, ρ[, q::PhasePartition])

The saturated liquid ice potential temperature where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function liquid_ice_pottemp_sat(
    param_set::APS,
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    q_v_sat = q_vap_saturation(param_set, T, ρ, q)
    return liquid_ice_pottemp(param_set, T, ρ, PhasePartition(q_v_sat))
end

"""
    liquid_ice_pottemp_sat(param_set, T, ρ, q_tot)

The saturated liquid ice potential temperature where

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
"""
function liquid_ice_pottemp_sat(
    param_set::APS,
    T::FT,
    ρ::FT,
    q_tot::FT,
) where {FT <: Real}
    return liquid_ice_pottemp(
        param_set,
        T,
        ρ,
        PhasePartition_equil(param_set, T, ρ, q_tot),
    )
end

"""
    liquid_ice_pottemp_sat(ts::ThermodynamicState)

The liquid potential temperature given a thermodynamic state `ts`.
"""
liquid_ice_pottemp_sat(ts::ThermodynamicState) = liquid_ice_pottemp_sat(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
)

"""
    exner_given_pressure(param_set, p[, q::PhasePartition])

The Exner function where
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function exner_given_pressure(
    param_set::APS,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    _MSLP::FT = MSLP(param_set)
    # gas constant and isobaric specific heat of moist air
    _R_m = gas_constant_air(param_set, q)
    _cp_m = cp_m(param_set, q)

    return (p / _MSLP)^(_R_m / _cp_m)
end

"""
    exner(param_set, T, ρ[, q::PhasePartition)])

The Exner function where
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function exner(
    param_set::APS,
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    return exner_given_pressure(param_set, air_pressure(param_set, T, ρ, q), q)
end

"""
    exner(ts::ThermodynamicState)

The Exner function, given a thermodynamic state `ts`.
"""
exner(ts::ThermodynamicState) = exner(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
)

"""
    relative_humidity(param_set, T, p, e_int, q::PhasePartition)

The relative humidity, given
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `p` pressure
 - `e_int` internal energy per unit mass
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
 """
function relative_humidity(
    param_set::APS,
    T::FT,
    p::FT,
    e_int::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT <: Real}
    _R_v::FT = R_v(param_set)
    q_vap = q.tot - q.liq - q.ice
    p_vap =
        q_vap *
        air_density(param_set, T, p, q) *
        _R_v *
        air_temperature(param_set, e_int, q)
    liq_frac = liquid_fraction(param_set, T, q)
    p_vap_sat =
        liq_frac * saturation_vapor_pressure(param_set, T, Liquid()) +
        (1 - liq_frac) * saturation_vapor_pressure(param_set, T, Ice())
    return p_vap / p_vap_sat
end

"""
    relative_humidity(ts::ThermodynamicState)

The relative humidity, given a thermodynamic state `ts`.
"""
relative_humidity(ts::ThermodynamicState{FT}) where {FT <: Real} =
    relative_humidity(
        ts.param_set,
        air_temperature(ts),
        air_pressure(ts),
        internal_energy(ts),
        PhasePartition(ts),
    )
