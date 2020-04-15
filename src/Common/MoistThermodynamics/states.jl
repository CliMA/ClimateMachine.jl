export PhasePartition
# Thermodynamic states
export ThermodynamicState,
    PhaseDry,
    PhaseDry_given_pT,
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
struct PhasePartition{FT <: Real}
    "total specific humidity"
    tot::FT
    "liquid water specific humidity (default: `0`)"
    liq::FT
    "ice specific humidity (default: `0`)"
    ice::FT
end

PhasePartition(q_tot::FT, q_liq::FT) where {FT <: Real} =
    PhasePartition(q_tot, q_liq, zero(FT))
PhasePartition(q_tot::FT) where {FT <: Real} =
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
struct PhaseEquil{FT, PS} <: ThermodynamicState{FT}
    "parameter set (e.g., planet parameters)"
    param_set::PS
    "internal energy"
    e_int::FT
    "density of air (potentially moist)"
    ρ::FT
    "total specific humidity"
    q_tot::FT
    "temperature: computed via [`saturation_adjustment`](@ref)"
    T::FT
end
function PhaseEquil(
    param_set::APS,
    e_int::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int = 3,
    tol::FT = FT(1e-1),
    sat_adjust::Function = saturation_adjustment,
) where {FT <: Real}
    # TODO: Remove these safety nets, or at least add warnings
    # waiting on fix: github.com/vchuravy/GPUifyLoops.jl/issues/104
    q_tot_safe = clamp(q_tot, FT(0), FT(1))
    T = sat_adjust(param_set, e_int, ρ, q_tot_safe, maxiter, tol)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot_safe, T)
end

"""
    PhaseDry{FT} <: ThermodynamicState

A dry thermodynamic state (`q_tot = 0`).

# Constructors

    PhaseDry(e_int, ρ)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseDry{FT, PS} <: ThermodynamicState{FT}
    "parameter set (e.g., planet parameters)"
    param_set::PS
    "internal energy"
    e_int::FT
    "density of dry air"
    ρ::FT
end
PhaseDry(param_set::APS, e_int::FT, ρ::FT) where {FT} =
    PhaseDry{FT, typeof(param_set)}(param_set, e_int, ρ)

"""
    PhaseDry_given_pT(p, T)

Constructs a [`PhaseDry`](@ref) thermodynamic state from:

 - `p` pressure
 - `T` temperature
"""
function PhaseDry_given_pT(
    param_set::APS,
    p::FT,
    T::FT,
) where {FT <: Real}
    e_int = internal_energy(param_set, T)
    ρ = air_density(param_set, T, p)
    return PhaseDry{FT, typeof(param_set)}(param_set, e_int, ρ)
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
function LiquidIcePotTempSHumEquil(
    param_set::APS,
    θ_liq_ice::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int = 30,
    tol::FT = FT(1e-1),
) where {FT <: Real}
    T = saturation_adjustment_q_tot_θ_liq_ice(
        param_set,
        θ_liq_ice,
        ρ,
        q_tot,
        maxiter,
        tol,
    )
    q_pt = PhasePartition_equil(param_set, T, ρ, q_tot)
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot, T)
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
function LiquidIcePotTempSHumEquil_given_pressure(
    param_set::APS,
    θ_liq_ice::FT,
    p::FT,
    q_tot::FT,
    maxiter::Int = 30,
    tol::FT = FT(1e-1),
) where {FT <: Real}
    T = saturation_adjustment_q_tot_θ_liq_ice_given_pressure(
        param_set,
        θ_liq_ice,
        p,
        q_tot,
        maxiter,
        tol,
    )
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q.tot, T)
end

"""
    TemperatureSHumEquil(T, p, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from temperature.

 - `T` temperature
 - `p` pressure
 - `q_tot` total specific humidity
"""
function TemperatureSHumEquil(
    param_set::APS,
    T::FT,
    p::FT,
    q_tot::FT,
) where {FT <: Real}
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot, T)
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
struct PhaseNonEquil{FT, PS} <: ThermodynamicState{FT}
    "parameter set (e.g., planet parameters)"
    param_set::PS
    "internal energy"
    e_int::FT
    "density of air (potentially moist)"
    ρ::FT
    "phase partition"
    q::PhasePartition{FT}
end
function PhaseNonEquil(
    param_set::APS,
    e_int::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT}
    return PhaseNonEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q)
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
function LiquidIcePotTempSHumNonEquil(
    param_set::APS,
    θ_liq_ice::FT,
    ρ::FT,
    q_pt::PhasePartition{FT},
    maxiter::Int = 5,
    tol::FT = FT(1e-1),
) where {FT <: Real}
    T = air_temperature_from_liquid_ice_pottemp_non_linear(
        param_set,
        θ_liq_ice,
        ρ,
        maxiter,
        tol,
        q_pt,
    )
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseNonEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_pt)
end

"""
    LiquidIcePotTempSHumNonEquil_given_pressure(θ_liq_ice, p, q_pt)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `θ_liq_ice` liquid-ice potential temperature
 - `p` pressure
 - `q_pt` phase partition
"""
function LiquidIcePotTempSHumNonEquil_given_pressure(
    param_set::APS,
    θ_liq_ice::FT,
    p::FT,
    q_pt::PhasePartition{FT},
) where {FT <: Real}
    T = air_temperature_from_liquid_ice_pottemp_given_pressure(
        param_set,
        θ_liq_ice,
        p,
        q_pt,
    )
    ρ = air_density(param_set, T, p, q_pt)
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseNonEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_pt)
end

"""
    fixed_lapse_rate_ref_state(z::FT,
                               T_surface::FT,
                               T_min::FT) where {FT<:AbstractFloat}

Fixed lapse rate hydrostatic reference state
"""
function fixed_lapse_rate_ref_state(
    param_set::APS,
    z::FT,
    T_surface::FT,
    T_min::FT,
) where {FT <: AbstractFloat}
    _grav::FT = grav(param_set)
    _cp_d::FT = cp_d(param_set)
    _R_d::FT = R_d(param_set)
    _MSLP::FT = MSLP(param_set)
    Γ = _grav / _cp_d
    z_tropopause = (T_surface - T_min) / Γ
    H_min = _R_d * T_min / _grav
    T = max(T_surface - Γ * z, T_min)
    p = _MSLP * (T / T_surface)^(_grav / (_R_d * Γ))
    T == T_min && (p = p * exp(-(z - z_tropopause) / H_min))
    ρ = p / (_R_d * T)
    return T, p, ρ
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
function tested_convergence_range(
    param_set::APS,
    n::Int,
    ::Type{FT},
) where {FT}
    n_RS1 = 10
    n_RS2 = 20
    n_RS = n_RS1 + n_RS2
    z_range = range(FT(0), stop = FT(2.5e4), length = n)
    relative_sat1 = range(FT(0), stop = FT(1), length = n_RS1)
    relative_sat2 = range(FT(1), stop = FT(1.02), length = n_RS2)
    relative_sat = [relative_sat1..., relative_sat2...]
    T_min = FT(150)
    T_surface = FT(350)

    args =
        fixed_lapse_rate_ref_state.(
            Ref(param_set),
            z_range,
            Ref(T_surface),
            Ref(T_min),
        )
    T, p, ρ = getindex.(args, 1), getindex.(args, 2), getindex.(args, 3)

    p = collect(Iterators.flatten([p for RS in 1:n_RS]))
    ρ = collect(Iterators.flatten([ρ for RS in 1:n_RS]))
    T = collect(Iterators.flatten([T for RS in 1:n_RS]))
    relative_sat = collect(Iterators.flatten([relative_sat for RS in 1:n]))

    # Additional variables
    q_sat = q_vap_saturation.(Ref(param_set), T, ρ)
    q_tot = min.(relative_sat .* q_sat, FT(1))
    q_pt = PhasePartition_equil.(Ref(param_set), T, ρ, q_tot)
    e_int = internal_energy.(Ref(param_set), T, q_pt)
    θ_liq_ice = liquid_ice_pottemp.(Ref(param_set), T, ρ, q_pt)
    return e_int, ρ, q_tot, q_pt, T, p, θ_liq_ice
end
