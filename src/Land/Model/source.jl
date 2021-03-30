#### Land sources
export PhaseChange

function heaviside(x::FT) where {FT}
    if x >= FT(0)
        output = FT(1)
    else
        output = FT(0)
    end
    return output
end


abstract type SoilSource{FT <: AbstractFloat} end

"""
    PhaseChange <: SoilSource
The function which computes the freeze/thaw source term for Richard's equation.
"""
Base.@kwdef struct PhaseChange{FT} <: SoilSource{FT}
    "Typical resolution in the vertical"
    Δz::FT = FT(NaN)
end



function land_source!(
    f::Function,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    f(land, source, state, diffusive, aux, t, direction)
end

function land_source!(
    source_type::PhaseChange,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)

    param_set = parameter_set(land)
    _ρliq = FT(ρ_cloud_liq(param_set))
    _ρice = FT(ρ_cloud_ice(param_set))
    _Tfreeze = FT(T_freeze(param_set))
    _LH_f0 = FT(LH_f0(param_set))
    _g = FT(grav(param_set))



    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    ν = land.soil.param_functions.porosity
    eff_porosity = ν - θ_i
    θ_l = volumetric_liquid_fraction(ϑ_l, eff_porosity)
    T = get_temperature(land.soil.heat, aux, t)
    κ_dry = k_dry(land.param_set, land.soil.param_functions)
    S_r = relative_saturation(θ_l, θ_i, ν)
    kersten = kersten_number(θ_i, S_r, land.soil.param_functions)
    κ_sat = saturated_thermal_conductivity(
        θ_l,
        θ_i,
        land.soil.param_functions.κ_sat_unfrozen,
        land.soil.param_functions.κ_sat_frozen,
    )
    κ = thermal_conductivity(κ_dry, kersten, κ_sat)
    ρc_ds = land.soil.param_functions.ρc_ds
    ρc_s = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, land.param_set)
    θ_r = land.soil.param_functions.θ_r

    hydraulics = land.soil.water.hydraulics
    θ_m = min(_ρice * θ_i / _ρliq + θ_l, ν)
    ψ0 = matric_potential(hydraulics, θ_m)
    ψT = _LH_f0 / _g / _Tfreeze * (T - _Tfreeze)
    if T < _Tfreeze
        θstar = θ_r + (ν - θ_r) * inverse_matric_potential(hydraulics, ψ0 + ψT)
    else
        θstar = θ_l
    end
    Δz = source_type.Δz
    τLTE = FT(ρc_s * Δz^2 / κ)
    ΔT = norm(diffusive.soil.heat.κ∇T) / κ * Δz
    ρ_w = FT(0.5) * (_ρliq + _ρice)

    τpt = τLTE * (ρ_w * _LH_f0 * (ν - θ_r)) / (ρc_s * ΔT)

    τft = max(τLTE, τpt)
    freeze_thaw =
        FT(1) / τft * (
            _ρliq *
            (θ_l - θstar) *
            heaviside(_Tfreeze - T) *
            heaviside(θ_l - θstar) - _ρice * θ_i * heaviside(T - _Tfreeze)
        )
    source.soil.water.ϑ_l -= freeze_thaw / _ρliq
    source.soil.water.θ_i += freeze_thaw / _ρice
end


# sources are applied additively

@generated function land_source!(
    stuple::Tuple,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    N = fieldcount(stuple)
    return quote
        Base.Cartesian.@nexprs $N i -> land_source!(
            stuple[i],
            land,
            source,
            state,
            diffusive,
            aux,
            t,
            direction,
        )
        return nothing
    end
end
