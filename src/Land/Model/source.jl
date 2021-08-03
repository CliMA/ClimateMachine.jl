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


"""
    PhaseChange <: TendencyDef{Source}
The function which computes the freeze/thaw source term for Richard's equation.
"""
Base.@kwdef struct PhaseChange{FT} <: TendencyDef{Source}
    "Typical resolution in the vertical"
    Δz::FT = FT(NaN)
    "O(1) factor multiplying the thermal time"
    γ_LTE::FT = FT(1)
end

prognostic_vars(::PhaseChange) =
    (VolumetricLiquidFraction(), VolumetricIceFraction())

function precompute(land::LandModel, args, tt::Source)
    dtup = DispatchedSet(map(land.source) do s
        (s, precompute(s, land, args, tt))
    end)
    return (; dtup)
end

function precompute(source_type::PhaseChange, land::LandModel, args, tt::Source)
    @unpack state, diffusive, aux, t, direction = args

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

    θ_r = land.soil.param_functions.water.θ_r(aux)

    hydraulics = land.soil.water.hydraulics(aux)
    θ_m = min(_ρice * θ_i / _ρliq + θ_l, ν)
    ψ0 = matric_potential(hydraulics, θ_m)
    ψT = _LH_f0 / _g / _Tfreeze * (T - _Tfreeze)
    if T < _Tfreeze
        θstar = θ_r + (ν - θ_r) * inverse_matric_potential(hydraulics, ψ0 + ψT)
    else
        θstar = θ_l
    end
    
    Δz = source_type.Δz
    γ_LTE = source_type.γ_LTE
    τLTE = FT(ρc_s * Δz^2 / κ)*γ_LTE

    freeze_thaw =
        FT(1) / τLTE * (
            _ρliq *
            (θ_l - θstar) *
            heaviside(_Tfreeze - T) *
            heaviside(θ_l - θstar) - _ρice * θ_i * heaviside(T - _Tfreeze)
        )
    return (; freeze_thaw)
end

function source(
    ::VolumetricLiquidFraction,
    s::PhaseChange,
    land::LandModel,
    args,
)
    @unpack state = args
    @unpack freeze_thaw = args.precomputed.dtup[s]
    FT = eltype(state)
    _ρliq = FT(ρ_cloud_liq(land.param_set))
    return -freeze_thaw / _ρliq
end

function source(::VolumetricIceFraction, s::PhaseChange, land::LandModel, args)
    @unpack state = args
    @unpack freeze_thaw = args.precomputed.dtup[s]
    FT = eltype(state)
    _ρice = FT(ρ_cloud_ice(land.param_set))
    return freeze_thaw / _ρice
end
