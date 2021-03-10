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
The function which computes the freeze/thaw source term for Richard's equation,
assuming the timescale is the maximum of the thermal timescale and the timestep.
"""
Base.@kwdef struct PhaseChange{FT} <: SoilSource{FT}
    "Timestep"
    Δt::FT = FT(NaN)
    "Timescale for temperature changes"
    τLTE::FT = FT(NaN)
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

    _ρliq = FT(ρ_cloud_liq(land.param_set))
    _ρice = FT(ρ_cloud_ice(land.param_set))
    _Tfreeze = FT(T_freeze(land.param_set))
    _LH_f0 = FT(LH_f0(land.param_set))
    _g = FT(grav(land.param_set))

    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    eff_porosity = land.soil.param_functions.porosity - θ_i
    θ_l = volumetric_liquid_fraction(ϑ_l, eff_porosity)
    T = get_temperature(land.soil.heat, aux, t)

    # The inverse matric potential is only defined for negative arguments.
    # This is calculated even when T > _Tfreeze, so to be safe,
    # take absolute value and then pick the sign to be negative.
    # But below, use heaviside function to only allow freezing in the
    # appropriate temperature range (T < _Tfreeze)
    ψ = -abs(_LH_f0 / _g / _Tfreeze * (T - _Tfreeze))
    hydraulics = land.soil.water.hydraulics
    θstar =
        land.soil.param_functions.porosity *
        inverse_matric_potential(hydraulics, ψ)

    τft = max(source_type.Δt, source_type.τLTE)
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

function land_source!(
    source_type::Root,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
    plant_hs:: # hs = hydraulic system define
)
    FT = eltype(state)

    _ρliq = FT(ρ_cloud_liq(land.param_set))
    _ρice = FT(ρ_cloud_ice(land.param_set))
    _Tfreeze = FT(T_freeze(land.param_set))
    _LH_f0 = FT(LH_f0(land.param_set))
    _g = FT(grav(land.param_set))

    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    eff_porosity = land.soil.param_functions.porosity - θ_i
    θ_l = volumetric_liquid_fraction(ϑ_l, eff_porosity)
    T = get_temperature(land.soil.heat, aux, t)

    ψ = -abs(_LH_f0 / _g / _Tfreeze * (T - _Tfreeze))
    hydraulics = land.soil.water.hydraulics
    θstar =
        land.soil.param_functions.porosity *
        inverse_matric_potential(hydraulics, ψ)


   # pass liquid soil water content
    function  root_extraction(
        plant_hs::AbstractPlantOrganism{FT},
        matric_potential::Array{FT,1},  # using a ; means we would have to set qsum, now we dont, can leave blank as we gave a defaul
        qsum::FT=FT(0), #(sum of flow rates, no water going into canopy by default), can leave qsum blank
    ) where {FT}

        
        for i_root in eachindex(plant_hs.roots)
            plant_hs.roots[i_root].p_ups = p_soil_array[i_root];
        end

        roots_flow!(plant_hs, qsum) # no water going into canopy (no daytime transpiration), this updates the flow rate in each root
        root_extraction = plant_hs.cache_q # array of flow rates in each root layer (mol W/s/layer), maybe not necessary to rewrite name

        return root_extraction

    end



#     function pressure_head(
#     model::AbstractHydraulicsModel{FT},
#     porosity::FT,
#     S_s::FT,
#     ϑ_l::FT,
#     θ_i::FT,
# ) where {FT}
#     eff_porosity = porosity - θ_i
#     S_l_eff = effective_saturation(eff_porosity, ϑ_l)
#     if S_l_eff < 1
#         S_l = effective_saturation(porosity, ϑ_l)
#         ψ = matric_potential(model, S_l)
#     else
#         ψ = (ϑ_l - eff_porosity) / S_s
#     end
#     return ψ
# end

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
