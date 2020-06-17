#### Land sources
using CLIMAParameters.Planet: ρ_cloud_liq
using CLIMAParameters.Planet: ρ_cloud_ice
using CLIMAParameters.Planet: T_freeze

export FreezeThaw

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
    ::Nothing,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) end



abstract type Source end
"""
    FreezeThaw <: Source

The function which computes the freeze/thaw source term for Richard's equation.
"""
struct FreezeThaw <: Source end

function land_source!(
    ::FreezeThaw,
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
    τft = land.soil.param_functions.τft

    ϑ_l, θ_ice = get_water_content(aux, state, t, land.soil.water)
    θ_l = volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
    T = get_temperature(land.soil.heat, aux, t, state)

    freeze_thaw = 1.0/τft *(_ρliq*θ_l*heaviside(_Tfreeze - T) -
                            _ρice*θ_ice*heaviside(T - _Tfreeze))

    source.soil.water.ϑ_l -= freeze_thaw/_ρliq
    source.soil.water.θ_ice += freeze_thaw/_ρice
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
