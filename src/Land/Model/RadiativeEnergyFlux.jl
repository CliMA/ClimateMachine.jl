module RadiativeEnergyFlux

using ...VariableTemplates
using DocStringExtensions

export AbstractNetSwFluxModel,
    PrescribedSwFluxAndAlbedo,
    PrescribedNetSwFlux,
    compute_net_sw_flux,
    compute_net_radiative_energy_flux

"""
   AbstractNetSwFluxModel{FT <: AbstractFloat}
"""
abstract type AbstractNetSwFluxModel{FT <: AbstractFloat} end

"""
    PrescribedSwFluxAndAlbedo{FT, FN1, FN2} <: AbstractNetSwFluxModel{FT}

Structure which contains functions for shortwave albedo and
shortwave fluxes. They are user-defined, constant across the domain
and can be a function of time.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct PrescribedSwFluxAndAlbedo{FT, FN1, FN2} <: AbstractNetSwFluxModel{FT}
    "Shortwave albedo in grid. No units."
    α::FN1
    "Shortwave flux in grid. Units of J m-2 s-1"
    swf::FN2
    function PrescribedSwFluxAndAlbedo(
        ::Type{FT};
        α::FN1 = (t) -> FT(NaN),
        swf::FN2 = (t) -> FT(NaN),
    ) where {FT, FN1, FN2}
        args = (α, swf)
        return new{FT, FN1, FN2}(args...)
    end
end

"""
    PrescribedNetSwFlux{FT,FN3} <: AbstractNetSwFluxModel{FT}

Structure which contains net shortwave flux values. The function is
user-defined, constant across the domain and can be a function of time.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct PrescribedNetSwFlux{FT, FN3} <: AbstractNetSwFluxModel{FT}
    "Net shortwave flux. Units of J m-2 s-1"
    nswf::FN3
    function PrescribedNetSwFlux(
        ::Type{FT};
        nswf::FN3 = (t) -> FT(NaN),
    ) where {FT, FN3}
        return new{FT, FN3}(nswf)
    end
end

"""
    compute_net_sw_flux(
        nswf_model::PrescribedSwFluxAndAlbedo{FT},
        t::Real,
    ) where {FT}

Computes the net shortwave flux as a function of time.
"""
function compute_net_sw_flux(
    nswf_model::PrescribedSwFluxAndAlbedo{FT},
    t::Real,
) where {FT}
    net_sw_flux = (FT(1) - nswf_model.α(t)) * nswf_model.swf(t)
    return net_sw_flux
end

"""
    compute_net_sw_flux(
        nswf_model:: PrescribedNetSwFlux{FT},
        t::Real
    ) where {FT}

Computes the net shortwave flux as a function of time.
"""
function compute_net_sw_flux(
    nswf_model::PrescribedNetSwFlux{FT},
    t::Real,
) where {FT}
    net_sw_flux = nswf_model.nswf(t)
    return net_sw_flux
end

"""
    compute_net_radiative_energy_flux(
        nswf_model::AbstractNetSwFluxModel{FT},
        t::Real
    )

Returns the net radiative energy flux as a function of time. Will
include long wave fluxes in future version.
"""
function compute_net_radiative_energy_flux(
    nswf_model::AbstractNetSwFluxModel{FT},
    t::Real,
) where {FT}
    net_radiative_flux = compute_net_sw_flux(nswf_model, t)
    return net_radiative_flux
end

end
