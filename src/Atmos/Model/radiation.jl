using CLIMA.PlanetParameters
export NoRadiation

abstract type RadiationModel end

vars_state(::RadiationModel, FT) = @vars()
vars_aux(::RadiationModel, FT) = @vars()
vars_integrals(::RadiationModel, FT) = @vars()

function atmos_nodal_update_aux!(::RadiationModel, ::AtmosModel, state::Vars, aux::Vars, t::Real) end
function preodefun!(::RadiationModel, aux::Vars, state::Vars, t::Real) end
function integrate_aux!(::RadiationModel, integ::Vars, state::Vars, aux::Vars) end
function flux_radiation!(::RadiationModel, flux::Grad, state::Vars, aux::Vars, t::Real) end

struct NoRadiation <: RadiationModel
end

