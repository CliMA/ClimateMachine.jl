using CLIMA.PlanetParameters
export RadiationModel, NoRadiation

abstract type RadiationModel end

vars_state(::RadiationModel, FT) = @vars()
vars_aux(::RadiationModel, FT) = @vars()
vars_integrals(::RadiationModel, FT) = @vars()
vars_reverse_integrals(::RadiationModel, FT) = @vars()

function atmos_nodal_update_aux!(::RadiationModel, ::AtmosModel, state::Vars, aux::Vars, t::Real) end
function preodefun!(::RadiationModel, aux::Vars, state::Vars, t::Real) end
function integral_set_aux!(::RadiationModel, integ::Vars, state::Vars, aux::Vars) end
function integral_load_aux!(::RadiationModel, aux::Vars, integ::Vars) end
function reverse_integral_set_aux!(::RadiationModel, integ::Vars, state::Vars, aux::Vars) end
function reverse_integral_load_aux!(::RadiationModel, aux::Vars, integ::Vars) end
function flux_radiation!(::RadiationModel, atmos::AtmosModel, flux::Grad, state::Vars, aux::Vars, t::Real) end

struct NoRadiation <: RadiationModel
end
