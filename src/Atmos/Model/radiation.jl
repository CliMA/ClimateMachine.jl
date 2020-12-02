export RadiationModel, NoRadiation

abstract type RadiationModel end

vars_state(::RadiationModel, ::AbstractStateType, FT) = @vars()

eq_tends(pv::PV, ::RadiationModel, ::Flux{FirstOrder}) where {PV} = ()

function atmos_nodal_update_auxiliary_state!(
    ::RadiationModel,
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function integral_set_auxiliary_state!(::RadiationModel, integ::Vars, aux::Vars) end
function integral_load_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function reverse_integral_set_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    aux::Vars,
) end
function reverse_integral_load_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function flux_first_order!(
    ::RadiationModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    ts,
    direction,
) end

struct NoRadiation <: RadiationModel end
