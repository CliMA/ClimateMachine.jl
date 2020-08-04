export RadiationModel, NoRadiation

abstract type RadiationModel end

vars_state(::RadiationModel, ::AbstractStateType, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::RadiationModel,
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function integral_set_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    aux::Vars,
    ::UpwardIntegrals,
) end
function integral_load_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
    ::UpwardIntegrals,
) end
function integral_set_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    aux::Vars,
    ::DownwardIntegrals,
) end
function integral_load_auxiliary_state!(
    ::RadiationModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
    ::DownwardIntegrals,
) end
function flux_radiation!(
    ::RadiationModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end

struct NoRadiation <: RadiationModel end
