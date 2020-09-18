export GCMModel, NoGCM, HadGEM

abstract type GCMModel end

vars_state(::GCMModel, ::Prognostic, FT) = @vars()
vars_state(::GCMModel, ::Auxiliary, FT) = @vars()
vars_state(::GCMModel, ::UpwardIntegrals, FT) = @vars()
vars_state(::GCMModel, ::DownwardIntegrals, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::GCMModel,
    ::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function integral_set_auxiliary_state!(
    ::GCMModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function integral_load_auxiliary_state!(::GCMModel, aux::Vars, integ::Vars) end
function reverse_integral_set_auxiliary_state!(
    ::GCMModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function reverse_integral_load_auxiliary_state!(
    ::GCMModel,
    aux::Vars,
    integ::Vars,
) end
function flux_radiation!(
    ::GCMModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end

struct NoGCM <: GCMModel end

struct HadGEM <: GCMModel end
vars_state(m::HadGEM, ::Auxiliary, FT) = @vars(
    ρ::FT,
    pfull::FT,
    ta::FT,
    hus::FT,
    ua::FT,
    va::FT,
    tntha::FT,
    tntva::FT,
    tntr::FT,
    tnhusha::FT,
    tnhusva::FT,
    wap::FT,
)
