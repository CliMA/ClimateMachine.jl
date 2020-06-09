export GCMModel, NoGCM, HadGem2

abstract type GCMModel end

vars_state_conservative(::GCMModel, FT) = @vars()
vars_state_auxiliary(::GCMModel, FT) = @vars()
vars_integrals(::GCMModel, FT) = @vars()
vars_reverse_integrals(::GCMModel, FT) = @vars()

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
function integral_load_auxiliary_state!(
    ::GCMModel,
    aux::Vars,
    integ::Vars,
) end
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

struct HadGem2 <: GCMModel end
vars_state_auxiliary(m::HadGem2, FT) = @vars(
    ρ::FT,
    p::FT,
    ta::FT,
    ρe::FT,
    ρq_tot::FT,
    ua::FT,
    va::FT,
    tntha::FT,
    tntva::FT,
    tntr::FT,
    tnhusha::FT,
    tnhusva::FT,
    wap::FT,
    T::FT
)
