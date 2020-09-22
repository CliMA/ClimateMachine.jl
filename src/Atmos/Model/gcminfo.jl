export GCMModel, NoGCM, HadGEM

abstract type GCMModel end

vars_state(::GCMModel, ::Prognostic, FT) = @vars()
vars_state(::GCMModel, ::Auxiliary, FT) = @vars()
vars_state(::GCMModel, ::Gradient, FT) = @vars()
vars_state(::GCMModel, ::GradientFlux, FT) = @vars()
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
@inline function atmos_nodal_update_auxiliary_state!(
    gcminfo::GCMModel,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function compute_gradient_argument!(
    gcminfo::GCMModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function compute_gradient_flux!(
    gcminfo::GCMModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function flux_first_order_gcm!(
    gcminfo::GCMModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function flux_second_order!(
    gcminfo::GCMModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    nothing
end
function flux_second_order!(gcminfo::GCMModel, flux::Grad, state::Vars, d_q_tot)
    nothing
end

struct NoGCM <: GCMModel end


"""
    Container for GCM variables from HadGEM2-A forcing,
used in the AMIP(0-4K) experiments
"""
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

vars_state(::HadGEM, ::Prognostic, FT) = @vars()
vars_state(::HadGEM, ::Gradient, FT) = @vars(ta::FT, hus::FT)
vars_state(::HadGEM, ::GradientFlux, FT) =
    @vars(∇ta::SVector{3, FT}, ∇hus::SVector{3, FT})

@inline function atmos_nodal_update_auxiliary_state!(
    gcminfo::HadGEM,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function compute_gradient_argument!(
    gcminfo::HadGEM,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.gcminfo.ta = aux.gcminfo.ta
    transform.gcminfo.hus = aux.gcminfo.hus
end

function compute_gradient_flux!(
    gcminfo::HadGEM,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.gcminfo.∇ta = ∇transform.gcminfo.ta
    diffusive.gcminfo.∇hus = ∇transform.gcminfo.hus
end

function flux_first_order_gcm!(
    gcminfo::HadGEM,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function flux_second_order!(
    gcminfo::HadGEM,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    nothing
end
function flux_second_order!(gcminfo::HadGEM, flux::Grad, state::Vars, d_q_tot)
    nothing
end
