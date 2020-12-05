export LSForcingModel, NoLSForcing, HadGEMVertical, CMIP_cfsite_Vertical

abstract type LSForcingModel end

vars_state(::LSForcingModel, ::AbstractStateType, FT) = @vars()

function compute_gradient_argument!(
    lsforcing::LSForcingModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

function compute_gradient_flux!(
    lsforcing::LSForcingModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end





"""
    NoLSForcing <: LSForcingModel
No large-scale forcing
"""
struct NoLSForcing <: LSForcingModel end






"""
    Container for GCM variables from HadGEM2-A forcing,
used in the AMIP experiments
"""
struct HadGEMVertical <: LSForcingModel end

vars_state(m::HadGEMVertical, ::Auxiliary, FT) = @vars(
    ta::FT,
    hus::FT,
    ua::FT,
    va::FT,
    tntΣhava::FT,
    Σtemp_tendency::FT,
    Σqt_tendency::FT,
    w_s::FT,
)

vars_state(::HadGEMVertical, ::Gradient, FT) = @vars(ta::FT, hus::FT)
vars_state(::HadGEMVertical, ::GradientFlux, FT) = @vars(∇ᵥta::FT, ∇ᵥhus::FT)

function compute_gradient_argument!(
    lsforcing::HadGEMVertical,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.lsforcing.ta = aux.lsforcing.ta
    transform.lsforcing.hus = aux.lsforcing.hus
end

function compute_gradient_flux!(
    lsforcing::HadGEMVertical,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.lsforcing.∇ᵥta = ∇transform.lsforcing.ta[3]
    diffusive.lsforcing.∇ᵥhus = ∇transform.lsforcing.hus[3]
end





"""
    Container for GCM variables from models for forcing my cfsite runs,
used in the AMIP(0-4K) experiments
"""
struct CMIP_cfsite_Vertical <: LSForcingModel end

vars_state(m::CMIP_cfsite_Vertical, ::Auxiliary, FT) = @vars(
    ta::FT,
    hus::FT,
    ua::FT,
    va::FT,
    tntΣhava::FT,
    Σtemp_tendency::FT,
    Σqt_tendency::FT,
    w_s::FT,
    ρ::FT,
    ρe::FT,
    ρq_tot::FT,
    ρq_rai::FT,
    ρq_sno::FT, # ion think this one is used yet but future proof my g

)

vars_state(::CMIP_cfsite_Vertical, ::Gradient, FT) = @vars(ta::FT, hus::FT)
vars_state(::CMIP_cfsite_Vertical, ::GradientFlux, FT) = @vars(∇ᵥta::FT, ∇ᵥhus::FT)

function compute_gradient_argument!(
    lsforcing::CMIP_cfsite_Vertical,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.lsforcing.ta = aux.lsforcing.ta
    transform.lsforcing.hus = aux.lsforcing.hus
end

function compute_gradient_flux!(
    lsforcing::CMIP_cfsite_Vertical,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.lsforcing.∇ᵥta = ∇transform.lsforcing.ta[3]
    diffusive.lsforcing.∇ᵥhus = ∇transform.lsforcing.hus[3]
end

