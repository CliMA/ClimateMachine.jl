export AbstractEnergyModel, EnergyModel, θModel
#### Energy component in atmosphere model
abstract type AbstractEnergyModel end
struct EnergyModel <: AbstractEnergyModel end
struct θModel <: AbstractEnergyModel end

vars_state(::EnergyModel, ::AbstractStateType, FT) = @vars()
vars_state(::EnergyModel, ::Prognostic, FT) = @vars(ρe::FT)
vars_state(::EnergyModel, ::Gradient, FT) = @vars(h_tot::FT)
vars_state(::EnergyModel, ::GradientFlux, FT) = @vars(∇h_tot::SVector{3, FT})
vars_state(::θModel, ::AbstractStateType, FT) = @vars()
vars_state(::θModel, ::Prognostic, FT) = @vars(ρθ_liq_ice::FT)
vars_state(::θModel, ::Gradient, FT) = @vars(θ_liq_ice::FT)
vars_state(::θModel, ::GradientFlux, FT) = @vars(∇θ_liq_ice::SVector{3, FT})

function compute_gradient_argument!(
    energy::EnergyModel,
    atmos::AtmosModel,
    transform,
    state,
    aux,
    t,
)
    ts = recover_thermo_state(atmos, state, aux)
    e_tot = state.energy.ρe * (1 / state.ρ)
    transform.energy.h_tot = total_specific_enthalpy(ts, e_tot)
end

function compute_gradient_argument!(
    energy::θModel,
    ::AtmosModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.energy.θ_liq_ice = state.energy.ρθ_liq_ice / state.ρ
end

compute_gradient_flux!(::EnergyModel, _...) = nothing

function compute_gradient_flux!(::θModel, diffusive, ∇transform, state, aux, t)
    diffusive.energy.∇θ_liq_ice = ∇transform.energy.θ_liq_ice
end

function compute_gradient_flux!(
    ::EnergyModel,
    diffusive,
    ∇transform,
    state,
    aux,
    t,
)
    diffusive.energy.∇h_tot = ∇transform.energy.h_tot
end
