export AbstractEnergyModel, EnergyModel, θModel
#### Energy component in atmosphere model
abstract type AbstractEnergyModel <: BalanceLaw end
struct EnergyModel end
struct θModel end

# TODO: do we need this?
# import ..Thermodynamics: total_energy
# function total_energy(state, ts)
#     e_pot = gravitational_potential(orientation, aux)
# end

vars_state(::EnergyModel, ::Prognostic, FT) = @vars(ρe::FT)
vars_state(::EnergyModel, ::Gradient, FT) = @vars(h_tot::FT)
vars_state(::EnergyModel, ::GradientFlux, FT) = @vars(∇h_tot::SVector{3, FT})
vars_state(::θModel, ::Prognostic, FT) = @vars(ρθ_liq_ice::FT)
vars_state(::θModel, ::Gradient, FT) = @vars(θ_liq_ice::FT)
vars_state(::θModel, ::GradientFlux, FT) = @vars(∇θ_liq_ice::SVector{3, FT})

function compute_gradient_argument!(atmos::AtmosModel, energy::EnergyModel, transform, state, aux, t)
    ts = recover_thermo_state(atmos, state, aux)
    e_tot = state.energy.ρe * (1 / state.ρ)
    transform.h_tot = total_specific_enthalpy(ts, e_tot)
end

function compute_gradient_argument!(
    ::AtmosModel,
    energy::θModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.energy.θ_liq_ice = state.energy.ρθ_liq_ice / state.ρ
end

compute_gradient_flux!(::EnergyModel, _...) = nothing

function compute_gradient_flux!(
    ::θModel,
    diffusive,
    ∇transform,
    state,
    aux,
    t,
)
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


function flux_first_order!(
    energy::EnergyModel,
    atmos::AtmosModel,
    flux::Grad,
    args,
)
    tend = Flux{FirstOrder}()
    flux.energy.ρe =
        Σfluxes(eq_tends(Energy(), atmos, tend), atmos, args)
end

function flux_first_order!(
    energy::θModel,
    atmos::AtmosModel,
    flux::Grad,
    args,
)
    tend = Flux{FirstOrder}()
    flux.energy.ρθ_liq_ice =
        Σfluxes(eq_tends(ρθ_liq_ice(), atmos, tend), atmos, args)
end

function flux_second_order!(
    energy::EnergyModel,
    flux::Grad,
    atmos::AtmosModel,
    args,
)
    tend = Flux{SecondOrder}()
    flux.energy.ρe =
        Σfluxes(eq_tends(Energy(), atmos, tend), atmos, args)
end

function flux_second_order!(
    energy::θModel,
    flux::Grad,
    atmos::AtmosModel,
    args,
)
    tend = Flux{SecondOrder}()
    flux.energy.ρθ_liq_ice =
        Σfluxes(eq_tends(ρθ_liq_ice(), atmos, tend), atmos, args)
end

function source!(energy::EnergyModel, source::Vars, atmos::AtmosModel, args)
    tend = Source()
    source.energy.ρe =
        Σsources(eq_tends(Energy(), atmos, tend), atmos, args)
end

function source!(energy::θModel, source::Vars, atmos::AtmosModel, args)
    tend = Source()
    source.energy.ρθ_liq_ice =
        Σsources(eq_tends(ρθ_liq_ice(), atmos, tend), atmos, args)
end

