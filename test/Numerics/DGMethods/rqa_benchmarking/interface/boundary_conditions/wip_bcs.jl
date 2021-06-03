

function boundary_flux.ρu(state, aux, parameters, n̂)
    # project and reflect
    state⁺.ρu = state⁻.ρu - dot(state⁻.ρu,  n̂) .* n̂ - dot(state⁻.ρu,  n̂) .* n̂    
end


# For tracers 
function calc_boundary_flux!(flux, state, aux, n̂, parameters, boundary_flux::BoundaryFlux, tracer_symbol::Val{:ρ})
    flux.ρ = boundary_flux.ρ(state, aux, parameters)
end

function calc_boundary_flux!(flux, state, aux, n̂, parameters, boundary_flux::BoundaryFlux, tracer_symbol::Val{:ρe})
    flux.ρe = boundary_flux.ρe(state, aux, parameters)
end

function calc_boundary_flux!(flux, state, aux, n̂, parameters, boundary_flux::BoundaryFlux, tracer_symbol::Val{:ρq})
    flux.ρq = boundary_flux.ρq(state, aux, parameters)
end

# For vectors 
function calc_boundary_flux!(flux, state, aux, n̂, parameters, boundary_flux::BoundaryFlux, vector_symbol::Val{:ρu})
    flux.ρu = boundary_flux.ρu(state, aux, parameters, n̂)
end

# Balance law interface component
function numerical_boundary_flux_second_order!(numerical_flux, bc::Flux, model, fluxᵀn, state, args)
    for symbol in (Val(:ρ), Val(:ρu), Val(:ρe), Val(:ρq))
        calc_boundary_flux!(fluxᵀn, state, aux, n̂, parameters, boundary_flux, symbol)
    end
end

#import this to extend it !!!!!!!!!!!!!!!!!!!!
function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    bctype::Impenetrable{FreeSlip},
    balance_law::DryAtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {S, A}
# I think S is a tuple of symbols
    state⁺.ρ = state⁻.ρ
    state⁺.ρq = state⁻.ρq
    state⁺.ρe = state⁻.ρe
    ρu⁻ = state⁻.ρu
    state⁺.ρu = ρu⁻ -  n⁻ ⋅ ρu⁻ .* SVector(n⁻) -  n⁻ ⋅ ρu⁻ .* SVector(n⁻)

    numerical_flux_first_order!(
        numerical_flux,
        balance_law,
        fluxᵀn,
        normal_vector,
        state⁻,
        aux⁻,
        state⁺,
        aux⁺,
        t,
        direction,
    )
end

# function boundary_state!(
#     nmf::NumericalFluxFirstOrder,
#     ::Val{6},
#     model::DryAtmosModel,
#     state⁺,
#     aux⁺,
#     n,
#     state⁻,
#     aux⁻,
#     _...,
# )
#     #  flux =  (flux_first_order(state⁺) + flux_first_order(state⁻)) / 2 + dissipation(state⁺, state⁻) 
#     # if dissipation = rusanov then dissipation(state⁺, state⁻) = c/2 * (state⁺ - state⁻)
#     # if dissipation = roe then 
    
#     # state⁺.ρu = - state⁻.ρu #  no slip boundary conditions
#     # dot(state⁺.ρu, n) * n = -dot(state⁻.ρu, n) * n # for free slip

#     # physics = model.physics
#     # eos = model.physics.eos
#     # calc_boundary_state(nmf, bctype, model)

#     state⁺.ρ = state⁻.ρ   # if no penetration then this is no flux on the boundary
#     state⁺.ρq = state⁻.ρq # if no penetration then this is no flux on the boundary
#     state⁺.ρe = state⁻.ρe # if pressure⁺ = pressure⁻ & no penetration then this is no flux boundary condition
#     aux⁺.Φ = aux⁻.Φ       # 

#     # state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n) # (I - 2* n n') is a reflection operator
#     # first subtract off the normal component, then go further to enact the reflection principle
#     state⁺.ρu =  ( state⁻.ρu - dot(state⁻.ρu, n) .* SVector(n) ) - dot(state⁻.ρu, n) .* SVector(n)

# end

# """
#     BulkFormulaEnergy(fn) :: EnergyBC
# Calculate the net inward energy flux across the boundary. The drag
# coefficient is `C_h = fn_C_h(atmos, state, aux, t, normu_int_tan)`. The surface
# temperature and q_tot are `T, q_tot = fn_T_and_q_tot(atmos, state, aux, t)`.
# Return the flux (in W m^-2).
# """
# struct BulkFormulaEnergy{FNX, FNTM} <: EnergyBC
#     fn_C_h::FNX
#     fn_T_and_q_tot::FNTM
# end
# function atmos_energy_boundary_state!(
#     nf,
#     bc_energy::BulkFormulaEnergy,
#     atmos,
#     _...,
# ) end
# function atmos_energy_normal_boundary_flux_second_order!(
#     nf,
#     bc_energy::BulkFormulaEnergy,
#     atmos,
#     fluxᵀn,
#     args,
# )
#     @unpack state⁻, aux⁻, t, state_int⁻, aux_int⁻ = args

#     u_int⁻ = state_int⁻.ρu / state_int⁻.ρ
#     u_int⁻_tan = projection_tangential(atmos, aux_int⁻, u_int⁻)
#     normu_int⁻_tan = norm(u_int⁻_tan)
#     C_h = bc_energy.fn_C_h(atmos, state⁻, aux⁻, t, normu_int⁻_tan)
#     T, q_tot = bc_energy.fn_T_and_q_tot(atmos, state⁻, aux⁻, t)
#     param_set = parameter_set(atmos)

#     # calculate MSE from the states at the boundary and at the interior point
#     ts = PhaseEquil_ρTq(param_set, state⁻.ρ, T, q_tot)
#     ts_int = recover_thermo_state(atmos, state_int⁻, aux_int⁻)
#     e_pot = gravitational_potential(atmos.orientation, aux⁻)
#     e_pot_int = gravitational_potential(atmos.orientation, aux_int⁻)
#     MSE = moist_static_energy(ts, e_pot)
#     MSE_int = moist_static_energy(ts_int, e_pot_int)

#     # TODO: use the correct density at the surface
#     ρ_avg = average_density(state⁻.ρ, state_int⁻.ρ)
#     # NOTE: difference from design docs since normal points outwards
#     fluxᵀn.energy.ρe -= C_h * ρ_avg * normu_int⁻_tan * (MSE - MSE_int)
# end

# function boundary_state!(nf, bc, model, state, args)
#     # loop
#     calc_thing!(state, nf, bc, physics)
# end

# function numerical_boundary_flux_second_order!(nf, bc::Flux, model, fluxᵀn, state, args)
#     # loop
#     calc_other_thing!(fluxᵀn, nf, bc, state, physics)
# end

# function calc_other_thing!(fluxᵀn, nf, bc::Flux, state⁻, aux⁻, physics)
#     fluxᵀn = bc.flux_function(state⁻, aux⁻, physics)
# end

# function calc_other_thing!(fluxᵀn, nf::WalKlub, bc, state, aux, physics)
#     ρ = state.ρ
#     ρu = state.ρu
#     ρq = state.ρq

#     u = ρu / ρ
#     q = ρq / ρ
#     u⟂ = tangential_magic(u, aux)
#     u_norm = norm(u⟂)

#     # obtain drag coefficients
#     Cₕ = bc.drag_coefficient_temperature(state, aux)
#     Cₑ = bc.drag_coefficient_moisture(state, aux)

#     # obtain surface fields
#     T_sfc, q_tot_sfc = bc.surface_fields(atmos, state, aux, t)

#     # surface cooling due to wind via transport of dry energy (sensible heat flux)
#     c_p = calc_c_p(...)
#     T   = calc_air_temperature(...)
#     H   = ρ * Cₕ * u_norm * c_p * (T - T_sfc)

#     # surface cooling due to wind via transport of moisture (latent energy flux)
#     L_v = calc_L_v(...)
#     E   = ρ * Cₗ * u_norm * L_v * (q - q_sfc)

#     fluxᵀn.ρ  -= E / L_v # ??! the atmosphere gains mass
#     fluxᵀn.ρe -= H + E   # if the sfc loses energy, the atmosphere gains energy
#     fluxᵀn.ρq -= E / L_v # the atmosphere gets more humid
# end