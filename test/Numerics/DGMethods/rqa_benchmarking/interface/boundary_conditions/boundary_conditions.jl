abstract type AbstractBoundaryCondition end

struct DefaultBC <: AbstractBoundaryCondition end

Base.@kwdef struct BulkFormulaTemperature{𝒯,𝒰,𝒱} <: AbstractBoundaryCondition 
  drag_coef_temperature::𝒯
  drag_coef_moisture::𝒰
  surface_temperature::𝒱
end

function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    ::DefaultBC,
    balance_law::DryAtmosModel,
    fluxᵀn::Vars{S},
    n̂::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {S, A}
    state⁺.ρ = state⁻.ρ
    state⁺.ρe = state⁻.ρe
    state⁺.ρq = state⁻.ρq

    ρu⁻ = state⁻.ρu
    
    # project and reflect for impenetrable condition, but 
    # leave tangential component untouched
    state⁺.ρu = ρu⁻ - n̂ ⋅ ρu⁻ .* SVector(n̂) - n̂ ⋅ ρu⁻ .* SVector(n̂)
    numerical_flux_first_order!(
      numerical_flux,
      balance_law,
      fluxᵀn,
      n̂,
      state⁻,
      aux⁻,
      state⁺,
      aux⁺,
      t,
      direction,
    )
end

function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    bctype::BulkFormulaTemperature,
    model::DryAtmosModel,
    fluxᵀn::Vars{S},
    n̂::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {S, A}
    # Impenetrable free-slip condition to reflect and project momentum 
    # at the boundary
    numerical_boundary_flux_first_order!(
        numerical_flux,
        bctype::DefaultBC,
        model,
        fluxᵀn,
        n̂,
        state⁻,
        aux⁻,
        state⁺,
        aux⁺,
        t,
        direction,
        state1⁻,
        aux1⁻,
    )
    
    # Apply bulks laws using the tangential velocity as energy flux
    ρ = state⁻.ρ
    ρu = state⁻.ρu
    ρq = state⁻.ρq
    eos = model.physics.eos
    parameters = model.physics.parameters
    LH_v0 = model.physics.parameters.LH_v0

    # obtain surface fields from bcs
    ϕ = lat(aux⁻.x, aux⁻.y, aux⁻.z)
    Cₕ = bctype.drag_coef_temperature(parameters, ϕ)
    Cₑ = bctype.drag_coef_moisture(parameters, ϕ)
    T_sfc = bctype.temperature(parameters, ϕ)

    # magnitude of tangential velocity (usually called speed)
    u = ρu / ρ
    speed_tangential = norm((I - n̂ ⊗ n̂) * u)
       
    # sensible heat flux
    cp = calc_cp(eos, state⁻, parameters)
    T = calc_air_temperature(eos, state⁻, aux⁻, parameters)
    H = ρ * Cₕ * speed_tangential * cp * (T - T_sfc)

    # latent heat flux
    q = ρq / ρ
    q_tot_sfc  = calc_saturation_specific_humidity(ρ, T_sfc, parameters) 
    E = ρ * Cₑ * speed_tangential * LH_v0 * (q - q_tot_sfc)

    fluxᵀn.ρ  = E / LH_v0 
    fluxᵀn.ρe = E + H
    fluxᵀn.ρq = E / LH_v0
end