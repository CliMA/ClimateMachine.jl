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
        bctype::Impenetrable{FreeSlip},
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
    
    # Apply drag law using the tangential velocity as energy flux
    # unpack
    ρ = state⁻.ρ
    ρu = state⁻.ρu
    ρq = state⁻.ρq
    eos = model.physics.eos
    parameters = model.physics.parameters
    LH_v0 = model.physics.parameters.LH_v0

    # obtain surface fields
    ϕ = lat(aux⁻.x, aux⁻.y, aux⁻.z)
    Cₕ = bctype.drag_coef_temperature(parameters, ϕ)
    Cₑ = bctype.drag_coef_moisture(parameters, ϕ)
    T_sfc = bctype.temperature(parameters, ϕ)

    u = ρu / ρ
    q = ρq / ρ

    # magnitude of tangential velocity (usually called speed)
    speed_tangential = norm((I - n̂ ⊗ n̂) * u)

    # saturation specific humidity
    #q_tot_sfc = calc_saturation_specific_humidity(eos, state⁻, aux⁻, parameters)
    pₜᵣ      = get_planet_parameter(:press_triple) 
    R_v      = get_planet_parameter(:R_v)
    Tₜᵣ      = get_planet_parameter(:T_triple)
    T_0      = get_planet_parameter(:T_0)
    cp_v     = get_planet_parameter(:cp_v)
    cp_l     = get_planet_parameter(:cp_l)
    Δcp = cp_v - cp_l
    pᵥₛ = pₜᵣ * (T_sfc / Tₜᵣ)^(Δcp / R_v) * exp((LH_v0 - Δcp * T_0) / R_v * (1 / Tₜᵣ - 1 / T_sfc))
    q_tot_sfc = pᵥₛ / (ρ * R_v * T_sfc)
       
    # surface cooling due to wind via transport of dry energy (sensible heat flux)
    cp = calc_cp(eos, state⁻, parameters)
    T = calc_air_temperature(eos, state⁻, aux⁻, parameters)
    H = ρ * Cₕ * speed_tangential * cp * (T - T_sfc)

    # surface cooling due to wind via transport of moisture (latent energy flux)
    E = 0.01 * ρ * Cₑ * speed_tangential * LH_v0 * (q - q_tot_sfc)

    #fluxᵀn.ρ = -E / LH_v0 
    #fluxᵀn.ρu += E / LH_v0 .* u
    fluxᵀn.ρe = E + H
    fluxᵀn.ρq = E / LH_v0
end