# General case - to be used with bc::NoBC or m::PrescribedXModels
function soil_boundary_flux!(
    nf,
    bc::AbstractBoundaryConditions,
    m::AbstractSoilComponentModel,
    land::LandModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
) end




function soil_boundary_state!(
    nf,
    bc::AbstractBoundaryConditions,
    m::AbstractSoilComponentModel,
    land::LandModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)

end

# Dirichlet Methods for SoilHeat and SoilWater

"""
    function soil_boundary_state!(
        nf,
        bc::Dirichlet,
        water::SoilWaterModel,
        land::LandModel,
        state⁺::Vars,
        aux⁺::Vars,
        nM,
        state⁻::Vars,
        aux⁻::Vars,
        t,
    )

The Dirichlet-type method for `soil_boundary_state!` for the
`SoilWaterModel`.
"""
function soil_boundary_state!(
    nf,
    bc::Dirichlet,
    water::SoilWaterModel,
    land::LandModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    bc_function = bc.state_bc
    state⁺.soil.water.ϑ_l = bc_function(aux⁻, t)

end


"""
    function soil_boundary_state!(
        nf,
        bc::Dirichlet,
        heat::SoilHeatModel,
        land::LandModel,
        state⁺::Vars,
        aux⁺::Vars,
        nM,
        state⁻::Vars,
        aux⁻::Vars,
        t,
    )

The Dirichlet-type method for `soil_boundary_state!` for the
`SoilHeatModel`.
"""
function soil_boundary_state!(
    nf,
    bc::Dirichlet,
    heat::SoilHeatModel,
    land::LandModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    bc_function = bc.state_bc
    param_set = parameter_set(land)
    ϑ_l, θ_i = get_water_content(land.soil.water, aux⁻, state⁻, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
    ρc_s = volumetric_heat_capacity(
        θ_l,
        θ_i,
        land.soil.param_functions.ρc_ds,
        param_set,
    )

    ρe_int_bc =
        volumetric_internal_energy(θ_i, ρc_s, bc_function(aux⁻, t), param_set)

    state⁺.soil.heat.ρe_int = ρe_int_bc
end

# Neumann conditions for SoilHeat and SoilWater

"""
    function soil_boundary_flux!(
        nf,
        bc::Neumann,
        water::SoilWaterModel,
        land::LandModel,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        t,
    )

The Neumann method for `soil_boundary_flux!` for the
`SoilWaterModel`.
"""
function soil_boundary_flux!(
    nf,
    bc::Neumann,
    water::SoilWaterModel,
    land::LandModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    bc_function = bc.scalar_flux_bc
    # See documentation for `Neumann`. The user supplies f = scalar_flux_bc
    # such that F⃗ = -K∇h = -f n̂. 
    # Since the BC is on K∇h, no minus sign is needed.
    diff⁺.soil.water.K∇h = n̂ * bc_function(aux⁻, t)
end


"""
    function soil_boundary_flux!(
        nf,
        bc::Neumann,
        heat::SoilHeatModel,
        land::LandModel,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        t,
    )

The Neumann method for `soil_boundary_flux!` for the
`SoilHeatModel`.
"""
function soil_boundary_flux!(
    nf,
    bc::Neumann,
    heat::SoilHeatModel,
    land::LandModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    bc_function = bc.scalar_flux_bc
    # See documentation for `Neumann`. The user supplies f = scalar_flux_bc
    # such that F⃗ = -κ∇T = -f n̂. 
    # Since the BC is on κ∇T, no minus sign is needed.
    diff⁺.soil.heat.κ∇T = n̂ * bc_function(aux⁻, t)
end


# SurfaceDriven conditions for for SoilHeat and SoilWater

"""
    function soil_boundary_flux!(
        nf,
        bc::SurfaceDrivenWaterBoundaryConditions,
        water::SoilWaterModel,
        land::LandModel,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        t,
    )

The Surface Driven BC method for `soil_boundary_flux!` for the
`SoilWaterModel`.
"""
function soil_boundary_flux!(
    nf,
    bc::SurfaceDrivenWaterBoundaryConditions,
    water::SoilWaterModel,
    land::LandModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)

    diff⁺.soil.water.K∇h = compute_surface_grad_bc(
        land.soil,
        bc.runoff_model,
        bc.precip_model,
        n̂,
        state⁻,
        diff⁻,
        aux⁻,
        t,
    )

end

"""
    soil_boundary_flux!(
        nf,
        bc::SurfaceDrivenHeatBoundaryConditions,
        heat::SoilHeatModel,
        land::LandModel,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        t,
    )

The Surface Driven BC method for `soil_boundary_flux!` for the
`SoilHeatModel`.
"""
function soil_boundary_flux!(
    nf,
    bc::SurfaceDrivenHeatBoundaryConditions,
    heat::SoilHeatModel,
    land::LandModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)

    net_surface_flux = compute_net_radiative_energy_flux(bc.nswf_model, t)
    diff⁺.soil.heat.κ∇T = n̂ * (-net_surface_flux)
end
