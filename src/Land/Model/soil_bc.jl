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
)

end




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

    ϑ_l, θ_i = get_water_content(land.soil.water, aux⁻, state⁻, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
    ρc_s = volumetric_heat_capacity(
        θ_l,
        θ_i,
        land.soil.param_functions.ρc_ds,
        land.param_set,
    )

    ρe_int_bc = volumetric_internal_energy(
        θ_i,
        ρc_s,
        bc_function(aux⁻, t),
        land.param_set,
    )

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
    # on faces at the top and bottom of the domain,
    # the vector n̂ can point in either ± ẑ, depending on which side of the domain
    # we are on. The user supplies a scalar flux F, such that the flux at the boundary
    # is assumed to be F⃗ = F ẑ - i.e. the user doesn't need to worry about the normal vector.
    # so, we take the absolute value of n̂ here.
    # The minus sign is because the condition is applied on minus the flux.
    # the same argument applies for the other directions.
    diff⁺.soil.water.K∇h = abs.(n̂) * (-bc_function(aux⁻, t))
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
    # on faces at the top and bottom of the domain,
    # the vector n̂ can point in either ± ẑ, depending on which side of the domain
    # we are on. The user supplies a scalar flux F, such that the flux at the boundary
    # is assumed to be F⃗ = F ẑ - i.e. the user doesn't need to worry about the normal vector.
    # so, we take the absolute value of n̂ here.
    # The minus sign is because the condition is applied on minus the flux.
    # the same argument applies for the other directions.
    diff⁺.soil.heat.κ∇T = abs.(n̂) * (-bc_function(aux⁻, t))
end


# SurfaceDriven conditions for SoilWater

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
    FT = eltype(state⁻)
    incident_water_flux = bc.precip_model(t)
    
    #This is not quite correct.
    if incident_water_flux < -norm(diff⁻.soil.water.K∇h)
        nothing
    else
        diff⁺.soil.water.K∇h = n̂ *  (-FT(2)*incident_water_flux) - diff⁻.soil.water.K∇h
    end
end


function soil_boundary_state!(
    nf,
    bc::SurfaceDrivenWaterBoundaryConditions,
    water::SoilWaterModel,
    land::LandModel,
    state⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    #This is not quite correct.
    FT = eltype(state⁻)
    bc_value = land.soil.param_functions.porosity - state⁻.soil.water.θ_i
    state⁺.soil.water.ϑ_l = bc_value

end
