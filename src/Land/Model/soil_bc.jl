# General case - applies to PrescribedHeatModel or PrescribedWaterModel.
# We can't dispach on boundaries at this point because Prescribed models
# do not have that field.
function soil_boundary_state!(
    nf,
    bctype,
    land::LandModel,
    soil::SoilModel,
    m::AbstractSoilComponentModel,
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
    bctype,
    land::LandModel,
    soil::SoilModel,
    m::AbstractSoilComponentModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)

end


# SoilWaterModel methods

"""
    function soil_boundary_state!(
        nf,
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        state⁺::Vars,
        aux⁺::Vars,
        nM,
        state⁻::Vars,
        aux⁻::Vars,
        bctype,
        t,
    )

The Dirichlet-type method for `soil_boundary_state!` for the
`SoilWaterModel`.
"""
function soil_boundary_state!(
    nf,
    bctype,
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    execute_based_on_boundaries!(
        nf,
        land,
        soil,
        water,
        water.boundaries,
        state⁺,
        aux⁺,
        nM,
        state⁻,
        aux⁻,
        bctype,
        t,
    )
end


"""
    function soil_boundary_state!(
        nf,
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        bctype,
        t,
    )

The Neumann-type method for `soil_boundary_state!` for the
`SoilWaterModel`.
"""
function soil_boundary_state!(
    nf,
    bctype,
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
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
    execute_based_on_boundaries!(
        nf,
        land,
        soil,
        water,
        water.boundaries,
        state⁺,
        diff⁺,
        aux⁺,
        n̂,
        state⁻,
        diff⁻,
        aux⁻,
        bctype,
        t,
    )
end


"""
    function execute_based_on_boundaries!(
        nf,
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        boundaries::SurfaceDrivenWaterBoundaryConditions,
        state⁺::Vars,
        aux⁺::Vars,
        nM,
        state⁻::Vars,
        aux⁻::Vars,
        bctype,
        t,
    )

The Dirichlet-type boundary method for `execute_based_on_boundaries!` for the
`SoilWaterModel` when used with `SurfaceDrivenWaterBoundaryConditions`. As
`SurfaceDrivenWaterBoundaryConditions` are Neumann, this does nothing.
"""
function execute_based_on_boundaries!(
    nf,
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    boundaries::SurfaceDrivenWaterBoundaryConditions,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
)
    nothing
end


"""
    function execute_based_on_boundaries!(
        nf,
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        boundaries::SurfaceDrivenWaterBoundaryConditions,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        bctype,
        t,
    )
 
The Neumann-type boundary method for `execute_based_on_boundaries!` for the
`SoilWaterModel` when used with `SurfaceDrivenWaterBoundaryConditions`. This applies
zero flux at the bottom of the domain, and a physical flux based on evaporation,
precipitation, and runoff at the top.
"""
function execute_based_on_boundaries!(
    nf,
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    boundaries::SurfaceDrivenWaterBoundaryConditions,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
)
    precip_model = boundaries.precip_model
    runoff_model = boundaries.runoff_model
    #compute surface flux
    net_surface_flux =
        compute_surface_flux(runoff_model, precip_model, state⁻, t)
    #top
    if bctype == 2
        diff⁺.soil.water.K∇h = n̂ * (-net_surface_flux)
        #bottom
    elseif bctype == 1
        diff⁺.soil.water.K∇h = n̂ * eltype(state⁻)(0.0)
    end


end

"""
    function execute_based_on_boundaries!(
        nf,
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        boundaries::GeneralBoundaryConditions,
        state⁺::Vars,
        aux⁺::Vars,
        nM,
        state⁻::Vars,
        aux⁻::Vars,
        bctype,
        t,
    )

The Dirichlet-type boundary method for `execute_based_on_boundaries!` for the
`SoilWaterModel` when used with `GeneralBoundaryConditions`. This applies the
user-supplied functions of space and time as Dirichlet conditions on `ϑ_l`.

Note that not supplying a function (so that it is `nothing`) results in no
boundary condition of this type applied.
"""
function execute_based_on_boundaries!(
    nf,
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    boundaries::GeneralBoundaryConditions,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
)

    bc = boundaries.dirichlet_bc
    if bctype == 2
        if bc.surface_state != nothing
            state⁺.soil.water.ϑ_l = bc.surface_state(aux⁻, t)
        else
            nothing
        end
    elseif bctype == 1
        if bc.bottom_state != nothing
            state⁺.soil.water.ϑ_l = bc.bottom_state(aux⁻, t)
        else
            nothing
        end
    end
end

"""
    function execute_based_on_boundaries!(
        nf,
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        boundaries::GeneralBoundaryConditions,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        bctype,
        t,
    )

The Neumann-type boundary method for `execute_based_on_boundaries!` for the
`SoilWaterModel` when used with `GeneralBoundaryConditions`. This applies the
user-supplied functions of space and time as Neumann conditions on the flux,
equal to -K∇h.

Note that not supplying a function (so that it is `nothing`) results in no
boundary condition of this type applied.
"""
function execute_based_on_boundaries!(
    nf,
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    boundaries::GeneralBoundaryConditions,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
)

    bc = boundaries.neumann_bc
    if bctype == 2
        if bc.surface_flux != nothing
            # Note that -K∇h is the flux, so we need a minus sign here.
            diff⁺.soil.water.K∇h = n̂ * (-bc.surface_flux(aux⁻, t))
        else
            nothing
        end
    elseif bctype == 1
        if bc.bottom_flux != nothing
            # Note that -K∇h is the flux, so we add in a minus sign
            # we add a second minus sign because we choose to specify a BC
            # in terms of ẑ, not n̂, as then it doesnt change directions.
            diff⁺.soil.water.K∇h = -n̂ * (-bc.bottom_flux(aux⁻, t))
        else
            nothing
        end
    end
end



# SoilHeatModel - GeneralBoundaryConditions only so far

"""
    function soil_boundary_state!(
        nf,
        bctype,
        land::LandModel,
        soil::SoilModel,
        heat::SoilHeatModel,
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
    bctype,
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    execute_based_on_boundaries!(
        nf,
        land,
        soil,
        heat,
        heat.boundaries,
        state⁺,
        aux⁺,
        nM,
        state⁻,
        aux⁻,
        bctype,
        t,
    )
end


"""
    function soil_boundary_state!(
        nf,
        bctype,
        land::LandModel,
        soil::SoilModel,
        heat::SoilHeatModel,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        t,
    )

The Neumann-type method for `soil_boundary_state!` for the
`SoilHeatModel`.
"""
function soil_boundary_state!(
    nf,
    bctype,
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
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
    execute_based_on_boundaries!(
        nf,
        land,
        soil,
        heat,
        heat.boundaries,
        state⁺,
        diff⁺,
        aux⁺,
        n̂,
        state⁻,
        diff⁻,
        aux⁻,
        bctype,
        t,
    )
end

"""
    function execute_based_on_boundaries!(
        nf,
        land::LandModel,
        soil::SoilModel,
        heat::SoilHeatModel,
        boundaries::GeneralBoundaryConditions,
        state⁺::Vars,
        aux⁺::Vars,
        nM,
        state⁻::Vars,
        aux⁻::Vars,
        bctype,
        t,
    )

The Dirichlet-type boundary method for `execute_based_on_boundaries!` for the
`SoilHeatModel` when used with `GeneralBoundaryConditions`. This applies the
user-supplied functions of space and time as Dirichlet conditions on `ρe_int`.

Note that not supplying a function (so that it is `nothing`) results in no
boundary condition of this type applied. Additionally, note that the user
supplies a Dirichlet condition for temperature, and this is converted into
a boundary condition on volumetric. internal energy.
"""
function execute_based_on_boundaries!(
    nf,
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    boundaries::GeneralBoundaryConditions,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
)
    bc = boundaries.dirichlet_bc
    if bctype == 2
        if bc.surface_state != nothing
            ϑ_l, θ_i = get_water_content(land.soil.water, aux⁻, state⁻, t)
            θ_l = volumetric_liquid_fraction(
                ϑ_l,
                land.soil.param_functions.porosity,
            )
            ρc_s = volumetric_heat_capacity(
                θ_l,
                θ_i,
                land.soil.param_functions.ρc_ds,
                land.param_set,
            )

            ρe_int_bc = volumetric_internal_energy(
                θ_i,
                ρc_s,
                bc.surface_state(aux⁻, t),
                land.param_set,
            )

            state⁺.soil.heat.ρe_int = ρe_int_bc
        else
            nothing
        end
    elseif bctype == 1
        if bc.bottom_state != nothing
            ϑ_l, θ_i = get_water_content(land.soil.water, aux⁻, state⁻, t)
            θ_l = volumetric_liquid_fraction(
                ϑ_l,
                land.soil.param_functions.porosity,
            )
            ρc_s = volumetric_heat_capacity(
                θ_l,
                θ_i,
                land.soil.param_functions.ρc_ds,
                land.param_set,
            )

            ρe_int_bc = volumetric_internal_energy(
                θ_i,
                ρc_s,
                bc.bottom_state(aux⁻, t),
                land.param_set,
            )

            state⁺.soil.heat.ρe_int = ρe_int_bc
        else
            nothing
        end
    end
end

"""
    function execute_based_on_boundaries!(
        nf,
        land::LandModel,
        soil::SoilModel,
        heat::SoilHeatModel,
        boundaries::GeneralBoundaryConditions,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        bctype,
        t,
    )

The Neumann-type boundary method for `execute_based_on_boundaries!` for the
`SoilHeatModel` when used with `GeneralBoundaryConditions`. This applies the
user-supplied functions of space and time as Neumann conditions on the flux,
equal to -κ∇T.

Note that not supplying a function (so that it is `nothing`) results in no
boundary condition of this type applied.
"""
function execute_based_on_boundaries!(
    nf,
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    boundaries::GeneralBoundaryConditions,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
)
    bc = boundaries.neumann_bc
    if bctype == 2
        if bc.surface_flux != nothing
            # Again, the flux is -κ∇T
            diff⁺.soil.heat.κ∇T = n̂ * (-bc.surface_flux(aux⁻, t))
        else
            nothing
        end
    elseif bctype == 1
        if bc.bottom_flux != nothing
            # two minus signs - the flux is minus κ∇T, and n̂ is -ẑ at the
            # bottom of the domain.
            diff⁺.soil.heat.κ∇T = -n̂ * (-bc.bottom_flux(aux⁻, t))
        else
            nothing
        end
    end
end
