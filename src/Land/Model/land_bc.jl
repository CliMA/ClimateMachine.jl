boundary_conditions(::LandModel) = (1, 2)

function boundary_state!(
    nf,
    bctype,
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
    args = (state⁺, diff⁺, aux⁺, n̂, state⁻, diff⁻, aux⁻, t)
    soil_boundary_state!(nf, bctype, land, land.soil, land.soil.water, args...)
    soil_boundary_state!(nf, bctype, land, land.soil, land.soil.heat, args...)
end


function boundary_state!(
    nf,
    bctype,
    land::LandModel,
    state⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    args = (state⁺, aux⁺, n̂, state⁻, aux⁻, t)
    soil_boundary_state!(nf, bctype, land, land.soil, land.soil.water, args...)
    soil_boundary_state!(nf, bctype, land, land.soil, land.soil.heat, args...)
end
