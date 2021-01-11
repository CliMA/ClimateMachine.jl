module River
 
using DocStringExtensions
using ..Land
using ..VariableTemplates
import ..BalanceLaws:
    BalanceLaw,
    vars_state,
    flux_first_order!, 
    Prognostic,
    Auxiliary, 
    Gradient,
    GradientFlux

using ...DGMethods: LocalGeometry
using StaticArrays: SVector

export RiverModel, NoRiverModel, river_boundary_flux!, river_boundary_state!, calculate_velocity


struct NoRiverModel <: BalanceLaw end
## we should change this - we should only specify the slope vector components, not the unit vector components
## and the magnitude
struct RiverModel{Sx,Sy,W,M} <: BalanceLaw
    slope_x::Sx
    slope_y::Sy
    width::W
    mannings::M
end

function RiverModel(
    slope_x::Function,
    slope_y::Function,
    width::Function;
    mannings::Function = (x, y) -> convert(eltype(x), 0.03))
    args = (
        slope_x,
        slope_y,
        width,
        mannings,
    )
    return RiverModel{typeof.(args)...}(args...)
end

function calculate_velocity(river, x::Real, y::Real, h::Real)
    FT = eltype(h)
    sx = FT(river.slope_x(x, y))
    sy = FT(river.slope_y(x, y))
    #magnitude_slope = FT(sqrt(sx^FT(2.0)+sy^FT(2.0)))
    mannings_coeff = FT(river.mannings(x, y))
    magnitude = h^FT(2/3) / mannings_coeff #* sqrt(magnitude_slope)
    #if the slope is positive, dz/dx >0, flow should be in opposite direction. add in  minus signs
    return SVector(-sign(sx) * magnitude*sqrt(abs(sx)),# / magnitude_slope, 
                   -sign(sy) * magnitude*sqrt(abs(sy)),# / magnitude_slope, 
                   zero(FT))
end

vars_state(river::RiverModel, st::Prognostic, FT) = @vars(area::FT)

function Land.land_init_aux!(land::LandModel, river::Union{NoRiverModel,RiverModel}, aux, geom::LocalGeometry)
end

function Land.land_nodal_update_auxiliary_state!(land::LandModel, river::Union{NoRiverModel,RiverModel}, state, aux, t)
end

function flux_first_order!(land::LandModel, river::NoRiverModel, flux::Grad, state::Vars, aux::Vars, t::Real, directions)
end

function flux_first_order!(land::LandModel, river::RiverModel, flux::Grad, state::Vars, aux::Vars, t::Real, directions)
    x = aux.x
    y = aux.y
    width = river.width(x, y)
    area = max(eltype(state)(0.0), state.river.area)
    height = area / width
    v = calculate_velocity(river, x, y, height)
    Q = area * v
    flux.river.area = Q
end 

# boundry conditions 

# General case - to be used with bc::NoBC
function river_boundary_flux!(
    nf,
    bc::Land.AbstractBoundaryConditions,
    m::Union{NoRiverModel,RiverModel},
    land::LandModel,
    _...,
)
end

function river_boundary_state!(
    nf,
    bc::Land.AbstractBoundaryConditions,
    m::Union{NoRiverModel,RiverModel},
    land::LandModel,
    _...,
)
end

# Dirichlet BC for River
function river_boundary_flux!(
    nf,
    bc::Land.Dirichlet,
    model::RiverModel,
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

function river_boundary_state!(
    nf,
    bc::Land.Dirichlet,
    model::RiverModel,
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
    state⁺.river.area = bc_function(aux⁻, t)
end


end
