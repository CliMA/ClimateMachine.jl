module Land

using DocStringExtensions
using LinearAlgebra, StaticArrays
using Statistics
using Interpolations

using PlantHydraulics

using CLIMAParameters
using CLIMAParameters.Planet:
    ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0, T_freeze, grav

using ..VariableTemplates
using ..MPIStateArrays
using ..BalanceLaws
import ..BalanceLaws:
    BalanceLaw,
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    boundary_conditions,
    parameter_set,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    nodal_update_auxiliary_state!,
    update_auxiliary_state!
using ..DGMethods: LocalGeometry, DGModel
export LandModel


"""
    LandModel{PS, S, LBC, SRC, IS} <: BalanceLaw

A BalanceLaw for land modeling.
Users may over-ride prescribed default values for each field.

# Usage

    LandModel(
        param_set,
        soil;
        boundary_conditions,
        source,
        init_state_prognostic
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LandModel{PS, S, LBC, SRC, IS} <: BalanceLaw
    "Parameter set"
    param_set::PS
    "Soil model"
    soil::S
    "struct of boundary conditions"
    boundary_conditions::LBC
    "Source Terms (Problem specific source terms)"
    source::SRC
    "Initial Condition (Function to assign initial values of state variables)"
    init_state_prognostic::IS
    # "Initial Condition (Function to assign initial values of state variables)"
    # init_state_prognostic::IS
end

parameter_set(m::LandModel) = m.param_set

"""
    LandModel(
        param_set::AbstractParameterSet,
        soil::BalanceLaw;
        boundary_conditions::LBC = (),
        source::SRC = (),
        init_state_prognostic::IS = nothing
    ) where {SRC, IS, LBC}

Constructor for the LandModel structure.
"""
function LandModel(
    param_set::AbstractParameterSet,
    soil::BalanceLaw;
    boundary_conditions::LBC = LandDomainBC(),
    source::SRC = (),
    init_state_prognostic::IS = nothing,
) where {SRC, IS, LBC}
    @assert init_state_prognostic ≠ nothing
    land = (param_set, soil, boundary_conditions, source, init_state_prognostic)
    return LandModel{typeof.(land)...}(land...)
end

"""
   root_extraction(
       plant_hs::AbstractPlantOrganism{FT},
       ψ::FT,
       qsum::FT=FT(0)
   ) where {FT}

Compute the volumetric water extraction by the roots.
"""
function root_extraction(
   plant_hs::GrassLikeOrganism{FT}, #AbstractPlantOrganism{FT},
   ψ::Float64, #Array{FT,1}, # using a ; means we would have to set qsum, now we dont, can leave blank as we gave a defaul
   qsum::FT=FT(0) # (sum of flow rates, no water going into canopy by default), can leave qsum blank
) where {FT}
    # we dont want to update his model at every point ; we store hydarulic head in aux, we could
    # access the whole thing, and charlie might know...
    for i_root in eachindex(plant_hs.roots)
        plant_hs.roots[i_root].p_ups = ψ[i_root];
    end
    roots_flow!(plant_hs, qsum) # no water going into canopy (no daytime transpiration), this updates the flow rate in each root
    root_extraction = plant_hs.cache_q # array of flow rates in each root layer (mol W/s/layer), maybe not necessary to rewrite name
    return root_extraction
end

function vc_integral(
            vc::LogisticSingle{FT},
            p_dos::FT,
            p_ups::FT,
            h::FT,
            E::FT,
            Kmax::FT
) where {FT<:AbstractFloat}
    @assert p_ups <= 0 && p_dos <= 0;

    # unpack data from VC
    @unpack a,b = vc;
    _krghe = Kmax * ρg_MPa(FT) * h * (a+1) / a + E;
    _lower = b * _krghe;
    _multi = Kmax * (a+1) / a * E;
    _upper_dos = log(a * _krghe * exp(b*p_dos) + E);
    _upper_ups = log(a * _krghe * exp(b*p_ups) + E);

    return _multi * (_upper_ups - _upper_dos) / _lower
end

function vars_state(land::LandModel, st::Prognostic, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end


function vars_state(land::LandModel, st::Auxiliary, FT)
    @vars begin
        z::FT
        soil::vars_state(land.soil, st, FT)
        roots_source::FT
    end
end

function vars_state(land::LandModel, st::Gradient, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end

function vars_state(land::LandModel, st::GradientFlux, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end

function nodal_init_state_auxiliary!(
    land::LandModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.z = geom.coord[3]
    land_init_aux!(land, land.soil, aux, geom)
    aux.roots_source = 0.0 #land.initial_roots_source(aux) # an input to the model, give initial water content to yujies code get source back
end

function flux_first_order!(
    land::LandModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    directions,
) end


function compute_gradient_argument!(
    land::LandModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_argument!(land, land.soil, transform, state, aux, t)
end

function compute_gradient_flux!(
    land::LandModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_flux!(
        land,
        land.soil,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )

end

function flux_second_order!(
    land::LandModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(
        land,
        land.soil,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )

end

function update_auxiliary_state!(
    dg::DGModel,
    land::LandModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    FT = eltype(t) #Float64
    # Get vector of psis
    update_auxiliary_state!(nodal_update_auxiliary_state!, dg, land, Q, t, elems)

    aux = dg.state_auxiliary
    h_ind = varsindex(vars_state(land, Auxiliary(), FT), :soil, :water, :h)
    h = Array(Q[:, h_ind, :][:])
    z_ind = varsindex(vars_state(land, Auxiliary(), FT), :z)
    z = Array(aux[:, z_ind, :][:])
    roots_source_ind = varsindex(vars_state(land, Auxiliary(), FT), :roots_source)

    # psi_vec = h - z # need to pass 4 columns in future
    # @show(psi_vec)

    # itp = interpolate(A, options...)
    # v = itp(x, y, ...)

    # # call Yujie's function - give array of psis
    # z_root = FT(-0.5)
    # z_canopy = FT(0.5)
    # soil_bounds = [FT(0), FT(-1)] # FT(-0.2), FT(-0.6), FT(-1)]
    # air_bounds = [FT(0), FT(1)]

    # plant_hs = create_grass(z_root, z_canopy, soil_bounds, air_bounds)

    # Ni = length(aux[:,roots_source_ind,1])
    # # @show(Ni)
    # Nj = length(aux[1,roots_source_ind,:])
    # # @show(Nj)
    # for i in 1:Ni
    #     for j in 1:Nj
    #         aux[i,roots_source_ind,j] = root_extraction(plant_hs, psi_vec) #; qsum=FT(0))
    #     end
    # end

    # itp = interpolate(A, options...)
    # v = itp(x, y, ...)

    # soil needs be same depth at roots model
    # do levels of different soil layers need to match? i.e. do the properties of his soil layers change?
    # need to supply at least as many psis as there are layers in Yujie's model
    # need to interpolate the psi vec we give to supply as many psis as there are layers in Yujie's model,
        # need to pass psi values at the level of the tips of the roots
    # need to interpolate the answer we get back to get a source term for all our points

    # aux.roots_source = s #(aux.z)
    # vars_state(land, Auxiliary(), FT), :roots_source) = s
    # psi_vec = dg.state_auxiliary[:,2,:] - dg.state_auxiliary[:,1,:]
    # @show(typeof(psi_vec))
    # @show(size(psi_vec))
    # @show(psi_vec)
end

function nodal_update_auxiliary_state!(
    land::LandModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    land_nodal_update_auxiliary_state!(land, land.soil, state, aux, t)
end

function source!(
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    land_source!(land.source, land, source, state, diffusive, aux, t, direction)
end


function init_state_prognostic!(
    land::LandModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
    args...,
)
    land.init_state_prognostic(land, state, aux, coords, t, args...)
end

include("RadiativeEnergyFlux.jl")
using .RadiativeEnergyFlux
include("SoilWaterParameterizations.jl")
using .SoilWaterParameterizations
include("SoilHeatParameterizations.jl")
using .SoilHeatParameterizations
include("soil_model.jl")
include("soil_water.jl")
include("soil_heat.jl")
include("Runoff.jl")
using .Runoff
include("land_bc.jl")
include("soil_bc.jl")
#include("PlantHydraulics.jl") # just need roots_flow! create_grass
include("source.jl")
end # Module
