import ClimateMachine.BalanceLaws:
    # declaration
    vars_state,
    # initialization
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    init_state_auxiliary!,
    # rhs computation
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    # boundary conditions
    boundary_conditions,
    boundary_state!
import ClimateMachine.NumericalFluxes:
    numerical_boundary_flux_first_order!
    #numerical_boundary_flux_second_order!

struct DryReferenceState{TP}
    temperature_profile::TP
end

"""
    Declaration of state variables

    vars_state returns a NamedTuple of data types.
"""
function vars_state(m::DryAtmosModel, st::Auxiliary, FT)
    @vars begin
        x::FT
        y::FT
        z::FT
        Φ::FT
        ∇Φ::SVector{3, FT} # TODO: only needed for the linear model
        ref_state::vars_state(m, m.physics.ref_state, st, FT)
    end
end

vars_state(::DryAtmosModel, ::DryReferenceState, ::Auxiliary, FT) =
    @vars(T::FT, p::FT, ρ::FT, ρu::SVector{3, FT}, ρe::FT, ρq::FT)
vars_state(::DryAtmosModel, ::NoReferenceState, ::Auxiliary, FT) = @vars()

function vars_state(::DryAtmosModel, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq::FT
    end
end

"""
    Initialization of state variables

    init_state_xyz! sets up the initial fields within our state variables
    (e.g., prognostic, auxiliary, etc.), however it seems to not initialized
    the gradient flux variables by default.
"""
function init_state_prognostic!(
        model::DryAtmosModel,
        state::Vars,
        aux::Vars,
        localgeo,
        t
    )
    x = aux.x
    y = aux.y
    z = aux.z

    parameters = model.physics.parameters
    ic = model.initial_conditions

    # TODO!: Set to 0 by default or assign IC
    if !isnothing(ic)
        state.ρ  = ic.ρ(parameters, x, y, z)
        state.ρu = ic.ρu(parameters, x, y, z)
        state.ρe = ic.ρe(parameters, x, y, z)
        state.ρq = ic.ρq(parameters, x, y, z)
    end

    return nothing
end

function nodal_init_state_auxiliary!(
    model::DryAtmosModel,
    state_auxiliary,
    tmp,
    geom,
)
    init_state_auxiliary!(model, model.physics.orientation, state_auxiliary, geom)
    init_state_auxiliary!(model, model.physics.ref_state, state_auxiliary, geom)
end

function init_state_auxiliary!(
    model::DryAtmosModel,
    ::SphericalOrientation,
    state_auxiliary,
    geom,
)
    g = model.physics.parameters.g

    r = norm(geom.coord)
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
    state_auxiliary.Φ = g * r
    state_auxiliary.∇Φ = g * geom.coord / r
end

function init_state_auxiliary!(
    model::DryAtmosModel,
    ::FlatOrientation,
    state_auxiliary,
    geom,
)
    g = model.physics.parameters.g

    FT = eltype(state_auxiliary)
    
    r = geom.coord[3]
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
    state_auxiliary.Φ = g * r
    state_auxiliary.∇Φ = SVector{3, FT}(0, 0, g)
end

function init_state_auxiliary!(
    ::DryAtmosModel,
    ::NoReferenceState,
    state_auxiliary,
    geom,
) end

function init_state_auxiliary!(
    model::DryAtmosModel,
    ref_state::DryReferenceState,
    state_auxiliary,
    geom,
)
    orientation = model.physics.orientation   
    R_d         = model.physics.parameters.R_d
    γ           = model.physics.parameters.γ
    Φ           = state_auxiliary.Φ

    FT = eltype(state_auxiliary)

    # Calculation of a dry reference state
    z = altitude(model, orientation, geom)
    T, p = ref_state.temperature_profile(model.physics.parameters, z)
    ρ  = p / R_d / T
    ρu = SVector{3, FT}(0, 0, 0)
    ρe = p / (γ - 1) + dot(ρu, ρu) / 2ρ + ρ * Φ
    ρq = FT(0)

    state_auxiliary.ref_state.T  = T
    state_auxiliary.ref_state.p  = p
    state_auxiliary.ref_state.ρ  = ρ
    state_auxiliary.ref_state.ρu = ρu
    state_auxiliary.ref_state.ρe = ρe
    state_auxiliary.ref_state.ρq = ρq    
end

"""
    LHS computations
"""
@inline function flux_first_order!(
    model::DryAtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    physics = model.physics
    lhs = model.physics.lhs
    ntuple(Val(length(lhs))) do s
        Base.@_inline_meta
        calc_component!(flux, lhs[s], state, aux, physics)
    end
end

"""
    RHS computations
"""
function source!(m::DryAtmosModel, source, state_prognostic, state_auxiliary, _...)
    sources = m.physics.sources
    physics = m.physics

    ntuple(Val(length(sources))) do s
        Base.@_inline_meta
        calc_component!(source, sources[s], state_prognostic, state_auxiliary, physics)
    end
end

"""
    Boundary conditions
"""
struct BottomBC <: AbstractBoundaryCondition end
struct TopBC <: AbstractBoundaryCondition end

boundary_conditions(model::DryAtmosModel) = model.boundary_conditions
#boundary_conditions(model::DryAtmosModel) = (BottomBC(), TopBC())

function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    bctype,
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
  return nothing
end

function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    bctype::BottomBC,
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
    state⁺.ρ = state⁻.ρ
    state⁺.ρe = state⁻.ρe
    state⁺.ρq = state⁻.ρq

    ρu⁻ = state⁻.ρu
    
    # project and reflect
    state⁺.ρu = ρu⁻ - n̂ ⋅ ρu⁻ .* SVector(n̂) - n̂ ⋅ ρu⁻ .* SVector(n̂)
    numerical_flux_first_order!(
      numerical_flux,
      model,
      fluxᵀn,
      n̂,
      state⁻,
      aux⁻,
      state⁺,
      aux⁺,
      t,
      direction,
    )
    
    ρ = state⁻.ρ
    ρu = state⁻.ρu
    ρe = state⁻.ρe
    ρq = state⁻.ρq
    eos = model.physics.eos
    parameters = model.physics.parameters
    cp_d = model.physics.parameters.cp_d
    LH_v0 = model.physics.parameters.LH_v0

    u = ρu / ρ
    q = ρq / ρ
    speed_tangential = norm((I - n̂ ⊗ n̂) * u)

    # obtain drag coefficients
    #Cₕ = bc.drag_coefficient_temperature(state, aux)
    #Cₑ = bc.drag_coefficient_moisture(state, aux)
    Cₕ = 0.044
    Cₑ = 0.044 

    # obtain surface fields
    #T_sfc, q_tot_sfc = bc.surface_fields(atmos, state, aux, t)
    T_sfc = 300.0
    q_tot_sfc = 0.02  

    # surface cooling due to wind via transport of dry energy (sensible heat flux)
    cp = calc_cp(eos, state⁻, parameters)
    T = calc_air_temperature(eos, state⁻, aux⁻, parameters)
    H = ρ * Cₕ * speed_tangential * cp * (T - T_sfc)

    # surface cooling due to wind via transport of moisture (latent energy flux)
    E = ρ * Cₑ * speed_tangential * LH_v0 * (q - q_tot_sfc)
    #println(E)
    #println(H)

    # fluxᵀn.ρ  += E / LH_v0 
    # fluxᵀn.ρu += E / LH_v0 .* u
    fluxᵀn.ρ = 0.0
    fluxᵀn.ρe = E + H
    fluxᵀn.ρq = E / LH_v0

    # fluxᵀn.ρe = -(E + H)
    # fluxᵀn.ρq = -(E / LH_v0)
end

function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    bctype::TopBC,
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
    
    # project and reflect
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
# function boundary_state!(
#     nmf::NumericalFluxFirstOrder,
#     bctype,
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

#function numerical_boundary_flux_second_order!(_...) 
#    return nothing
#end

function boundary_state!(
    nf::NumericalFluxSecondOrder,
    bc,
    lm::DryAtmosModel,
    args...,
)
    nothing
end

"""
    Utils
"""
function altitude(model::DryAtmosModel, ::SphericalOrientation, geom)
    return norm(geom.coord) - model.physics.parameters.a
end

function altitude(::DryAtmosModel, ::FlatOrientation, geom)
    @inbounds geom.coord[3]
end
