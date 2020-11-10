module SuperModels

using ClimateMachine

using ...DGMethods.NumericalFluxes

using ..HydrostaticBoussinesq:
    HydrostaticBoussinesqModel, NonLinearAdvectionTerm
using ..OceanProblems: InitialValueProblem
using ..CartesianDomains: array_type, communicator
using ..Ocean.Fields: field

using ...Mesh.Filters: CutoffFilter, ExponentialFilter
using ...Mesh.Grids: polynomialorder

using ClimateMachine:
    LSRK144NiegemannDiehlBusch,
    OceanBoxGCMConfigType,
    OceanBoxGCMSpecificInfo,
    DriverConfiguration

import ClimateMachine: SolverConfiguration

#####
##### Build default EarthParameters
#####

using CLIMAParameters: AbstractEarthParameterSet

struct EarthParameters <: AbstractEarthParameterSet end

#####
##### It's super good
#####

struct HydrostaticBoussinesqSuperModel{D, E, S, F, N, T, C}
    domain::D
    equations::E
    state::S
    fields::F
    numerical_fluxes::N
    timestepper::T
    solver_configuration::C
end

"""
    HydrostaticBoussinesqSuperModel(; domain, time_step,
                 parameters = EarthParameters(),
         initial_conditions = InitialConditions(),
                  advection = (momentum = NonLinearAdvectionTerm(), tracers = NonLinearAdvectionTerm()),
         turbulence_closure = (νʰ=0, νᶻ=0, κʰ=0, κᶻ=0),
                   coriolis = (f₀=0, β=0),
        rusanov_wave_speeds = (cʰ=0, cᶻ=0),
                   buoyancy = (αᵀ=0,)
           numerical_fluxes = ( first_order = RusanovNumericalFlux(),
                               second_order = CentralNumericalFluxSecondOrder(),
                                   gradient = CentralNumericalFluxGradient())
                timestepper = ClimateMachine.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
    )

Builds a `SuperModel` that solves the Hydrostatic Boussinesq equations.
"""
function HydrostaticBoussinesqSuperModel(;
    domain,
    time_step, # We don't want to have to provide this here, but neverthless it's required.
    parameters = EarthParameters(),
    initial_conditions = InitialConditions(),
    advection = (
        momentum = NonLinearAdvectionTerm(),
        tracers = NonLinearAdvectionTerm(),
    ),
    turbulence_closure = (νʰ = 0, νᶻ = 0, κʰ = 0, κᶻ = 0),
    coriolis = (f₀ = 0, β = 0),
    rusanov_wave_speeds = (cʰ = 0, cᶻ = 0),
    buoyancy = (αᵀ = 0,),
    numerical_fluxes = (
        first_order = RusanovNumericalFlux(),
        second_order = CentralNumericalFluxSecondOrder(),
        gradient = CentralNumericalFluxGradient(),
    ),
    timestepper = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    ),
    filters = nothing,
    modeldata = NamedTuple(),
    init_on_cpu = true,
    boundary_conditions = (
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(FreeSlip()), Insulating()),
    ),
)

    FT = eltype(domain)

    #####
    ##### Construct generic problem type InitialValueProblem
    #####

    problem = InitialValueProblem{FT}(
        dimensions = (domain.L.x, domain.L.y, domain.L.z),
        initial_conditions = initial_conditions,
        boundary_conditions = boundary_conditions,
    )

    #####
    ##### Build HydrostaticBoussinesqModel/Equations
    #####

    equations = HydrostaticBoussinesqModel{eltype(domain)}(
        parameters,
        problem,
        momentum_advection = advection.momentum,
        tracer_advection = advection.tracers,
        cʰ = convert(FT, rusanov_wave_speeds.cʰ),
        cᶻ = convert(FT, rusanov_wave_speeds.cᶻ),
        αᵀ = convert(FT, buoyancy.αᵀ),
        νʰ = convert(FT, turbulence_closure.νʰ),
        νᶻ = convert(FT, turbulence_closure.νᶻ),
        κʰ = convert(FT, turbulence_closure.κʰ),
        κᶻ = convert(FT, turbulence_closure.κᶻ),
        fₒ = convert(FT, coriolis.f₀),
        β = FT(coriolis.β),
    )

    ####
    #### "modeldata"
    ####
    #### OceanModels require filters (?). If one was not provided, we build a default.
    ####

    # Default vertical filter and horizontal exponential filter:
    if isnothing(filters)
        grid = domain.grid

        filters = (
            vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1),
            exp_filter = ExponentialFilter(grid, 1, 8),
        )
    end

    modeldata = merge(modeldata, filters)

    ####
    #### We build a DriverConfiguration here for the purposes of building
    #### a SolverConfiguration. Then we throw it away.
    ####

    driver_configuration = DriverConfiguration(
        OceanBoxGCMConfigType(),
        "",
        domain.Np,
        eltype(domain),
        array_type(domain),
        timestepper,
        equations.param_set,
        equations,
        communicator(domain.grid),
        domain.grid,
        numerical_fluxes.first_order,
        numerical_fluxes.second_order,
        numerical_fluxes.gradient,
        OceanBoxGCMSpecificInfo(),
    )

    ####
    #### Pass through the SolverConfiguration interface so that we use
    #### the checkpointing infrastructure
    ####

    solver_configuration = ClimateMachine.SolverConfiguration(
        zero(FT),
        convert(FT, time_step),
        driver_configuration,
        init_on_cpu = init_on_cpu,
        ode_dt = convert(FT, time_step),
        ode_solver_type = timestepper,
        modeldata = modeldata,
    )

    state = solver_configuration.Q

    u = field(domain, state, 1)
    v = field(domain, state, 2)
    η = field(domain, state, 3)
    θ = field(domain, state, 4)

    fields = (u = u, v = v, η = η, θ = θ)

    return HydrostaticBoussinesqSuperModel(
        domain,
        equations,
        state,
        fields,
        numerical_fluxes,
        timestepper,
        solver_configuration,
    )
end

current_time(model::HydrostaticBoussinesqSuperModel) =
    model.solver_configuration.solver.t
Δt(model::HydrostaticBoussinesqSuperModel) =
    model.solver_configuration.solver.dt
steps(model::HydrostaticBoussinesqSuperModel) =
    model.solver_configuration.solver.steps

end # module
