module SuperModels

using MPI

using ClimateMachine

using ClimateMachine: Settings

using ...DGMethods.NumericalFluxes

using ..HydrostaticBoussinesq:
    HydrostaticBoussinesqModel, NonLinearAdvectionTerm, Forcing

using ..OceanProblems: InitialValueProblem, InitialConditions
using ..Domains: array_type
using ..Ocean: FreeSlip, Impenetrable, Insulating, OceanBC, Penetrable
using ..Ocean.Fields: SpectralElementField

using ...Mesh.Filters: CutoffFilter, ExponentialFilter
using ...Mesh.Grids: polynomialorders, DiscontinuousSpectralElementGrid

using ClimateMachine:
    LS3NRK33Heuns,
    OceanBoxGCMConfigType,
    OceanBoxGCMSpecificInfo,
    DriverConfiguration

import ClimateMachine: SolverConfiguration

#####
##### It's super good
#####

struct HydrostaticBoussinesqSuperModel{D, G, E, S, F, N, T, C}
    domain::D
    grid::G
    equations::E
    state::S
    fields::F
    numerical_fluxes::N
    timestepper::T
    solver_configuration::C
end

"""
    HydrostaticBoussinesqSuperModel(; domain, time_step, parameters,
         initial_conditions = InitialConditions(),
                  advection = (momentum = NonLinearAdvectionTerm(), tracers = NonLinearAdvectionTerm()),
         turbulence_closure = (νʰ=0, νᶻ=0, κʰ=0, κᶻ=0),
                   coriolis = (f₀=0, β=0),
        rusanov_wave_speeds = (cʰ=0, cᶻ=0),
                   buoyancy = (αᵀ=0,)
           numerical_fluxes = ( first_order = RusanovNumericalFlux(),
                               second_order = CentralNumericalFluxSecondOrder(),
                                   gradient = CentralNumericalFluxGradient())
                timestepper = ClimateMachine.ExplicitSolverType(solver_method=LS3NRK33Heuns),
    )

Builds a `SuperModel` that solves the Hydrostatic Boussinesq equations.
"""
function HydrostaticBoussinesqSuperModel(;
    domain,
    parameters,
    time_step, # We don't want to have to provide this here, but neverthless it's required.
    initial_conditions = InitialConditions(),
    advection = (
        momentum = NonLinearAdvectionTerm(),
        tracers = NonLinearAdvectionTerm(),
    ),
    turbulence_closure = (νʰ = 0, νᶻ = 0, κʰ = 0, κᶻ = 0),
    coriolis = (f₀ = 0, β = 0),
    rusanov_wave_speeds = (cʰ = 0, cᶻ = 0),
    buoyancy = (αᵀ = 0,),
    forcing = Forcing(),
    numerical_fluxes = (
        first_order = RusanovNumericalFlux(),
        second_order = CentralNumericalFluxSecondOrder(),
        gradient = CentralNumericalFluxGradient(),
    ),
    timestepper = ClimateMachine.ExplicitSolverType(
        solver_method = LS3NRK33Heuns,
    ),
    filters = nothing,
    modeldata = NamedTuple(),
    array_type = Settings.array_type,
    mpicomm = MPI.COMM_WORLD,
    init_on_cpu = true,
    boundary_tags = ((0, 0), (0, 0), (1, 2)),
    boundary_conditions = (
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(FreeSlip()), Insulating()),
    ),
)

    #####
    ##### Build the grid
    #####

    # Change global setting if its set here
    Settings.array_type = array_type

    grid = DiscontinuousSpectralElementGrid(
        domain;
        boundary_tags = boundary_tags,
        mpicomm = mpicomm,
        array_type = array_type,
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
        forcing = forcing,
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
        filters = (
            vert_filter = CutoffFilter(grid, polynomialorders(grid)),
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
        (domain.Np, domain.Np),
        eltype(domain),
        array_type,
        timestepper,
        equations.param_set,
        equations,
        MPI.COMM_WORLD,
        grid,
        numerical_fluxes.first_order,
        numerical_fluxes.second_order,
        numerical_fluxes.gradient,
        nothing,
        nothing, # filter
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
        Courant_number = 0.4,
        modeldata = modeldata,
    )

    state = solver_configuration.Q

    u = SpectralElementField(domain, grid, state, 1)
    v = SpectralElementField(domain, grid, state, 2)
    η = SpectralElementField(domain, grid, state, 3)
    θ = SpectralElementField(domain, grid, state, 4)

    fields = (u = u, v = v, η = η, θ = θ)

    return HydrostaticBoussinesqSuperModel(
        domain,
        grid,
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
current_step(model::HydrostaticBoussinesqSuperModel) =
    model.solver_configuration.solver.steps

end # module
