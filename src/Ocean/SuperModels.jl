module SuperModels

using ClimateMachine

using ...DGMethods.NumericalFluxes

using ..HydrostaticBoussinesq: HydrostaticBoussinesqModel
using ..OceanProblems: InitialValueProblem
using ..CartesianDomains: array_type, communicator
using ..Ocean.Fields: field

using ...Mesh.Filters: CutoffFilter, ExponentialFilter
using ...Mesh.Grids: polynomialorder

using ClimateMachine: LSRK144NiegemannDiehlBusch, OceanBoxGCMConfigType,
                      OceanBoxGCMSpecificInfo, DriverConfiguration

import ClimateMachine: SolverConfiguration

#####
##### Build default EarthParameters
#####

using CLIMAParameters: AbstractEarthParameterSet

struct EarthParameters <: AbstractEarthParameterSet end

#####
##### It's super good
#####

struct HydrostaticBoussinesqSuperModel{D, E, S, F, N, T, V}
              domain :: D
           equations :: E
               state :: S
              fields :: F
    numerical_fluxes :: N
         timestepper :: T
              solver :: V
end

"""
    HydrostaticBoussinesqSuperModel(;
        domain,
        parameters = EarthParameters(),
        initial_conditions = InitialConditions(),
        advection = (momentum = NonLinearAdvectionTerm(), tracers = NonLinearAdvectionTerm()),
        turbulence = (νʰ=0, νᶻ=0, κʰ=0, κᶻ=0),
        coriolis = (f₀=0, β=0),
        rusanov_wave_speeds = (cʰ=0, cᶻ=0),
        buoyancy = (αᵀ,)
        numerical_fluxes = (first_order = RusanovNumericalFlux(), second_order = CentralNumericalFluxSecondOrder(), gradient = CentralNumericalFluxGradient())
        timestepper = ClimateMachine.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
)
"""
function HydrostaticBoussinesqSuperModel(; domain,
             parameters = EarthParameters(),
     initial_conditions = InitialConditions(),
              advection = (momentum=NonLinearAdvectionTerm(), tracers=NonLinearAdvectionTerm()),
             turbulence = (νʰ=0, νᶻ=0, κʰ=0, κᶻ=0),
               coriolis = (f₀=0, β=0),
    rusanov_wave_speeds = (cʰ=0, cᶻ=0),
               buoyancy = (αᵀ,),
       numerical_fluxes = (first_order = RusanovNumericalFlux(), 
                           second_order = CentralNumericalFluxSecondOrder(),
                           gradient = CentralNumericalFluxGradient()),
            timestepper = ClimateMachine.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
                filters = nothing,
              modeldata = NamedTuple(),
            init_on_cpu = true,
)

    FT = eltype(domain)

    #####
    ##### Construct generic problem type InitialValueProblem
    #####
    
    problem = InitialValueProblem(
        dimensions = (domain.L.x, domain.L.y, domain.L.z),
        initial_conditions = initial_conditions
    )

    #####
    ##### Build HydrostaticBoussinesqEquations (currently called HydrostaticBoussinesqModel)
    #####
    
    equations = HydrostaticBoussinesqModel{eltype(domain)}(
        parameters,
        problem,
        momentum_advection = advection.momentum,
        tracer_advection = advection.tracers,
        cʰ = FT(rusanov_wave_speeds.cʰ),
        cᶻ = FT(rusanov_wave_speeds.cᶻ),
        αᵀ = FT(buoyancy.αᵀ),
        νʰ = FT(turbulence.νʰ),         # Horizontal viscosity (m² s⁻¹)
        νᶻ = FT(turbulence.νᶻ),         # Horizontal viscosity (m² s⁻¹)
        κʰ = FT(turbulence.κʰ),         # Horizontal diffusivity (m² s⁻¹)
        κᶻ = FT(turbulence.κᶻ),         # Horizontal diffusivity (m² s⁻¹)
        fₒ = FT(coriolis.f₀),           # Coriolis parameter (s⁻¹)
        β = FT(coriolis.β)             # Coriolis parameter gradient (m⁻¹ s⁻¹)
    )

    #####
    ##### "modeldata"
    #####
    ##### OceanModels require filters (?). If one was not provided, we build a default.
    #####
    
    # Default vertical filter and horizontal exponential filter:
    if isnothing(filters)
        grid = domain.grid

        filters = (vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1),
                    exp_filter = ExponentialFilter(grid, 1, 8))
    end

    modeldata = merge(modeldata, filters)

    #####
    ##### We build a DriverConfiguration here for the purposes of building
    ##### a SolverConfiguration. Then we throw it away.
    #####
    
    driver_configuration =  DriverConfiguration(
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
    
    #####
    ##### Pass through the SolverConfiguration interface so that we use
    ##### the checkpointing infrastructure
    #####
    
    solver_configuration = ClimateMachine.SolverConfiguration(
        zero(FT),
        convert(FT, 1e-44), # Approximately the shortest simulation imaginable (s)
        driver_configuration,
        init_on_cpu = init_on_cpu,
        ode_dt = convert(FT, 1e-44),
        ode_solver_type = timestepper,
        modeldata = modeldata
    )

    state = solver_configuration.Q
    solver = solver_configuration

    u = field(domain, state, 1)
    v = field(domain, state, 2)
    η = field(domain, state, 3)
    θ = field(domain, state, 4)

    fields = (u=u, v=v, η=η, θ=θ)

    return HydrostaticBoussinesqSuperModel(domain, equations, state, fields, numerical_fluxes, timestepper, solver)
end

time(model::HydrostaticBoussinesqSuperModel) = model.solver.solver.t
Δt(model::HydrostaticBoussinesqSuperModel) = model.solver.solver.dt
steps(model::HydrostaticBoussinesqSuperModel) = model.solver.solver.steps

#=
function SolverConfiguration(
    model::HydrostaticBoussinesqSuperModel;
    name = "",
    stop_time,
    time_step,
    start_time = 0.0,
    init_on_cpu = true,
    filters = nothing,
    modeldata = NamedTuple(),
)

    # Default vertical filter and horizontal exponential filter:
    if isnothing(filters)
        grid = model.domain.grid

        filters = (vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1),
                    exp_filter = ExponentialFilter(grid, 1, 8))
    end

    modeldata = merge(modeldata, filters)

    driver_configuration =  DriverConfiguration(
        OceanBoxGCMConfigType(),
        name,
        model.domain.Np,
        eltype(model.domain),
        array_type(model.domain),
        model.timestepper,
        model.equations.param_set,
        model.equations,
        communicator(model.domain.grid),
        model.domain.grid,
        model.numerical_fluxes.first_order,
        model.numerical_fluxes.second_order,
        model.numerical_fluxes.gradient,
        OceanBoxGCMSpecificInfo(),
    )

    solver_configuration = ClimateMachine.SolverConfiguration(
        start_time,
        stop_time,
        driver_configuration,
        init_on_cpu = init_on_cpu,
        ode_dt = time_step,
        ode_solver_type = model.timestepper,
        modeldata = modeldata
    )

    return solver_configuration
end
=#

end # module
