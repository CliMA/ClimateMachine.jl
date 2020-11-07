module SuperHydrostaticBoussinesqModels

using ClimateMachine

using ...DGMethods.NumericalFluxes

using ..HydrostaticBoussinesq: HydrostaticBoussinesqModel
using ..OceanProblems: InitialValueProblem
using ..CartesianDomains: array_type, communicator

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
##### "Super" HydrostaticBoussinesqModel
#####

struct SuperHydrostaticBoussinesqModel{D, N, E, T}
              domain :: D
    numerical_fluxes :: N
           equations :: E
         timestepper :: T
end

"""
    SuperHydrostaticBoussinesqModel(;
        domain,
        parameters = EarthParameters(),
        initial_conditions = InitialConditions(),
        advection = (momentum = NonLinearAdvectionTerm(),
                     tracers = NonLinearAdvectionTerm()),
        turbulence = (νʰ=0, νᶻ=0, κʰ=0, κᶻ=0),
        coriolis = (f₀=0, β=0),
        rusanov_wave_speeds = (cʰ=0, cᶻ=0),
        buoyancy = (αᵀ,)
        numerical_fluxes = (first_order = RusanovNumericalFlux(), 
                            second_order = CentralNumericalFluxSecondOrder(),
                            gradient = CentralNumericalFluxGradient())
        timestepper = ClimateMachine.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
)
"""
function SuperHydrostaticBoussinesqModel(; domain,
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
)

    FT = eltype(domain)

    problem = InitialValueProblem(
        dimensions = (domain.L.x, domain.L.y, domain.L.z),
        initial_conditions = initial_conditions
    )

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

    return SuperHydrostaticBoussinesqModel(domain, numerical_fluxes, equations, timestepper)
end

function SolverConfiguration(
    model::SuperHydrostaticBoussinesqModel;
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

end # module
