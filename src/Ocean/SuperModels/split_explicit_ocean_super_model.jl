using LinearAlgebra
using StaticArrays
using Logging
using Printf
using Dates

using ...BalanceLaws: vars_state, Prognostic, Auxiliary
using ...Mesh.Topologies
using ...Mesh.Grids
using ...Mesh.Filters
using ...DGMethods
using ...DGMethods.NumericalFluxes
using ...MPIStateArrays
using ...ODESolvers
using ...VariableTemplates: flattenednames
using ...Ocean.SplitExplicit01
using ...GenericCallbacks
using ...VTK
using ...Checkpoint

import ...DGMethods:
    update_auxiliary_state!,
    update_auxiliary_state_gradient!,
    VerticalDirection

import ..Ocean.SplitExplicit01:
    ocean_init_aux!,
    ocean_init_state!,
    ocean_boundary_state!,
    CoastlineFreeSlip,
    CoastlineNoSlip,
    OceanFloorFreeSlip,
    OceanFloorNoSlip,
    OceanSurfaceNoStressNoForcing,
    OceanSurfaceStressNoForcing,
    OceanSurfaceNoStressForcing,
    OceanSurfaceStressForcing

using CLIMAParameters.Planet: grav

#####
##### It's super good
#####

struct SplitExplicitOceanSuperModel{D, G, B, C, F, T}
    domain::D
    grid::G
    barotropic::B
    baroclinic::C
    fields::F
    timestepper::T
end

"""
    SplitExplicitOceanSuperModel(; domain, time_step, parameters,
         initial_conditions = InitialConditions(),
         turbulence_closure = (νʰ=0, νᶻ=0, κʰ=0, κᶻ=0, κᶜ=0),
                   coriolis = (f₀=0, β=0),
        rusanov_wave_speeds = (cʰ=0, cᶻ=0),
                   buoyancy = (αᵀ=0,)
           numerical_fluxes = ( first_order = RusanovNumericalFlux(),
                               second_order = CentralNumericalFluxSecondOrder(),
                                   gradient = CentralNumericalFluxGradient())
                timestepper = ClimateMachine.ExplicitSolverType(solver_method=LS3NRK33Heuns),
    )

Builds a `SuperModel` that solves the Hydrostatic Boussinesq equations.

Note: `barotropic_time_step` is adjusted so that `barotropic_time_step / baroclinic_time_step` is a
multiple of 12.

`fractional_barotropic_averaging_window` controls time-averaging required to reconcile the 
fast barotropic and slow baroclinic mode. In particular, `fractional_barotropic_averaging_window`
is the fractional size of the averaging window for the barotropic mode
relative to the `baroclinic_time_step` (in other words, the window size expressed as a
fraction of `baroclinic_time_step`). `fractional_barotropic_averaging_window` must be less than 1 to be
a valid fraction, and also significantly greater than 0 for valid averaging, so that
`ϵ > fractional_barotropic_averaging_window >= 1` where `ϵ` is not that small (probably `ϵ ≈ 1/8`).

Note: `fractional_barotropic_averaging_window` is a *target* window 
that's adjusted depending on the number of Runge-Kutta stages.
"""
function SplitExplicitOceanSuperModel(;
    domain,
    parameters,
    baroclinic_time_step,
    barotropic_time_step,
    initial_time = 0.0,
    fractional_barotropic_averaging_window = 1/3,
    implicit_gmres_stages = 0,
    initial_conditions = InitialConditions(),
    turbulence_closure = (νʰ = 0, νᶻ = 0, κʰ = 0, κᶻ = 0, κᶜ = 0),
    coriolis = (f₀ = 0, β = 0),
    rusanov_wave_speeds = (cʰ = 0, cᶻ = 0),
    buoyancy = (αᵀ = 0,),
    forcing = Forcing(),
    numerical_fluxes = (
        first_order = RusanovNumericalFlux(),
        second_order = CentralNumericalFluxSecondOrder(),
        gradient = CentralNumericalFluxGradient(),
    ),
    filters = nothing,
    modeldata = NamedTuple(),
    array_type = Settings.array_type,
    mpicomm = MPI.COMM_WORLD,
    init_on_cpu = true,
    boundary_tags = ((0, 0), (0, 0), (1, 2)),
    boundary_conditions = (
        CoastlineNoSlip(),
        OceanFloorNoSlip(),
        OceanSurfaceStressForcing(),
    ),
)
    
    #####
    ##### Build the grid
    #####

    # Change global setting if its set here
    Settings.array_type = array_type
    FT = eltype(domain)

    west, east = domain.x
    south, north = domain.y
    bottom, top = domain.z

    element_coordinates = (
        range(west, east, length = domain.Ne.x + 1),
        range(south, north, length = domain.Ne.y + 1),
        range(bottom, top, length = domain.Ne.z + 1),
    )

    topology = StackedBrickTopology(
        mpicomm,
        element_coordinates;
        periodicity = tuple(domain.periodicity...),
        boundary = boundary_tags,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = domain.Np,
    )

    horizontal_topology = BrickTopology(
        mpicomm,
        element_coordinates[1:2],
        periodicity = tuple(domain.periodicity[1], domain.periodicity[2]),
    )

    horizontal_grid = DiscontinuousSpectralElementGrid(
        horizontal_topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = domain.Np,
    )

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

    baroclinic_equations = OceanModel{eltype(domain)}(problem;
        grav = grav(parameters),
        cʰ = convert(FT, rusanov_wave_speeds.cʰ),
        cᶻ = convert(FT, rusanov_wave_speeds.cᶻ),
        αᵀ = convert(FT, buoyancy.αᵀ),
        νʰ = convert(FT, turbulence_closure.νʰ),
        νᶻ = convert(FT, turbulence_closure.νᶻ),
        κʰ = convert(FT, turbulence_closure.κʰ),
        κᶻ = convert(FT, turbulence_closure.κᶻ),
        κᶜ = convert(FT, turbulence_closure.κᶜ),
        fₒ = convert(FT, coriolis.f₀),
        β = convert(FT, coriolis.β),
        add_fast_substeps = 1 / fractional_barotropic_averaging_window,
        numImplSteps = implicit_gmres_stages,
        ivdc_dt = baroclinic_time_step / implicit_gmres_stages,
    )

    barotropic_equations = BarotropicModel(baroclinic_equations)

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

    min_Δx = min_node_distance(grid, HorizontalDirection())
    min_Δz = min_node_distance(grid, VerticalDirection())
    c = sqrt(grav(parameters) * domain.L.z)

    @info @sprintf(
        """ Gravity wave CFL condition
            min(Δx) / 2√(gH) = %.1f
              fast time step = %.1f
            Gravity wave CFL = %.1f\n""",
        min_Δx / 2c,
        barotropic_time_step,
        barotropic_time_step * 2c / min_Δx,
    )

    # 2 horizontal directions + harmonic viscosity or diffusion: 2^2 factor in CFL:
    νʰ = baroclinic_equations.νʰ
    νᶻ = baroclinic_equations.νᶻ
    κʰ = baroclinic_equations.κʰ
    κᶻ = baroclinic_equations.κᶻ

    viscous_time_step_limit = 1 / (2 * νʰ / min_Δx^2 + νᶻ / min_Δz^2) / 4
    diffusive_time_step_limit = 1 / (2 * κʰ / min_Δx^2 + κᶻ / min_Δz^2) / 4

    @info @sprintf(
        """ Viscous and diffusive CFL conditions
            1 / (8νʰ / min(Δx)^2 + 4νᶻ / min(Δz)^2) = %.1f
            1 / (8κʰ / min(Δx)^2 + 4κᶻ / min(Δz)^2) = %.1f
                                     slow time step = %.1f
                                        Viscous CFL = %.1f
                                      Diffusive CFL = %.1f\n""",
        viscous_time_step_limit,
        diffusive_time_step_limit,
        baroclinic_time_step,
        baroclinic_time_step / viscous_time_step_limit,
        baroclinic_time_step / diffusive_time_step_limit,
    )

    baroclinic_discretization = OceanDGModel(
        baroclinic_equations,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    barotropic_discretization = DGModel(
        barotropic_equations,
        horizontal_grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    baroclinic_state = init_ode_state(baroclinic_discretization, FT(0); init_on_cpu = true)
    barotropic_state = init_ode_state(barotropic_discretization, FT(0); init_on_cpu = true)

    u = SpectralElementField(domain, grid, baroclinic_state, 1)
    v = SpectralElementField(domain, grid, baroclinic_state, 2)
    η = SpectralElementField(domain, grid, baroclinic_state, 3)
    θ = SpectralElementField(domain, grid, baroclinic_state, 4)

    fields = (u = u, v = v, η = η, θ = θ)

    #####
    ##### Timestepper setup
    #####

    baroclinic_timestepper = LS3NRK33Heuns(baroclinic_discretization, baroclinic_state,
                                           dt = baroclinic_time_step, t0 = initial_time)

    barotropic_timestepper = LS3NRK33Heuns(barotropic_discretization, barotropic_state,
                                           dt = barotropic_time_step, t0 = initial_time)

    timestepper = SplitExplicitLSRK3nSolver(baroclinic_timestepper, barotropic_timestepper)

    #####
    ##### Organize model properties
    #####

    baroclinic = (state = baroclinic_state,
                  discretization = baroclinic_discretization,
                  equations = baroclinic_equations)

    barotropic = (state = barotropic_state,
                  discretization = barotropic_discretization,
                  equations = barotropic_equations)

    return SplitExplicitOceanSuperModel(
        domain,
        grid,
        barotropic,
        baroclinic,
        fields,
        timestepper,
    )
end
