# Test that the way we specify boundary conditions works as expected
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.Runoff
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state

import ClimateMachine.DGMethods.FVReconstructions: FVLinear

@testset "NoRunoff" begin
    ClimateMachine.init()

    FT = Float64

    function init_soil_water!(land, state, aux, localgeo, time)
        myfloat = eltype(state)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
    end

    soil_param_functions =
        SoilParamFunctions{FT}(porosity = 0.75, Ksat = 1e-7, S_s = 1e-3)
    surface_precip_amplitude = FT(3e-8)
    f = FT(pi * 2.0 / 300.0)
    precip = (t) -> surface_precip_amplitude * sin(f * t)
    ϑ_l0 = (aux) -> eltype(aux)(0.2)
    bc = LandDomainBC(
        bottom_bc = LandComponentBC(
            soil_water = Neumann((aux, t) -> eltype(aux)(0.0)),
        ),
        surface_bc = LandComponentBC(
            soil_water = SurfaceDrivenWaterBoundaryConditions(
                FT;
                precip_model = DrivenConstantPrecip{FT}(precip),
                runoff_model = NoRunoff(),
            ),
        ),
    )

    soil_water_model = SoilWaterModel(FT; initialϑ_l = ϑ_l0)
    soil_heat_model = PrescribedTemperatureModel()

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    N_poly = 5
    nelem_vert = 50


    # Specify the domain boundaries
    zmax = FT(0)
    zmin = FT(-1)

    driver_config = ClimateMachine.SingleStackConfiguration(
        "LandModel",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        m;
        zmin = zmin,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
    )


    t0 = FT(0)
    timeend = FT(150)
    dt = FT(0.05)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    n_outputs = 60
    mygrid = solver_config.dg.grid
    every_x_simulation_time = ceil(Int, timeend / n_outputs)
    state_types = (Prognostic(), Auxiliary(), GradientFlux())
    dons_arr =
        Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0] # store time data

    # We specify a function which evaluates `every_x_simulation_time` and returns
    # the state vector, appending the variables we are interested in into
    # `dons_arr`.

    callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
        dons = dict_of_nodal_states(solver_config, state_types; interp = true)
        push!(dons_arr, dons)
        push!(time_data, gettime(solver_config.solver))
        nothing
    end

    # # Run the integration
    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))

    # Get z-coordinate
    z = get_z(solver_config.dg.grid; rm_dupes = true)
    N = length(dons_arr)
    # Note - we take the indices 2:N here to avoid the t = 0 spot. Gradients
    # are not calculated before the invoke! command, so we cannot compare at t=0.
    computed_surface_flux = [dons_arr[k]["soil.water.K∇h[3]"][end] for k in 2:N]
    t = time_data[2:N]
    prescribed_surface_flux = t -> FT(-1) * FT(3e-8 * sin(pi * 2.0 * t / 300.0))
    MSE = mean((prescribed_surface_flux.(t) .- computed_surface_flux) .^ 2.0)
    @test MSE < 5e-7
end



@testset "Explicit Dirichlet BC Comparison" begin
    # This tests the "if" branch of compute_surface_grad_bc
    ClimateMachine.init()
    FT = Float64
    function init_soil_water!(land, state, aux, localgeo, time)
        myfloat = eltype(state)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
    end

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.495,
        Ksat = 0.0443 / (3600 * 100),
        S_s = 1e-4,
    )
    surface_precip_amplitude = FT(-5e-4)

    precip = (t) -> surface_precip_amplitude
    hydraulics = vanGenuchten{FT}(n = 2.0)
    function hydrostatic_profile(z, zm, porosity, n, α)
        myf = eltype(z)
        m = FT(1 - 1 / n)
        S = FT((FT(1) + (α * (z - zm))^n)^(-m))
        return FT(S * porosity)
    end
    N_poly = 2
    nelem_vert = 50

    # Specify the domain boundaries
    zmax = FT(0)
    zmin = FT(-0.35)
    Δz = abs(FT((zmin - zmax) / nelem_vert / 2))

    ϑ_l0 =
        (aux) -> eltype(aux)(hydrostatic_profile(
            aux.z,
            -0.35,
            0.495,
            hydraulics.n,
            hydraulics.α,
        ))

    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = hydraulics,
        initialϑ_l = ϑ_l0,
    )
    soil_heat_model = PrescribedTemperatureModel()

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()


    bc = LandDomainBC(
        bottom_bc = LandComponentBC(
            soil_water = Neumann((aux, t) -> eltype(aux)(0.0)),
        ),
        surface_bc = LandComponentBC(
            soil_water = SurfaceDrivenWaterBoundaryConditions(
                FT;
                precip_model = DrivenConstantPrecip{FT}(precip),
                runoff_model = CoarseGridRunoff{FT}(Δz),
            ),
        ),
    )


    m = LandModel(
        param_set,
        m_soil;
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    driver_config = ClimateMachine.SingleStackConfiguration(
        "LandModel",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        m;
        zmin = zmin,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
        #fv_reconstruction = FVLinear(),
    )


    t0 = FT(0)
    timeend = FT(60 * 60)
    dt = FT(0.1)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    state_types = (Prognostic(), Auxiliary(), GradientFlux())

    ClimateMachine.invoke!(solver_config;)# user_callbacks = (callback,))
    srf_dons = dict_of_nodal_states(solver_config, state_types; interp = true)


    ###### Repeat with explicit dirichlet BC
    soil_water_model2 = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = hydraulics,
        initialϑ_l = ϑ_l0,
    )
    soil_heat_model2 = PrescribedTemperatureModel()

    m_soil2 =
        SoilModel(soil_param_functions, soil_water_model2, soil_heat_model2)
    bc2 = LandDomainBC(
        bottom_bc = LandComponentBC(
            soil_water = Neumann((aux, t) -> eltype(aux)(0.0)),
        ),
        surface_bc = LandComponentBC(
            soil_water = Dirichlet((aux, t) -> eltype(aux)(0.495)),
        ),
    )


    m2 = LandModel(
        param_set,
        m_soil2;
        boundary_conditions = bc2,
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    driver_config2 = ClimateMachine.SingleStackConfiguration(
        "LandModel",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        m2;
        zmin = zmin,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
        #  fv_reconstruction = FVLinear(),
    )

    solver_config2 = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config2,
        ode_dt = dt,
    )

    ClimateMachine.invoke!(solver_config2;)
    dir_dons = dict_of_nodal_states(solver_config2, state_types; interp = true)
    # Here we take the solution near the surface, where changes between the two
    # would occur.
    @test mean(
        (
            srf_dons["soil.water.ϑ_l"][(end - 15):end] .-
            dir_dons["soil.water.ϑ_l"][(end - 15):end]
        ) .^ 2.0,
    ) < 5e-5
end

@testset "Explicit Flux BC Comparison" begin
    # This tests the else branch of compute_surface_grad_bc
    ClimateMachine.init()
    FT = Float64
    function init_soil_water!(land, state, aux, localgeo, time)
        myfloat = eltype(state)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
    end

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.495,
        Ksat = 0.0443 / (3600 * 100),
        S_s = 1e-4,
    )
    surface_precip_amplitude = FT(0)

    precip = (t) -> surface_precip_amplitude
    hydraulics = vanGenuchten{FT}(n = 2.0)
    function hydrostatic_profile(z, zm, porosity, n, α)
        myf = eltype(z)
        m = FT(1 - 1 / n)
        S = FT((FT(1) + (α * (z - zm))^n)^(-m))
        return FT(S * porosity)
    end

    ϑ_l0 =
        (aux) -> eltype(aux)(hydrostatic_profile(
            aux.z,
            -1,
            0.495,
            hydraulics.n,
            hydraulics.α,
        ))

    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = hydraulics,
        initialϑ_l = ϑ_l0,
    )
    soil_heat_model = PrescribedTemperatureModel()

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    N_poly = (1, 0)
    nelem_vert = 200

    # Specify the domain boundaries
    zmax = FT(0)
    zmin = FT(-1)
    Δz = FT(1 / 200 / 2)
    bc = LandDomainBC(
        bottom_bc = LandComponentBC(
            soil_water = Neumann((aux, t) -> eltype(aux)(0.0)),
        ),
        surface_bc = LandComponentBC(
            soil_water = SurfaceDrivenWaterBoundaryConditions(
                FT;
                precip_model = DrivenConstantPrecip{FT}(precip),
                runoff_model = CoarseGridRunoff{FT}(Δz),
            ),
        ),
    )


    m = LandModel(
        param_set,
        m_soil;
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    driver_config = ClimateMachine.SingleStackConfiguration(
        "LandModel",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        m;
        zmin = zmin,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
        fv_reconstruction = FVLinear(),
    )


    t0 = FT(0)
    timeend = FT(100000)
    dt = FT(4)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    state_types = (Prognostic(), Auxiliary(), GradientFlux())

    ClimateMachine.invoke!(solver_config;)

    srf_dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    # Note - we only look at differences between the solutions at the surface, because
    # near the bottom they will agree.
    z = srf_dons["z"][150:end]
    error1 = sqrt(mean(
        (
            srf_dons["soil.water.ϑ_l"][150:end] .-
            hydrostatic_profile.(z, -1, 0.495, hydraulics.n, hydraulics.α)
        ) .^ 2.0,
    ))
    error2 = sqrt(mean(srf_dons["soil.water.K∇h[3]"][150:end] .^ 2.0))
    @test error1 < 1e-5
    @test error2 < eps(FT)
end
