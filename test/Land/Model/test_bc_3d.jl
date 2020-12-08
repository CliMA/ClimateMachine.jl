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

@testset "Boundary condition functions" begin
    ClimateMachine.init()

    FT = Float64

    function init_soil_water!(land, state, aux, localgeo, time)
        myfloat = eltype(state)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
    end

    soil_param_functions =
        SoilParamFunctions{FT}(porosity = 0.75, Ksat = 1e-7, S_s = 1e-3)
    bottom_flux_amplitude = FT(-3.0)
    f = FT(pi * 2.0 / 300.0)
    # nota bene: the flux is -K∇h
    bottom_flux =
        (aux, t) -> bottom_flux_amplitude * sin(f * t) * aux.soil.water.K
    surface_flux = nothing
    surface_state = (aux, t) -> eltype(aux)(0.2)
    bottom_state = nothing
    ϑ_l0 = (aux) -> eltype(aux)(0.2)
    bc = GeneralBoundaryConditions(
        Dirichlet(surface_state = surface_state, bottom_state = bottom_state),
        Neumann(surface_flux = surface_flux, bottom_flux = bottom_flux),
    )

    soil_water_model = SoilWaterModel(FT; initialϑ_l = ϑ_l0, boundaries = bc)
    soil_heat_model = PrescribedTemperatureModel()

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    N_poly = 5
    xres = FT(0.2)
    yres = FT(0.2)
    zres = FT(0.01)
    # Specify the domain boundaries.
    zmax = FT(0);
    zmin = FT(-1);
    xmax = FT(1)
    ymax = FT(1)

    driver_config = ClimateMachine.MultiColumnLandModel(
        "LandModel",
        (N_poly, N_poly),
        (xres, yres, zres),
        xmax,
        ymax,
        zmax,
        param_set,
        m;
        zmin = zmin,
    )


    t0 = FT(0)
    timeend = FT(300)
    dt = FT(0.5)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    n_outputs  = 30
    every_x_simulation_time = ceil(Int, timeend / n_outputs);

    state_types = (Auxiliary(), GradientFlux())
    dons_arr = Dict[dict_of_nodal_states(solver_config, state_types; interp = false)]
    time_data = FT[0] # store time data

    # We specify a function which evaluates `every_x_simulation_time` and returns
    # the state vector, appending the variables we are interested in into
    # `all_data`.
    
    callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
        dons = dict_of_nodal_states(solver_config, state_types; interp = false)
        push!(dons_arr, dons)
        push!(time_data, gettime(solver_config.solver))
        nothing
    end;
    
    # # Run the integration
    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));
    computed_bottom_∇h =
        [dons_arr[k]["soil.water.K∇h[3]"][1] for k in 2:n_outputs] ./ [dons_arr[k]["soil.water.K"][1] for k in 2:n_outputs]


    t = time_data[2:n_outputs]
    # we need a -1 out in front here because the flux BC is on -K∇h
    prescribed_bottom_∇h = t -> FT(-1) * FT(-3.0 * sin(pi * 2.0 * t / 300.0))

    MSE = mean((prescribed_bottom_∇h.(t) .- computed_bottom_∇h) .^ 2.0)
    computed_y1_∇h =
        maximum(abs.([dons_arr[k]["soil.water.K∇h[2]"][1] for k in 2:n_outputs] ./ [dons_arr[k]["soil.water.K"][1] for k in 2:n_outputs]))
    computed_x1_∇h =
        maximum(abs.([dons_arr[k]["soil.water.K∇h[1]"][1] for k in 2:n_outputs] ./ [dons_arr[k]["soil.water.K"][1] for k in 2:n_outputs]))
    @test MSE < 1e-4
    @test computed_x1_∇h < 1e-10
    @test computed_y1_∇h < 1e-10
end
