using MPI
using OrderedCollections
using StaticArrays
using Test
using Statistics
using DelimitedFiles

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SnowModel
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
using ClimateMachine.ArtifactWrappers

#@testset "Snow Model" begin

    ClimateMachine.init()
    FT = Float64

    soil_water_model = PrescribedWaterModel()
    soil_heat_model = PrescribedTemperatureModel()
    soil_param_functions = nothing

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

    snow_parameters = SnowParameters{FT,FT,FT,FT}(0.0,0.0,10.0)# κsnow = 0, ρsnow = 0, zsnow = 1m
    Q_surf = (t) -> eltype(t)(900.0*sin(2.0*π/3600/24*t))
    forcing = PrescribedForcing(FT;Q_surf = Q_surf)
    ic = (aux) -> eltype(aux)(-1.8e5)
    m_snow = SingleLayerSnowModel{typeof(snow_parameters), typeof(forcing),typeof(ic)}(
        snow_parameters,
        forcing,
        ic
    )


    function init_land_model!(land, state, aux, localgeo, time)
        state.snow.ρe_int = land.snow.initial_ρe_int(aux)
    end

    sources = (FluxDivergence{FT}(),)

    m = LandModel(
        param_set,
        m_soil;
        snow = m_snow,
        source = sources,
        init_state_prognostic = init_land_model!,
    )

    N_poly = 1
    nelem_vert = 1

    # Specify the domain boundaries
    zmax = FT(10)
    zmin = FT(0)

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
    timeend = FT(60 * 60 * 48)
    dt = FT(60*60)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    n_outputs = 30;
    
    every_x_simulation_time = ceil(Int, (timeend-t0) / n_outputs);
    
    # Create a place to store this output.
    state_types = (Prognostic(),)
    dons_arr = Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0] # store time data
    
    callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
        dons = dict_of_nodal_states(solver_config, state_types; interp = true)
        push!(dons_arr, dons)
        push!(time_data, gettime(solver_config.solver))
        nothing
    end;
    
    # # Run the integration
    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));
    z = get_z(solver_config.dg.grid; rm_dupes = true);
    N = length(dons_arr)

#end
