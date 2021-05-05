# Test that freeze thaw alone reproduces expected behavior: exponential behavior
# for liquid water content, ice content, and total water conserved

using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
using CLIMAParameters.Planet: ρ_cloud_liq
using CLIMAParameters.Planet: ρ_cloud_ice

using PlantHydraulics

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.SoilHeatParameterizations
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

    import ClimateMachine.DGMethods.FVReconstructions: FVLinear # diffeerent spat. disc insead of dg
#testset "Freeze thaw alone" begin
    struct tmp_model <: BalanceLaw end
    struct tmp_param_set <: AbstractParameterSet end

    function init_soil_water!(land, state, aux, coordinates, time)
        state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
    end

    FT = Float64
    ClimateMachine.init()

    N_poly = 1 # set to 1 to use finite volume in the vertical because linear discr metho
    nelem_vert = 1# 6 points n_poly^2 * nelems
    zmax = FT(0)
    zmin = FT(-1)
    t0 = FT(0)
    timeend = FT(10)
    dt = FT(0.05)
    A = FT(1)

    n_outputs = 10
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    roots_source = Roots{FT}(A = A)

    soil_param_functions =
        SoilParamFunctions{FT}(porosity = 0.75, Ksat = 0.0, S_s = 1e-3)

    bottom_flux = (aux, t) -> eltype(aux)(0.0)
    surface_flux = (aux, t) -> eltype(aux)(0.0)
    bc = LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann(bottom_flux)),
        surface_bc = LandComponentBC(soil_water = Neumann(surface_flux)),
    )
    ϑ_l0 = (aux) -> eltype(aux)(0.4)
    θ_i0 = (aux) -> eltype(aux)(0.0)

    soil_water_model = SoilWaterModel(FT; initialϑ_l = ϑ_l0, initialθ_i = θ_i0)

    soil_heat_model =
        PrescribedTemperatureModel((aux, t) -> eltype(aux)(300))

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = (roots_source,)
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

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    state_types = (Prognostic(),)
    dons_arr =
        Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0]
    callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
        dons = dict_of_nodal_states(solver_config, state_types; interp = true)
        push!(dons_arr, dons)
        push!(time_data, gettime(solver_config.solver))
        nothing
    end
    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))
    m_liq_numeric = dons_arr[n_outputs + 1]["soil.water.ϑ_l"][end]
    m_liq_analytic =  (1e-10 - 1 / A) + 1 / A * exp(A * timeend)
    #@test m_liq_numeric .- m_liq_analytic < 1e-9
#end
