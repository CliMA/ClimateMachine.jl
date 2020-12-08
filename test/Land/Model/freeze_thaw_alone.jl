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

@testset "Freeze thaw alone" begin
    struct tmp_model <: BalanceLaw end
    struct tmp_param_set <: AbstractParameterSet end

    function get_grid_spacing(
        N_poly::Int64,
        nelem_vert::Int64,
        zmax::FT,
        zmin::FT,
    ) where {FT}
        test_config = ClimateMachine.SingleStackConfiguration(
            "TmpModel",
            N_poly,
            nelem_vert,
            zmax,
            tmp_param_set(),
            tmp_model();
            zmin = zmin,
        )

        Δ = min_node_distance(test_config.grid)
        return Δ
    end

    function init_soil_water!(land, state, aux, coordinates, time)
        state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
    end

    FT = Float64
    ClimateMachine.init()

    N_poly = 5
    nelem_vert = 50
    zmax = FT(0)
    zmin = FT(-1)
    t0 = FT(0)
    timeend = FT(30)
    dt = FT(0.05)

    n_outputs = 30
    every_x_simulation_time = ceil(Int, timeend / n_outputs)
    Δ = get_grid_spacing(N_poly, nelem_vert, zmax, zmin)
    cs = FT(3e6)
    κ = FT(1.5)
    τLTE = FT(cs * Δ^FT(2.0) / κ)
    freeze_thaw_source = PhaseChange{FT}(Δt = dt, τLTE = τLTE)

    soil_param_functions =
        SoilParamFunctions{FT}(porosity = 0.75, Ksat = 0.0, S_s = 1e-3)

    bottom_flux = (aux, t) -> eltype(aux)(0.0)
    surface_flux = (aux, t) -> eltype(aux)(0.0)
    surface_state = nothing
    bottom_state = nothing

    bc = GeneralBoundaryConditions(
        Dirichlet(surface_state = surface_state, bottom_state = bottom_state),
        Neumann(surface_flux = surface_flux, bottom_flux = bottom_flux),
    )
    ϑ_l0 = (aux) -> eltype(aux)(1e-10)
    θ_i0 = (aux) -> eltype(aux)(0.33)

    soil_water_model = SoilWaterModel(
        FT;
        initialϑ_l = ϑ_l0,
        initialθ_i = θ_i0,
        boundaries = bc,
    )

    soil_heat_model =
        PrescribedTemperatureModel((aux, t) -> eltype(aux)(276.15))

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = (freeze_thaw_source,)
    m = LandModel(
        param_set,
        m_soil;
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
    m_liq = [
        ρ_cloud_liq(param_set) * mean(dons_arr[k]["soil.water.ϑ_l"])
        for k in 1:(n_outputs + 1)
    ]
    m_ice = [
        ρ_cloud_ice(param_set) * mean(dons_arr[k]["soil.water.θ_i"])
        for k in 1:(n_outputs + 1)
    ]
    total_water = m_ice + m_liq
    τft = max(dt, τLTE)
    m_ice_of_t = m_ice[1] * exp.(-1.0 .* (time_data .- time_data[1]) ./ τft)
    m_liq_of_t = -m_ice_of_t .+ (m_liq[1] + m_ice[1])
    @test mean(abs.(m_liq .- m_liq_of_t)) < 1e-9
    @test mean(abs.(m_ice .- m_ice_of_t)) < 1e-9
end
