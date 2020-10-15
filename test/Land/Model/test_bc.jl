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
    soil_water_model = SoilWaterModel(
        FT;
        initialϑ_l = ϑ_l0,
        dirichlet_bc = Dirichlet(
            surface_state = surface_state,
            bottom_state = bottom_state,
        ),
        neumann_bc = Neumann(
            surface_flux = surface_flux,
            bottom_flux = bottom_flux,
        ),
    )
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
    timeend = FT(300)
    dt = FT(0.05)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary
    grads = solver_config.dg.state_gradient_flux
    K∇h_vert_ind =
        varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[3]
    K_ind = varsindex(vars_state(m, Auxiliary(), FT), :soil, :water, :K)
    n_outputs = 30

    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    all_data = Dict([k => Dict() for k in 1:n_outputs]...)

    iostep = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        K = aux[:, K_ind, :]
        K∇h_vert = grads[:, K∇h_vert_ind, :]
        all_vars = Dict{String, Array}(
            "t" => [t],
            "K" => K,
            "K∇h_vert" => K∇h_vert,
        )
        all_data[iostep[1]] = all_vars
        iostep[1] += 1
        nothing
    end

    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))
    t = ODESolvers.gettime(solver_config.solver)
    K = aux[:, K_ind, :]
    K∇h_vert = grads[:, K∇h_vert_ind, :]
    all_vars = Dict{String, Array}("t" => [t], "K" => K, "K∇h_vert" => K∇h_vert)
    all_data[n_outputs] = all_vars


    computed_bottom_∇h =
        [all_data[k]["K∇h_vert"][1] for k in 1:n_outputs] ./ [all_data[k]["K"][1] for k in 1:n_outputs]


    t = [all_data[k]["t"][1] for k in 1:n_outputs]
    # we need a -1 out in front here because the flux BC is on -K∇h
    prescribed_bottom_∇h = t -> FT(-1) * FT(-3.0 * sin(pi * 2.0 * t / 300.0))

    MSE = mean((prescribed_bottom_∇h.(t) .- computed_bottom_∇h) .^ 2.0)
    @test MSE < 1e-7
end
