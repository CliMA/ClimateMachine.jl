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
    mpicomm = MPI.COMM_WORLD

    FT = Float64

    function init_soil_water!(land, state, aux, coordinates, time)
        FT = eltype(state)
        state.soil.water.ϑ_l = FT(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_ice = FT(land.soil.water.initialθ_ice(aux))
    end

    soil_param_functions =
        SoilParamFunctions(porosity = 0.75, Ksat = 1e-7, S_s = 1e-3)
    bottom_flux =
        (aux, t) -> FT(-3.0 * sin(pi * 2.0 * t / 300.0) * aux.soil.water.K)
    surface_flux = nothing
    surface_state = (aux, t) -> FT(0.2)
    bottom_state = nothing
    ϑ_l0 = (aux) -> FT(0.2)
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

    soil_heat_model = PrescribedTemperatureModel{FT}()

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

    n_outputs = 30

    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    all_data = Dict([k => Dict() for k in 1:n_outputs]...)

    step = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        grads = SingleStackUtils.get_vars_from_nodal_stack(
            mygrid,
            solver_config.dg.state_gradient_flux,
            vars_state(m, GradientFlux(), FT),
        )
        state_vars = SingleStackUtils.get_vars_from_nodal_stack(
            mygrid,
            Q,
            vars_state(m, Prognostic(), FT),
        )
        aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
            mygrid,
            aux,
            vars_state(m, Auxiliary(), FT),
        )
        all_vars = OrderedDict(state_vars..., aux_vars..., grads...)
        all_vars["t"] = [t]
        all_data[step[1]] = all_vars

        step[1] += 1
        nothing
    end

    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

    t = ODESolvers.gettime(solver_config.solver)
    state_vars = SingleStackUtils.get_vars_from_nodal_stack(
        mygrid,
        Q,
        vars_state(m, Prognostic(), FT),
    )
    grads = SingleStackUtils.get_vars_from_nodal_stack(
        mygrid,
        solver_config.dg.state_gradient_flux,
        vars_state(m, GradientFlux(), FT),
    )
    aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
        mygrid,
        aux,
        vars_state(m, Auxiliary(), FT),
    )
    all_vars = OrderedDict(state_vars..., aux_vars..., grads...)
    all_vars["t"] = [t]
    all_data[n_outputs] = all_vars


    computed_bottom_∇h =
        [all_data[k]["soil.water.K∇h[3]"][1] for k in 1:n_outputs] ./ [all_data[k]["soil.water.K"][1] for k in 1:n_outputs]


    t = [all_data[k]["t"][1] for k in 1:n_outputs]
    prescribed_bottom_∇h = t -> FT(-3.0 * sin(pi * 2.0 * t / 300.0))

    MSE = mean((prescribed_bottom_∇h.(t) .- computed_bottom_∇h) .^ 2.0)
    @test MSE < 1e-7
end
