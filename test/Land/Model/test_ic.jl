# Test that the way we specify boundary conditions works as expected
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test
using Plots

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

ClimateMachine.init()

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

    FT = Float64

    function init_soil_water!(land, state, aux, localgeo, time)
        myfloat = eltype(state)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
    end

    # Sand van Genuchten parameters, Bonan19, table 8.3, p. 151
    porosity = FT(0.43)
    α = FT(14.5) #m-1
    n = FT(2.68) #unitless
    Ksat = FT(0.0000825) #ms-1 (0.2970/(60^2))

    # # Clay van Genuchten parameters
    # porosity = FT(0.38)
    # α = FT(0.8) #m-1
    # n = FT(1.09) #unitless
    # Ksat = FT(5.56e-7) #ms-1 (0.0020/(60^2))

    soil_param_functions =
        SoilParamFunctions{FT}(porosity = porosity, Ksat = Ksat, S_s = 1e-3)
    # nota bene: the flux is -K∇h
   # ϑ_l0 = (aux) -> eltype(aux)(porosity-0.1)
    #ϑ_l0 = FT(soil_param_functions.porosity)

    bottom_flux = (aux, t) -> FT(0.0)
    # Δz = min_node_distance(driver_config.grid)  # "Vertical resolution at the surface"
    # soil_hydraulics = vanGenuchten{FT}(n = n, α = α)
    # θ_l = volumetric_liquid_fraction(ϑ_l0, soil_param_functions.porosity)
    # S = effective_saturation(
    #     soil_param_functions.porosity,
    #     θ_l,
    # )
    # icflux = -Ksat*(FT(1)-matric_potential(soil_hydraulics,S)/Δz)
    # surface_flux = (aux, t) -> FT(icflux)
    surface_state = (aux, t) -> eltype(aux)(porosity)

    # bc = GeneralBoundaryConditions(
    #     Dirichlet(surface_state = nothing, bottom_state = nothing),
    #     Neumann(surface_flux = surface_flux, bottom_flux = bottom_flux),
    # )
    # ϑ_l0 = (aux) -> eltype(aux)(soil_param_functions.porosity-0.1)

    bc = GeneralBoundaryConditions(
        Dirichlet(surface_state = surface_state, bottom_state = nothing),
        Neumann(surface_flux = nothing, bottom_flux = bottom_flux),
    )
    ϑ_l0 = (aux) -> eltype(aux)(soil_param_functions.porosity-0.2)

    #soil_water_model = SoilWaterModel(FT; initialϑ_l = ϑ_l0, boundaries = bc)
    soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = n, α = α),
    initialϑ_l = ϑ_l0,
    boundaries = bc,
);
    soil_heat_model = PrescribedTemperatureModel()

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        source = sources,
        init_state_prognostic = init_soil_water!,
    )

    N_poly = 1
    nelem_vert = 10

    # Specify the domain boundaries
    zmax = FT(0)
    zmin = FT(-0.1)

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

    # Choose the initial and final times, as well as a timestep.
    t0 = FT(0)
    timeend = FT(60*5)
    dt = FT(0.001)

    # Create the solver configuration.
    solver_config =
        ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

    # Determine how often you want output.
    const n_outputs = 5;

    const every_x_simulation_time = ceil(Int, timeend / n_outputs);

    state_types = (Prognostic(), Auxiliary(), GradientFlux())
    all_data = Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0] # store time data

    # We specify a function which evaluates `every_x_simulation_time` and returns
    # the state vector, appending the variables we are interested in into
    # `all_data`.

    callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
        dons = dict_of_nodal_states(solver_config, state_types; interp = true)
        push!(all_data, dons)
        push!(time_data, gettime(solver_config.solver))
        nothing
    end;

    # # Run the integration
    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

    # Get the final state and create plots:
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(all_data, dons)
    push!(time_data, gettime(solver_config.solver));

    # Get z-coordinate
    z = get_z(solver_config.dg.grid; rm_dupes = true);
    # # # Create some plots

    output_dir = @__DIR__;

    t = time_data; #./ (60);

    plot(
        all_data[1]["soil.water.ϑ_l"],
        all_data[1]["z"],
        label = string("t = ", string(t[1]), "s"),
        #xlim = [0.47, 0.501],
        ylabel = "z",
        xlabel = "ϑ_l",
        legend = :bottomleft,
        title = "IC test",
    );
    plot!(
        all_data[4]["soil.water.ϑ_l"],
        all_data[4]["z"],
        label = string("t = ", string(t[2]), "s"),
    );
    plot!(
        all_data[6]["soil.water.ϑ_l"],
        all_data[6]["z"],
        label = string("t = ", string(t[6]), "s"),
    );

    savefig(joinpath(output_dir, "ic_test_dirichlet_vol_wat_content.png"))

        plot(
        all_data[1]["soil.water.K∇h[3]"],
        all_data[1]["z"],
        label = string("t = ", string(t[1]), "s"),
        #xlim = [0.47, 0.501],
        ylabel = "z",
        xlabel = "K∇h",
        legend = :bottomright,
        title = "IC test",
    );
    plot!(
        all_data[4]["soil.water.K∇h[3]"],
        all_data[4]["z"],
        label = string("t = ", string(t[2]), "s"),
    );
    plot!(
        all_data[6]["soil.water.K∇h[3]"],
        all_data[6]["z"],
        label = string("t = ", string(t[6]), "s"),
    );
    plot!(
        soil_param_functions.Ksat*ones(length(all_data[4]["soil.water.K∇h[3]"]),1),
        all_data[6]["z"],
        label = string("K_sat =", string(soil_param_functions.Ksat), "ms-1"),
    );


    # save the output.
    #savefig(joinpath(output_dir, "ic_test_2_bc_neumann_ic_overleaf.png"))
    savefig(joinpath(output_dir, "ic_test_dirichlet_infiltration.png"))

    # solver_config = ClimateMachine.SolverConfiguration(
    #     t0,
    #     timeend,
    #     driver_config,
    #     ode_dt = dt,
    # )
    # mygrid = solver_config.dg.grid
    # Q = solver_config.Q
    # aux = solver_config.dg.state_auxiliary
    # grads = solver_config.dg.state_gradient_flux
    # z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
    # K∇h_vert_ind =
    #     varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[3]
    # K_ind = varsindex(vars_state(m, Auxiliary(), FT), :soil, :water, :K)
    # n_outputs = 5

    # every_x_simulation_time = ceil(Int, timeend / n_outputs)

    # # Create a place to store this output.
    # mygrid = solver_config.dg.grid;
    # Q = solver_config.Q;
    # aux = solver_config.dg.state_auxiliary;
    # grads = solver_config.dg.state_gradient_flux

    # z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
    # ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
    # K∇h_vert_ind = varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[3]

    # z = aux[:, z_ind, :][:]
    # ϑ_l = Q[:, ϑ_l_ind, :][:]
    # K∇h_vert = zeros(length(ϑ_l)) .+ FT(NaN)

    # all_data = [Dict{String, Array}("ϑ_l" => ϑ_l, "K∇h" => K∇h_vert, "z" => z)]

    # time_data = FT[0] # store time data

    # callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    # z = aux[:, z_ind, :][:]
    # ϑ_l = Q[:, ϑ_l_ind, :][:]
    # K∇h_vert = grads[:, K∇h_vert_ind, :][:]

    # dons = Dict{String, Array}("ϑ_l" => ϑ_l, "K∇h" => K∇h_vert, "z" => z)
    # push!(all_data, dons)
    # push!(time_data, gettime(solver_config.solver))
    # nothing
    # end;

    # # # Run the integration
    # ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

    # # Get the final state and create plots:
    # z = aux[:, z_ind, :][:]
    # ϑ_l = Q[:, ϑ_l_ind, :][:]
    # K∇h_vert = grads[:, K∇h_vert_ind, :][:]

    # dons = Dict{String, Array}("ϑ_l" => ϑ_l, "K∇h" => K∇h_vert, "z" => z)
    # push!(all_data, dons)
    # push!(time_data, gettime(solver_config.solver))

    # # # # Create some plots

    # output_dir = @__DIR__;

    # t = time_data; #./ (60);

    # ratio1 = round(FT(porosity - 0.1)/porosity, digits=2)

    # plot(
    #     soil_param_functions.Ksat*ones(length(all_data[4]["K∇h"]),1),
    #     all_data[4]["z"],
    #     label = string("K_sat =", string(soil_param_functions.Ksat), "ms-1"),
    #     legend = :bottomright,
    # );


    # plot!(
    #     all_data[2]["K∇h"],
    #     all_data[2]["z"],
    #     label = string("t = ", string(t[2]), "sec"),
    # );

    # plot!(
    #     all_data[4]["K∇h"],
    #     all_data[4]["z"],
    #     label = string("t = ", string(t[4]), "sec"),
    #     ylabel = "z",
    #     xlabel = "K∇h",
    #     title = string("Maxwell Infiltration with", " IC/porosity =", string(ratio1), ", simtime =", string(timeend), "s"),
    # );

######

