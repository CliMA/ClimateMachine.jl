using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using Test
using Pkg.Artifacts
using DelimitedFiles

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


@testset "hydrostatic test 1" begin
    ClimateMachine.init()
    FT = Float64

    function init_soil_water!(land, state, aux, localgeo, time)
        myfloat = eltype(aux)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
    end

    soil_heat_model = PrescribedTemperatureModel()

    Ksat = (aux) -> eltype(aux)(0.0443 / (3600 * 100))
    S_s = (aux) -> eltype(aux)((1e-3) * exp(-0.2 * aux.z))
    vgn = FT(2)
    wpf = WaterParamFunctions(FT; Ksat = Ksat, S_s = S_s)

    soil_param_functions = SoilParamFunctions(FT; porosity = 0.495, water = wpf)
    bottom_flux = (aux, t) -> eltype(aux)(0.0)
    surface_flux = bottom_flux
    ϑ_l0 = (aux) -> eltype(aux)(0.494)
    bc = LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann(bottom_flux)),
        surface_bc = LandComponentBC(soil_water = Neumann(surface_flux)),
    )
    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = vanGenuchten(FT; n = vgn),
        initialϑ_l = ϑ_l0,
    )

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    N_poly = 2
    nelem_vert = 20

    # Specify the domain boundaries
    zmax = FT(0)
    zmin = FT(-10)

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
    timeend = FT(60 * 60 * 24 * 200)

    dt = FT(500)
    n_outputs = 3
    every_x_simulation_time = ceil(Int, timeend / n_outputs)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    aux = solver_config.dg.state_auxiliary
    state_types = (Prognostic(), Auxiliary())
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
    z = dons_arr[1]["z"]
    interface_z = -1.0395
    function hydrostatic_profile(z, zm, porosity, n, α)
        myf = eltype(z)
        m = FT(1 - 1 / n)
        S = FT((FT(1) + (α * (z - zm))^n)^(-m))
        return FT(S * porosity)
    end
    function soln(z, interface, porosity, n, α, δ, S_s)
        if z < interface
            return porosity + S_s * (interface - z) * exp(-δ * z)
        else
            return hydrostatic_profile(z, interface, porosity, n, α)
        end
    end

    MSE = mean(
        (
            soln.(z, interface_z, 0.495, vgn, 2.6, 0.2, 1e-3) .-
            dons_arr[4]["soil.water.ϑ_l"]
        ) .^ 2.0,
    )
    @test MSE < 1e-4
end


@testset "hydrostatic test 2" begin
    ClimateMachine.init()
    FT = Float64

    function init_soil_water!(land, state, aux, localgeo, time)
        myfloat = eltype(aux)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
    end

    soil_heat_model = PrescribedTemperatureModel()

    Ksat = (4.42 / 3600 / 100)
    S_s = 1e-3
    wpf = WaterParamFunctions(FT; Ksat = Ksat, S_s = S_s)

    soil_param_functions = SoilParamFunctions(FT, porosity = 0.6, water = wpf)
    bottom_flux = (aux, t) -> eltype(aux)(0.0)
    surface_flux = bottom_flux
    bc = LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann(bottom_flux)),
        surface_bc = LandComponentBC(soil_water = Neumann(surface_flux)),
    )
    sigmoid(x, offset, width) =
        typeof(x)(exp((x - offset) / width) / (1 + exp((x - offset) / width)))
    function soln(z::f, interface::f, porosity::f) where {f}
        function hydrostatic_profile(
            z::f,
            interface::f,
            porosity::f,
            n::f,
            α::f,
            m::f,
        )
            ψ_interface = f(-1)
            ψ = -(z - interface) + ψ_interface
            S = (f(1) + (-α * ψ)^n)^(-m)
            return S * porosity
        end
        if z < interface
            return hydrostatic_profile(
                z,
                interface,
                porosity,
                f(1.31),
                f(1.9),
                f(1) - f(1) / f(1.31),
            )
        else
            return hydrostatic_profile(
                z,
                interface,
                porosity,
                f(1.89),
                f(7.5),
                f(1) - f(1) / f(1.89),
            )
        end
    end
    ϑ_l0 = (aux) -> soln(aux.z, -1.0, 0.6)

    vgα(aux) = aux.z < -1.0 ? 1.9 : 7.5
    vgn(aux) = aux.z < -1.0 ? 1.31 : 1.89

    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = vanGenuchten(FT; n = vgn, α = vgα),
        initialϑ_l = ϑ_l0,
    )

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    N_poly = 1
    nelem_vert = 80

    # Specify the domain boundaries
    zmax = FT(0)
    zmin = FT(-2)

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
    timeend = FT(60 * 60 * 12)

    dt = FT(5)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )

    ClimateMachine.invoke!(solver_config)

    state_types = (Prognostic(), Auxiliary())
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    z = dons["z"]

    RMSE =
        mean(
            (soln.(z, Ref(-1.0), Ref(0.6)) .- dons["soil.water.ϑ_l"]) .^ 2.0,
        )^0.5
    @test RMSE < 2.0 * eps(FT)
end
