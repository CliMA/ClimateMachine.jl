# Test that Richard's equation agrees with solution from Bonan's book,
# simulation 8.2
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
using ClimateMachine.ArtifactWrappers

haverkamp_dataset = ArtifactWrapper(
    joinpath("test", "Land", "Model", "Artifacts.toml"),
    "richards",
    ArtifactFile[ArtifactFile(
        url = "https://caltech.box.com/shared/static/dfijf07io7h5dk1k87saaewgsg9apq8d.csv",
        filename = "bonan_haverkamp_data.csv",
    ),],
)
haverkamp_dataset_path = get_data_folder(haverkamp_dataset)

@testset "Richard's equation - Haverkamp test" begin
    ClimateMachine.init()
    FT = Float64

    function init_soil_water!(land, state, aux, localgeo, time)
        myfloat = eltype(aux)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
    end

    soil_heat_model = PrescribedTemperatureModel()

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.495,
        Ksat = 0.0443 / (3600 * 100),
        S_s = 1e-3,
    )
    # Keep in mind that what is passed is aux⁻.
    # Fluxes are multiplied by ẑ (normal to the surface, -normal to the bottom,
    # where normal point outs of the domain.)


    surface_state = (aux, t) -> eltype(aux)(0.494)
    # The goal here is to have ∇h = ẑ enforced by the BC
    # the BC is on K∇h, and multiplied by ẑ internally.
    bottom_flux = (aux, t) -> aux.soil.water.K * eltype(aux)(-1)
    ϑ_l0 = (aux) -> eltype(aux)(0.24)

    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = Haverkamp{FT}(),
        initialϑ_l = ϑ_l0,
        dirichlet_bc = Dirichlet(
            surface_state = surface_state,
            bottom_state = nothing,
        ),
        neumann_bc = Neumann(surface_flux = nothing, bottom_flux = bottom_flux),
    )

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    N_poly = 5
    nelem_vert = 10

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
    timeend = FT(60 * 60 * 24)

    dt = FT(6)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary

    ClimateMachine.invoke!(solver_config)
    ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
    ϑ_l = Array(Q[:, ϑ_l_ind, :][:])
    z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
    z = Array(aux[:, z_ind, :][:])

    # Compare with Bonan simulation data at 1 day.
    data = joinpath(haverkamp_dataset_path, "bonan_haverkamp_data.csv")
    ds_bonan = readdlm(data, ',')
    bonan_moisture = reverse(ds_bonan[:, 1])
    bonan_z = reverse(ds_bonan[:, 2]) ./ 100.0


    # Create an interpolation from the Bonan data
    bonan_moisture_continuous = Spline1D(bonan_z, bonan_moisture)
    bonan_at_clima_z = bonan_moisture_continuous.(z)
    MSE = mean((bonan_at_clima_z .- ϑ_l) .^ 2.0)
    @test MSE < 1e-5
end
