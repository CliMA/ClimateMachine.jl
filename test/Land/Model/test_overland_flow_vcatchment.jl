using Test
using Statistics
using DelimitedFiles

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SurfaceFlow
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state
using ArtifactWrappers


@testset "V Catchment Maxwell River Model" begin
    tv_dataset = ArtifactWrapper(
        @__DIR__,
        isempty(get(ENV, "CI", "")),
        "tiltedv",
        ArtifactFile[ArtifactFile(
            url = "https://caltech.box.com/shared/static/qi2gftjw2vu2j66b0tyfef427xxj3ug7.csv",
            filename = "TiltedVOutput.csv",
        ),],
    )
    tv_dataset_path = get_data_folder(tv_dataset)

    ClimateMachine.init()
    FT = Float64

    soil_water_model = PrescribedWaterModel()
    soil_heat_model = PrescribedTemperatureModel()
    soil_param_functions = nothing

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

    function x_slope(x, y)
        MFT = eltype(x)
        if x < MFT(800)
            MFT(-0.05)
        elseif x <= MFT(820)
            MFT(0)
        else
            MFT(0.05)
        end
    end

    function y_slope(x, y)
        MFT = eltype(x)
        MFT(-0.02)
    end

    function channel_mannings(x, y)
        MFT = eltype(x)
        return x >= MFT(800) && x <= MFT(820) ? MFT(2.5 * 60 * 10^-3) :
               MFT(2.5 * 60 * 10^-4)
    end

    m_surface = OverlandFlowModel(x_slope, y_slope; mannings = channel_mannings)

    bc = LandDomainBC(
        miny_bc = LandComponentBC(
            surface = Dirichlet((aux, t) -> eltype(aux)(0)),
        ),
        minx_bc = LandComponentBC(
            surface = Dirichlet((aux, t) -> eltype(aux)(0)),
        ),
        maxx_bc = LandComponentBC(
            surface = Dirichlet((aux, t) -> eltype(aux)(0)),
        ),
    )

    function init_land_model!(land, state, aux, localgeo, time)
        state.surface.height = eltype(state)(0)
    end

    # units in m / s 
    precip(x, y, t) = t < eltype(t)(90 * 60) ? eltype(t)(3e-6) : eltype(t)(0.0)

    sources = (Precip{FT}(precip),)

    m = LandModel(
        param_set,
        m_soil;
        surface = m_surface,
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_land_model!,
    )

    N_poly_hori = 1
    N_poly_vert = 1
    xres = FT(20)
    yres = FT(20)
    zres = FT(1)
    # Specify the domain boundaries.
    zmax = FT(1)
    zmin = FT(0)
    xmax = FT(1620)
    ymax = FT(1000)


    driver_config = ClimateMachine.MultiColumnLandModel(
        "LandModel",
        (N_poly_hori, N_poly_vert),
        (xres, yres, zres),
        xmax,
        ymax,
        zmax,
        param_set,
        m;
        zmin = zmin,
        numerical_flux_first_order = RusanovNumericalFlux(),
    )

    t0 = FT(0)
    timeend = FT(180 * 60)
    dt = FT(0.5)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    mygrid = solver_config.dg.grid
    Q = solver_config.Q

    h_index = varsindex(vars_state(m, Prognostic(), FT), :surface, :height)
    n_outputs = 60

    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    dons = Dict([k => Dict() for k in 1:n_outputs]...)

    iostep = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        h = Q[:, h_index, :]
        all_vars = Dict{String, Array}("t" => [t], "h" => h)
        dons[iostep[1]] = all_vars
        iostep[1] += 1
        return
    end

    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

    aux = solver_config.dg.state_auxiliary
    x = Array(aux[:, 1, :])
    y = Array(aux[:, 2, :])
    z = Array(aux[:, 3, :])
    # Get points at outlet (y = ymax)
    mask2 = (Float64.(y .== 1000.0)) .== 1
    n_outputs = length(dons)
    function compute_Q(h, xv)
        height = max.(h, 0.0)
        v = calculate_velocity(m_surface, xv, 1000.0, height)# put in y = 1000.0
        speed = sqrt(v[1]^2.0 + v[2]^2.0 + v[3]^2.0)
        Q_outlet = speed .* height .* 60.0 # multiply by 60 so it is per minute, not per second
        return Q_outlet
    end
    # We divide by 4 because we have 4 nodal points with the same value at each x (z = 0, 1)
    # Multiply by xres because the solution at each point roughly represents that the solution for that range in x
    Q =
        [
            sum(compute_Q.(Array(dons[k]["h"])[:][mask2[:]], x[mask2[:]]))
            for k in 1:n_outputs
        ] ./ 4.0 .* xres
    data = joinpath(tv_dataset_path, "TiltedVOutput.csv")
    ds_tv = readdlm(data, ',')
    error = sqrt(mean(Q .- ds_tv))
    @test error < 5e-7
end
