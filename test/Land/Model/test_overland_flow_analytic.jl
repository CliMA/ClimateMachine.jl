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
using ClimateMachine.Land.SurfaceFlow
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


# Test that the land model with no surface flow works correctly
@testset "NoSurfaceFlow Model" begin
    function init_land_model!(land, state, aux, localgeo, time) end
    ClimateMachine.init()
    FT = Float64

    soil_water_model = PrescribedWaterModel()
    soil_heat_model = PrescribedTemperatureModel()
    soil_param_functions = nothing

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    m_surface = NoSurfaceFlowModel()

    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        surface = m_surface,
        source = sources,
        init_state_prognostic = init_land_model!,
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
    timeend = FT(10)
    dt = FT(1)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary

    ClimateMachine.invoke!(solver_config;)

    t = ODESolvers.gettime(solver_config.solver)
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
    #Make sure it runs, and that there are no state variables, and only "x,y,z" as aux.
    @test t == timeend
    @test size(Base.collect(keys(aux_vars)))[1] == 3
    @test size(Base.collect(keys(state_vars)))[1] == 0
end


# Constant slope analytical test case defined as Model 1 / Eqn 6
# DOI: 10.1061/(ASCE)0733-9429(2007)133:2(217)
@testset "Analytical Overland Model" begin
    function warp_constant_slope(
        xin,
        yin,
        zin;
        topo_max = 0.2,
        zmin = -0.1,
        xmax = 400,
    )
        FT = eltype(xin)
        zmax = FT((FT(1.0) - xin / xmax) * topo_max)
        alpha = FT(1.0) - zmax / zmin
        zout = zmin + (zin - zmin) * alpha
        x, y, z = xin, yin, zout
        return x, y, z
    end
    ClimateMachine.init()
    FT = Float64

    soil_water_model = PrescribedWaterModel()
    soil_heat_model = PrescribedTemperatureModel()
    soil_param_functions = nothing

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

    m_surface = OverlandFlowModel(
        (x, y) -> eltype(x)(-0.0016),
        (x, y) -> eltype(x)(0.0);
        mannings = (x, y) -> eltype(x)(0.025),
    )


    bc = LandDomainBC(
        minx_bc = LandComponentBC(
            surface = Dirichlet((aux, t) -> eltype(aux)(0)),
        ),
    )

    function init_land_model!(land, state, aux, localgeo, time)
        state.surface.height = eltype(state)(0)
    end

    # units in m / s 
    precip(x, y, t) = t < (30 * 60) ? 1.4e-5 : 0.0

    sources = (Precip{FT}(precip),)

    m = LandModel(
        param_set,
        m_soil;
        surface = m_surface,
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_land_model!,
    )

    N_poly = 1
    xres = FT(2.286)
    yres = FT(0.25)
    zres = FT(0.1)
    # Specify the domain boundaries.
    zmax = FT(0)
    zmin = FT(-0.1)
    xmax = FT(182.88)
    ymax = FT(1.0)
    topo_max = FT(0.0016 * xmax)

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
        meshwarp = (x...) -> warp_constant_slope(
            x...;
            topo_max = topo_max,
            zmin = zmin,
            xmax = xmax,
        ),
    )

    t0 = FT(0)
    timeend = FT(60 * 60)
    dt = FT(10)

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

    # Compare flowrate analytical derivation
    aux = solver_config.dg.state_auxiliary

    # get all nodal points at the max X bound of the domain
    mask = Array(aux[:, 1, :] .== 182.88)
    n_outputs = length(dons)
    # get prognostic variable height from nodal state (m^2)
    height = [mean(Array(dons[k]["h"])[mask[:]]) for k in 1:n_outputs]
    # get similation timesteps (s)
    time_data = [dons[l]["t"][1] for l in 1:n_outputs]


    alpha = sqrt(0.0016) / 0.025
    i = 1.4e-5
    L = xmax
    m = 5 / 3
    t_c = (L * i^(1 - m) / alpha)^(1 / m)
    t_r = 30 * 60

    q = height .^ (m) .* alpha

    function g(m, y, i, t_r, L, alpha, t)
        output = L / alpha - y^(m) / i - y^(m - 1) * m * (t - t_r)
        return output
    end

    function dg(m, y, i, t_r, L, alpha, t)
        output = -y^(m - 1) * m / i - y^(m - 2) * m * (m - 1) * (t - t_r)
        return output
    end

    function analytic(t, alpha, t_c, t_r, i, L, m)
        if t < t_c
            return alpha * (i * t)^(m)
        end

        if t <= t_r && t > t_c
            return alpha * (i * t_c)^(m)
        end

        if t > t_r
            yL = (i * (t - t_r))
            delta = 1
            error = g(m, yL, i, t_r, L, alpha, t)
            while abs(error) > 1e-4
                delta =
                    -g(m, yL, i, t_r, L, alpha, t) /
                    dg(m, yL, i, t_r, L, alpha, t)
                yL = yL + delta
                error = g(m, yL, i, t_r, L, alpha, t)
            end
            return alpha * yL^m

        end

    end

    q = Array(q) # copy to host if GPU array
    @test sqrt_rmse_over_max_q =
        sqrt(mean(
            (analytic.(time_data, alpha, t_c, t_r, i, L, m) .- q) .^ 2.0,
        )) / maximum(q) < 3e-3
end
