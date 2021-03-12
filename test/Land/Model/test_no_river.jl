# Test that the land model still runs, even with the lowest/simplest
# version of soil (prescribed heat and prescribed water - no state
# variables)
using MPI
using OrderedCollections
using StaticArrays
using Test
using Statistics

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.River
using ClimateMachine.Land.Runoff
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state


@testset "NoRiver Model" begin

    function init_land_model!(land, state, aux, localgeo, time) 
    end

    ClimateMachine.init()
    FT = Float64

    soil_water_model = PrescribedWaterModel()
    soil_heat_model = PrescribedTemperatureModel()
    soil_param_functions = nothing

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    m_river = NoRiverModel()

    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        river = m_river,
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
    timeend = FT(60)
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
    #Make sure it runs, and that there are no state variables, and only "z" as aux.
    @test t == timeend
    @test size(Base.collect(keys(aux_vars)))[1] == 3
    @test size(Base.collect(keys(state_vars)))[1] == 0
end

function warp_constant_slope(xin, yin, zin; topo_max = 0.2, zmin = -0.1, xmax = 400)
    FT = eltype(xin)
    zmax = FT((FT(1.0)-xin / xmax) * topo_max)
    alpha = FT(1.0) - zmax / zmin
    zout = zmin + (zin - zmin) * alpha
    x, y, z = xin, yin, zout
    return x, y, z
end
    
function inverse_constant_slope(xin, yin, zin; topo_max = 0.2, zmin =  -5, xmax = 400)
    FT = eltype(xin)
    zmax = FT((FT(1.0)-xin / xmax) * topo_max)
    alpha = FT(1.0) - zmax / zmin
    zout = (zin-zmin)/alpha+zmin
    return zout
end

@testset "Analytical River Model" begin
    ClimateMachine.init()
    FT = Float64

    soil_water_model = PrescribedWaterModel()
    soil_heat_model = PrescribedTemperatureModel()
    soil_param_functions = nothing

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    
    m_river = RiverModel(
        (x,y) -> eltype(x)(-0.0016),
        (x,y) -> eltype(x)(0.0),
        (x,y) -> eltype(x)(1);
        mannings = (x,y) -> eltype(x)(0.025)
    )
    
    # Analytical test case defined as Model 1 in DOI:
    # 10.1061/(ASCE)0733-9429(2007)133:2(217) 
    # Eqn 6
    bc = LandDomainBC(
        minx_bc = LandComponentBC(river = Dirichlet((aux, t) -> eltype(aux)(0))),
    )
 
    function init_land_model!(land, state, aux, localgeo, time)
        state.river.area = eltype(state)(0)
    end

    # units in m / s 
    precip(x, y, t) = t < (30 * 60) ? 1.4e-5 : 0.0

    sources = (Precip{FT}(precip),)

    m = LandModel(
        param_set,
        m_soil;
        river = m_river,
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_land_model!,
    )

    N_poly = 1;
    xres = FT(2.286)
    yres = FT(0.25)
    zres = FT(0.1)
    # Specify the domain boundaries.
    zmax = FT(0);
    zmin = FT(-0.1);
    xmax = FT(182.88)
    ymax = FT(1.0)
    topo_max = FT(0.0016*xmax)

    driver_config = ClimateMachine.MultiColumnLandModel(
        "LandModel",
        (N_poly, N_poly),
        (xres,yres,zres),
        xmax,
        ymax,
        zmax,
        param_set,
        m;
        zmin = zmin,
        meshwarp = (x...) -> warp_constant_slope(x...;
        topo_max = topo_max, zmin = zmin, xmax = xmax),
    );

    t0 = FT(0)
    timeend = FT(60*60)
    dt = FT(10)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    
    area_index =
        varsindex(vars_state(m, Prognostic(), FT), :river, :area)
    n_outputs = 60

    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    dons = Dict([k => Dict() for k in 1:n_outputs]...)

    iostep = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        area = Q[:, area_index, :]
        all_vars = Dict{String, Array}(
            "t" => [t],
            "area" => area,
        )
        dons[iostep[1]] = all_vars
        iostep[1] += 1
        return
    end

    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

    # Compare flowrate analytical derivation
    aux = solver_config.dg.state_auxiliary;
     
    # get all nodal points at the max X bound of the domain
    mask = Array(aux[:,1,:] .== 182.88)
    n_outputs = length(dons)
    # get prognostic variable area from nodal state (m^2)
    area = [mean(Array(dons[k]["area"])[mask[:]]) for k in 1:n_outputs]
    height = area ./ ymax
    # get similation timesteps (s)
    time_data = [dons[l]["t"][1] for l in 1:n_outputs]
   

    alpha = sqrt(0.0016)/0.025
    i = 1.4e-5
    L = xmax
    m = 5/3
    t_c = (L*i^(1-m)/alpha)^(1/m)
    t_r = 30*60
    
    q = height.^(m) .* alpha
    
    function g(m,y, i, t_r, L, alpha, t)
        output = L/alpha-y^(m)/i-y^(m-1)*m*(t-t_r)
        return output
    end
    
    function dg(m,y, i, t_r, L, alpha, t)
        output = -y^(m-1)*m/i-y^(m-2)*m*(m-1)*(t-t_r)
        return output
    end
    
    function analytic(t,alpha, t_c, t_r, i, L, m)
        if t < t_c
            return alpha*(i*t)^(m)
        end
        
        if t <= t_r && t > t_c
            return alpha*(i*t_c)^(m)
        end
        
        if t > t_r
            yL = (i*(t-t_r))
            delta = 1
            error = g(m,yL,i,t_r,L,alpha,t)
            while abs(error) > 1e-4
                delta = -g(m,yL,i,t_r,L,alpha,t)/dg(m,yL,i,t_r,L,alpha,t)
                yL = yL+ delta
                error = g(m,yL,i,t_r,L,alpha,t)
            end
            return alpha*yL^m    
            
        end
        
    end
    
    # The Ref's here are to ensure it works on CPU and GPU compatible array backends (q can be a GPU array)
    @test sqrt_rmse_over_max_q = sqrt(mean((analytic.(time_data, alpha, t_c, t_r, i, L, m) .- q).^4.0))/ maximum(q) < 3e-3
end
@testset " Integrating River and Soil at same time I" begin
    FT = Float64;
    
    ClimateMachine.init(; disable_gpu = true);
    
    soil_heat_model = PrescribedTemperatureModel();

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.4,
        Ksat = 6.94e-5 / 60,
        S_s = 5e-4,
    );
    no_precip = (t) -> eltype(t)(0)
    function he(z)
        z_interface = -1.0
        ν = 0.4
        S_s = 5e-4
        α = 100
        n = 2.0
        m = 0.5
        if z < z_interface
            return -S_s * (z - z_interface) + ν
        else
            return ν * (1 + (α * (z - z_interface))^n)^(-m)
        end
    end
    ϑ_l0 = (aux) -> eltype(aux)(he(aux.z))

    m_river = RiverModel(
        (x,y) -> eltype(x)(-0.0016),
        (x,y) -> eltype(x)(0.0),
        (x,y) -> eltype(x)(1);
        mannings = (x,y) -> eltype(x)(0.025)
    )

    
    N_poly = 1;
    xres = FT(2.286)
    yres = FT(0.25)
    zres = FT(0.1)
    # Specify the domain boundaries.
    zmax = FT(0);
    zmin = FT(-1.0);
    xmax = FT(182.88)
    ymax = FT(1.0)
    topo_max = FT(0.0016*xmax)
    Δz = FT(zres/2)

    bc =  LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
        surface_bc = LandComponentBC(soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                                                   precip_model = DrivenConstantPrecip{FT}(no_precip),
                                                                                   runoff_model = CoarseGridRunoff{FT}(Δz)
                                                                                       )
                                     ),
        miny_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  ),
        minx_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  river = Dirichlet((aux, t) -> eltype(aux)(0))
                                  ),
        maxx_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  ),
        maxy_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  )
    )
    
    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = vanGenuchten{FT}(n = 2.0,  α = 100.0),
        initialϑ_l = ϑ_l0,
    );
    
    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);
    
    function init_land_model!(land, state, aux, localgeo, time)
        state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
        state.river.area = eltype(state)(0)
        
    end


    river_precip_of_t = (x,y,t) -> eltype(t)(t < (30*60) ? 1.4e-5 : 0)
    sources = (Precip{FT}(river_precip_of_t),)
    model = LandModel(
        param_set,
        m_soil;
        river = m_river,
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_land_model!,
    );
    
    driver_config = ClimateMachine.MultiColumnLandModel(
        "LandModel",
        (N_poly, N_poly),
        (xres,yres,zres),
        xmax,
        ymax,
        zmax,
        param_set,
        model;
        zmin = zmin,
        meshwarp = (x...) -> warp_constant_slope(x...;
        topo_max = topo_max, zmin = zmin, xmax = xmax),
    );

    t0 = FT(0)
    timeend = FT(60*60)
    dt = FT(10)
    n_outputs = 50;
    every_x_simulation_time = ceil(Int, timeend / n_outputs);
    solver_config =
        ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
    
    
    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary;
    area_index =
        varsindex(vars_state(model, Prognostic(), FT), :river, :area)
    moisture_index = varsindex(vars_state(model, Prognostic(), FT), :soil, :water, :ϑ_l)
    dons = Dict([k => Dict() for k in 1:n_outputs]...)

    iostep = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        area = Q[:, area_index, :]
        ϑ_l = Q[:,moisture_index,:]
        all_vars = Dict{String, Array}(
            "t" => [t],
            "area" => area,
            "ϑ_l" => ϑ_l,
        )
        dons[iostep[1]] = all_vars
        iostep[1] += 1
        return
    end

    filter_vars = ("river.area",)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            filter_vars,
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    ClimateMachine.invoke!(solver_config; user_callbacks = (cbtmarfilter, callback,));
    x = aux[:,1,:]
    y = aux[:,2,:]
    z = aux[:,3,:]
    ztrue = inverse_constant_slope.(x,y,z;topo_max = topo_max, zmin = zmin, xmax = xmax)
    mask = ((FT.(x .== 182.88) .+ FT.(ztrue .== 0.0)) .== 2)

    N = sum([length(dons[k]) ==3 for k in 1:n_outputs])
    # get prognostic variable area from nodal state (m^2)
    height = [mean(Array(dons[k]["area"])[mask[:]]) for k in 1:N]
    
    alpha = sqrt(0.0016)/(0.025)
    i = 1.4e-5
    L = xmax
    m = 5/3
    t_c = (L*i^(1-m)/alpha)^(1/m)
    t_r = 30*60
    q = height.^(m) .* alpha 
    function g(m,y, i, t_r, L, alpha, t)
        output = L/alpha-y^(m)/i-y^(m-1)*m*(t-t_r)
        return output
    end
    
    function dg(m,y, i, t_r, L, alpha, t)
        output = -y^(m-1)*m/i-y^(m-2)*m*(m-1)*(t-t_r)
        return output
    end
    function analytic(t,alpha, t_c, t_r, i, L, m)
        if t < t_c
            return alpha*(i*t)^(m)
        end
        
        if t <= t_r && t > t_c
            return alpha*(i*t_c)^(m)
        end
        
        if t > t_r
            yL = (i*(t-t_r))
            delta = 1
            error = g(m,yL,i,t_r,L,alpha,t)
            while abs(error) > 1e-4
                delta = -g(m,yL,i,t_r,L,alpha,t)/dg(m,yL,i,t_r,L,alpha,t)
                yL = yL+ delta
                error = g(m,yL,i,t_r,L,alpha,t)
            end
            return alpha*yL^m    
            
        end
        
    end

    time_data = [dons[l]["t"][1] for l in 1:N]
    
    solution = analytic.(time_data, alpha, t_c, t_r, i, L, m)
    #test that river didnt affect soil
    @test sum(he.(z[:]) .- dons[N]["ϑ_l"][:]) < eps(FT)
    # test that soil didnt affect river
    @test sqrt_rmse_over_max_q = sqrt(mean((analytic.(time_data, alpha, t_c, t_r, i, L, m) .- q).^4.0))/ maximum(q) < 3e-3

end

@testset " Integrating River and Soil at same time II" begin
    FT = Float64;
    
    ClimateMachine.init(; disable_gpu = true);
    
    soil_heat_model = PrescribedTemperatureModel();

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.4,
        Ksat = 6.94e-5 / 60,
        S_s = 5e-4,
    );
    no_precip = (t) -> eltype(t)(0)
    function he(z)
        z_interface = -1.0
        ν = 0.4
        S_s = 5e-4
        α = 100
        n = 2.0
        m = 0.5
        if z < z_interface
            return -S_s * (z - z_interface) + ν
        else
            return ν * (1 + (α * (z - z_interface))^n)^(-m)
        end
    end
    ϑ_l0 = (aux) -> eltype(aux)(he(aux.z))

    m_river = RiverModel(
        (x,y) -> eltype(x)(-0.0016),
        (x,y) -> eltype(x)(0.0),
        (x,y) -> eltype(x)(1);
        mannings = (x,y) -> eltype(x)(0.025)
    )

    
    N_poly = 1;
    xres = FT(2.286)
    yres = FT(0.25)
    zres = FT(0.1)
    # Specify the domain boundaries.
    zmax = FT(0);
    zmin = FT(-1.0);
    xmax = FT(182.88)
    ymax = FT(1.0)
    topo_max = FT(0.0016*xmax)
    Δz = FT(zres/2)

    bc =  LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
        surface_bc = LandComponentBC(soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                                                   precip_model = DrivenConstantPrecip{FT}(no_precip),
                                                                                   runoff_model = CoarseGridRunoff{FT}(Δz)
                                                                                       )
                                     ),
        miny_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  ),
        minx_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  river = Dirichlet((aux, t) -> eltype(aux)(0))
                                  ),
        maxx_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  ),
        maxy_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  )
    )
    
    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = vanGenuchten{FT}(n = 2.0,  α = 100.0),
        initialϑ_l = ϑ_l0,
    );
    
    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);
    
    function init_land_model!(land, state, aux, localgeo, time)
        state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
        state.river.area = eltype(state)(0)
        
    end


    river_precip_of_t = (x,y,t) -> eltype(t)(0)
    sources = (SoilRunoff{FT}(),)
    model = LandModel(
        param_set,
        m_soil;
        river = m_river,
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_land_model!,
    );
    
    driver_config = ClimateMachine.MultiColumnLandModel(
        "LandModel",
        (N_poly, N_poly),
        (xres,yres,zres),
        xmax,
        ymax,
        zmax,
        param_set,
        model;
        zmin = zmin,
        meshwarp = (x...) -> warp_constant_slope(x...;
        topo_max = topo_max, zmin = zmin, xmax = xmax),
    );

    t0 = FT(0)
    timeend = FT(60*60)
    dt = FT(10)
    n_outputs = 50;
    every_x_simulation_time = ceil(Int, timeend / n_outputs);
    solver_config =
        ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
    
    
    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary;
    area_index =
        varsindex(vars_state(model, Prognostic(), FT), :river, :area)
    moisture_index = varsindex(vars_state(model, Prognostic(), FT), :soil, :water, :ϑ_l)
    dons = Dict([k => Dict() for k in 1:n_outputs]...)

    iostep = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        area = Q[:, area_index, :]
        ϑ_l = Q[:,moisture_index,:]
        all_vars = Dict{String, Array}(
            "t" => [t],
            "area" => area,
            "ϑ_l" => ϑ_l,
        )
        dons[iostep[1]] = all_vars
        iostep[1] += 1
        return
    end

    filter_vars = ("river.area",)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            filter_vars,
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    ClimateMachine.invoke!(solver_config; user_callbacks = (cbtmarfilter, callback,));
    x = aux[:,1,:]
    y = aux[:,2,:]
    z = aux[:,3,:]
    ztrue = inverse_constant_slope.(x,y,z;topo_max = topo_max, zmin = zmin, xmax = xmax)
    mask = ((FT.(x .== 182.88) .+ FT.(ztrue .== 0.0)) .== 2)

    N = sum([length(dons[k]) ==3 for k in 1:n_outputs])
    # get prognostic variable area from nodal state (m^2)
    height = [mean(Array(dons[k]["area"])[mask[:]]) for k in 1:N]

    
    solution = FT(0.0)
    #test that river didnt affect soil
    @test sum(he.(z[:]) .- dons[N]["ϑ_l"][:]) < eps(FT)
    # test that soil didnt affect river
    @test sqrt_rmse = sqrt(mean((solution .- height).^2.0)) < eps(FT)

end


@testset " Integrating River and Soil at same time II - FV" begin
    FT = Float64;
    
    ClimateMachine.init(; disable_gpu = true);
    
    soil_heat_model = PrescribedTemperatureModel();

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.4,
        Ksat = 6.94e-5 / 60,
        S_s = 5e-4,
    );
    no_precip = (t) -> eltype(t)(0)
    function he(z)
        z_interface = -1.0
        ν = 0.4
        S_s = 5e-4
        α = 100
        n = 2.0
        m = 0.5
        if z < z_interface
            return -S_s * (z - z_interface) + ν
        else
            return ν * (1 + (α * (z - z_interface))^n)^(-m)
        end
    end
    ϑ_l0 = (aux) -> eltype(aux)(he(aux.z))

    m_river = RiverModel(
        (x,y) -> eltype(x)(-0.0016),
        (x,y) -> eltype(x)(0.0),
        (x,y) -> eltype(x)(1);
        mannings = (x,y) -> eltype(x)(0.025)
    )

    
    N_poly = 1;
    xres = FT(2.286)
    yres = FT(0.25)
    zres = FT(0.1)
    # Specify the domain boundaries.
    zmax = FT(0);
    zmin = FT(-1.0);
    xmax = FT(182.88)
    ymax = FT(1.0)
    topo_max = FT(0.0016*xmax)
    Δz = FT(zres/2)

    bc =  LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
        surface_bc = LandComponentBC(soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                                                   precip_model = DrivenConstantPrecip{FT}(no_precip),
                                                                                   runoff_model = CoarseGridRunoff{FT}(Δz)
                                                                                       )
                                     ),
        miny_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  ),
        minx_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  river = Dirichlet((aux, t) -> eltype(aux)(0))
                                  ),
        maxx_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  ),
        maxy_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  )
    )
    
    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = vanGenuchten{FT}(n = 2.0,  α = 100.0),
        initialϑ_l = ϑ_l0,
    );
    
    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);
    
    function init_land_model!(land, state, aux, localgeo, time)
        state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
        state.river.area = eltype(state)(0)
        
    end


    river_precip_of_t = (x,y,t) -> eltype(t)(0)
    sources = (SoilRunoff{FT}(),)
    model = LandModel(
        param_set,
        m_soil;
        river = m_river,
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_land_model!,
    );
    
    driver_config = ClimateMachine.MultiColumnLandModel(
        "LandModel",
        (N_poly, 0),
        (xres,yres,zres),
        xmax,
        ymax,
        zmax,
        param_set,
        model;
        zmin = zmin,
        meshwarp = (x...) -> warp_constant_slope(x...;
                                                 topo_max = topo_max, zmin = zmin, xmax = xmax),
        fv_reconstruction = FVLinear(),

    );

    t0 = FT(0)
    timeend = FT(60*60)
    dt = FT(10)
    n_outputs = 50;
    every_x_simulation_time = ceil(Int, timeend / n_outputs);
    solver_config =
        ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
    
    
    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary;
    area_index =
        varsindex(vars_state(model, Prognostic(), FT), :river, :area)
    moisture_index = varsindex(vars_state(model, Prognostic(), FT), :soil, :water, :ϑ_l)
    dons = Dict([k => Dict() for k in 1:n_outputs]...)

    iostep = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        area = Q[:, area_index, :]
        ϑ_l = Q[:,moisture_index,:]
        all_vars = Dict{String, Array}(
            "t" => [t],
            "area" => area,
            "ϑ_l" => ϑ_l,
        )
        dons[iostep[1]] = all_vars
        iostep[1] += 1
        return
    end

    filter_vars = ("river.area",)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            filter_vars,
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    ClimateMachine.invoke!(solver_config; user_callbacks = (cbtmarfilter, callback,));
    x = aux[:,1,:]
    y = aux[:,2,:]
    z = aux[:,3,:]
    ztrue = inverse_constant_slope.(x,y,z;topo_max = topo_max, zmin = zmin, xmax = xmax)
    mask = ((FT.(x .== 182.88) .+ FT.(abs.(ztrue.+Δz) .< eps(FT))) .== 2)

    N = sum([length(dons[k]) ==3 for k in 1:n_outputs])
    # get prognostic variable area from nodal state (m^2)
    height = [mean(Array(dons[k]["area"])[mask[:]]) for k in 1:N]

    
    solution = FT(0.0)
    #test that river didnt affect soil
    @test sum(he.(z[:]) .- dons[N]["ϑ_l"][:]) < eps(FT)
    # test that soil didnt affect river
    @test sqrt_rmse = sqrt(mean((solution .- height).^2.0)) < eps(FT)

end


@testset " Integrating River and Soil III - Maxwell slope" begin
    # quite slow, unclear if we need this - tests overland analytic solution in different regime
    FT = Float64;
    
    ClimateMachine.init(; disable_gpu = true);
    
    soil_heat_model = PrescribedTemperatureModel();

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.4,
        Ksat = 6.94e-5 / 60,
        S_s = 5e-4,
    );
    no_precip = (t) -> eltype(t)(0)
    function he(z)
        z_interface = -1.0
        ν = 0.4
        S_s = 5e-4
        α = 100
        n = 2.0
        m = 0.5
        if z < z_interface
            return -S_s * (z - z_interface) + ν
        else
            return ν * (1 + (α * (z - z_interface))^n)^(-m)
        end
    end
    ϑ_l0 = (aux) -> eltype(aux)(he(aux.z))

    m_river = RiverModel(
        (x,y) -> eltype(x)(-0.0005),
        (x,y) -> eltype(x)(0.0),
        (x,y) -> eltype(x)(1);
        mannings = (x,y) -> eltype(x)(3.31e-3*60)
    )

    
    N_poly = 1;
    xres = FT(5)
    yres = FT(80)
    zres = FT(0.2)
    # Specify the domain boundaries.
    zmax = FT(0);
    zmin = FT(-5);
    xmax = FT(400)
    ymax = FT(320)
    Δz = FT(zres/2)

    bc =  LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
        surface_bc = LandComponentBC(soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                                                   precip_model = DrivenConstantPrecip{FT}(no_precip),
                                                                                   runoff_model = CoarseGridRunoff{FT}(Δz)
                                                                                       )
                                     ),
        miny_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  ),
        minx_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  river = Dirichlet((aux, t) -> eltype(aux)(0))
                                  ),
        maxx_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  ),
        maxy_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                  )
    )
    
    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = vanGenuchten{FT}(n = 2.0,  α = 100.0),
        initialϑ_l = ϑ_l0,
    );
    
    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);
    
    function init_land_model!(land, state, aux, localgeo, time)
        state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
        state.river.area = eltype(state)(0)
        
    end

    river_precip_of_t = (x,y,t) -> eltype(t)(t < (200*60) ? 5.5e-6 : 0)
    sources = (Precip{FT}(river_precip_of_t),)
    model = LandModel(
        param_set,
        m_soil;
        river = m_river,
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_land_model!,
    );
    
    topo_max = FT(0.2)
    # Create the driver configuration.
    driver_config = ClimateMachine.MultiColumnLandModel(
        "LandModel",
        (N_poly, N_poly),
        (xres,yres,zres),
        xmax,
        ymax,
        zmax,
        param_set,
        model;
        zmin = zmin,
        numerical_flux_first_order = RusanovNumericalFlux(),
        meshwarp = (x...) -> warp_constant_slope(x...;topo_max = topo_max, zmin = zmin, xmax = xmax),
    );
    
    
    # Choose the initial and final times, as well as a timestep.
    t0 = FT(0)
    timeend = FT(60* 300)
    dt = FT(10);
    n_outputs = 50;
    every_x_simulation_time = ceil(Int, timeend / n_outputs);
    solver_config =
        ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
    
    
    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary;
    area_index =
        varsindex(vars_state(model, Prognostic(), FT), :river, :area)
    moisture_index = varsindex(vars_state(model, Prognostic(), FT), :soil, :water, :ϑ_l)
    
    
    dons = Dict([k => Dict() for k in 1:n_outputs]...)
    iostep = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        area = Q[:, area_index, :]
        ϑ_l = Q[:, moisture_index, :]
        
        all_vars = Dict{String, Array}(
            "t" => [t],
            "area" => area,
            "ϑ_l" => ϑ_l,
        )
        dons[iostep[1]] = all_vars
        iostep[1] += 1
        return
    end
    filter_vars = ("river.area",)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            filter_vars,
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end
    
    ClimateMachine.invoke!(solver_config; user_callbacks = (cbtmarfilter, callback,));
    x = aux[:,1,:]
    y = aux[:,2,:]
    z = aux[:,3,:]
    ztrue = inverse_constant_slope.(x,y,z;topo_max = topo_max, zmin = zmin, xmax = xmax)

    mask = ((FT.(x .== 400.0)) + FT.(ztrue .== 0.0)) .==2
    n_outputs = sum([length(dons[k]) ==3 for k in 1:n_outputs])

    
    # get prognostic variable area from nodal state (m^2)
    height = [mean(Array(dons[k]["area"])[mask[:]]) for k in 1:n_outputs]
    time_data = [dons[l]["t"][1] for l in 1:n_outputs]
    alpha = sqrt(0.0005)/(3.31e-3*60)
    i = 5.5e-6
    L = xmax
    m = 5/3
    t_c = (L*i^(1-m)/alpha)^(1/m)
    t_r = 200*60
    q = height.^(m) .* alpha 
    function g(m,y, i, t_r, L, alpha, t)
        output = L/alpha-y^(m)/i-y^(m-1)*m*(t-t_r)
        return output
    end
    
    function dg(m,y, i, t_r, L, alpha, t)
        output = -y^(m-1)*m/i-y^(m-2)*m*(m-1)*(t-t_r)
        return output
    end
    
    function analytic2(t,alpha, t_c, t_r, i, L, m)
        yLr = i*t_r
        t_c2 = L/alpha/yLr^(m-1)
        t_p = t_r + (t_c2-t_r)/m
        if t < t_r
            return alpha*(i*t)^(m)
        end
        
        if t <= t_p && t > t_r
            return alpha*(i*t_r)^(m)
        end
        
        if t > t_p
            yL = (i*(t_r))
            delta = 1
            error = g(m,yL,i,t_r,L,alpha,t)
            while abs(error) > 1e-10
                delta = -g(m,yL,i,t_r,L,alpha,t)/dg(m,yL,i,t_r,L,alpha,t)
                yL = yL+ delta
                error = g(m,yL,i,t_r,L,alpha,t)
            end
            return alpha*yL^m    
            
        end
        
    end
    
    solution = analytic2.(time_data, alpha, t_c, t_r, i, L, m)
    #test that river didnt affect soil
    @test sum(he.(z[:]) .- dons[n_outputs]["ϑ_l"][:]) < eps(FT)
    # test that soil didnt affect river
    @test mean(abs.(solution .- q)) .< 1e-6 # 
end




#Maxwell - V - include as integration test? if so need data.
#=

function warp_tilted_v(xin, yin, zin)
    FT = eltype(xin)
    slope_sides = FT(0.05)
    slope_v = (0.02)
    zbase = slope_v*(yin-FT(1000))
    zleft = FT(0.0)
    zright = FT(0.0)
    if xin < FT(800)
        zleft = slope_sides*(xin-FT(800))
    end
    if xin > FT(820)
        zright = slope_sides*(FT(1)+(xin-FT(820))/FT(800))
    end
    zout = zbase+zleft+zright
    x, y, z = xin, yin, zout
    return x, y, z
end

#@testset "V Catchment Maxwell River Model" begin
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
        return x >= MFT(800) && x <= MFT(820) ? MFT(2.5*60 * 10^-3) : MFT(2.5*60 * 10^-4)
    end
    
    m_river = RiverModel(
        x_slope,
        y_slope,
        (x,y) -> eltype(x)(1);
        mannings = channel_mannings,
    )
    
    bc = LandDomainBC(
        miny_bc = LandComponentBC(river = Dirichlet((aux, t) -> eltype(aux)(0))),
        minx_bc = LandComponentBC(river = Dirichlet((aux, t) -> eltype(aux)(0))),
        maxx_bc = LandComponentBC(river = Dirichlet((aux, t) -> eltype(aux)(0))),
    )
    
    function init_land_model!(land, state, aux, localgeo, time)
        state.river.area = eltype(state)(0)
    end
    
    # units in m / s 
    precip(x, y, t) = t < (90 * 60) ? 3e-6 : 0.0
    
    sources = (Precip{FT}(precip),)
    
    m = LandModel(
        param_set,
        m_soil;
        river = m_river,
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_land_model!,
    )
    
    N_poly_hori = 1;
    N_poly_vert = 1;
    xres = FT(20)
    yres = FT(20)
    zres = FT(1)
    # Specify the domain boundaries.
    zmax = FT(1);
    zmin = FT(0);
    xmax = FT(1620)
    ymax = FT(1000)
    
    
    driver_config = ClimateMachine.MultiColumnLandModel(
        "LandModel",
        (N_poly_hori, N_poly_vert),
        (xres,yres,zres),
        xmax,
        ymax,
        zmax,
        param_set,
        m;
        zmin = zmin,
        numerical_flux_first_order = RusanovNumericalFlux()
        # meshwarp = (x...) -> warp_tilted_v(x...),
    );
    
    t0 = FT(0)
    timeend = FT(180*60)
    dt = FT(0.5)
    
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    mygrid = solver_config.dg.grid
    Q = solver_config.Q
    
    area_index =
        varsindex(vars_state(m, Prognostic(), FT), :river, :area)
    n_outputs = 60
    
    every_x_simulation_time = ceil(Int, timeend / n_outputs)
    
    dons = Dict([k => Dict() for k in 1:n_outputs]...)
    
    iostep = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        area = Q[:, area_index, :]
        all_vars = Dict{String, Array}(
            "t" => [t],
            "area" => area,
        )
        dons[iostep[1]] = all_vars
        iostep[1] += 1
        return
    end
    
    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))
    
    # Compare flowrate analytical derivation
    aux = solver_config.dg.state_auxiliary;
    x = aux[:,1,:]
    y = aux[:,2,:]
    z = aux[:,3,:]
    # get all nodal points at the max X bound of the domain
    mask2 = (Float64.(aux[:,2,:] .== 1000.0)) .==1# .+ Float64.(aux[:,1,:] .<= 820) .+ Float64.(aux[:,1,:] .> 800)) .== 3
     n_outputs = length(dons)
    function compute_Q(a,xv)
        height = max.(a,0.0) ./ 1.0# width = 1 here, everywhere
        v = calculate_velocity(m_river,xv, 1000.0, height)# put in y = 1000.0
        speed = sqrt(v[1]^2.0+v[2]^2.0 +v[3]^2.0)
        Q_outlet = speed .* height  .* 60.0 # multiply by 60 so it is per minute, not per second
        return Q_outlet
    end
    # We divide by 4 because we have 4 nodal points with the same value at each x (z = 0, 1)
    # Multiply by xres because the solution at each point roughly represents that the solution for that range in x
    # Maybe there is a conceptual error here?
    Q = [sum(compute_Q.(Array(dons[k]["area"])[:][mask2[:]], x[mask2[:]])) for k in 1:n_outputs] ./4.0 .* xres



    # get similation timesteps (s)
    time_data = [dons[l]["t"][1] for l in 1:n_outputs]
    #also helpful
    #scatter(aux[:,1,:][:], aux[:,2,:][:], dons[50]["area"][:])
#end
=#
