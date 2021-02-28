#add TMAR to this for river.area
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.River
using ClimateMachine.Land.Runoff
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

const FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

soil_heat_model = PrescribedTemperatureModel();

soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.4,
    Ksat = 6.94e-4 / 60,
    S_s = 5e-4,
);
heaviside(x) = 0.5 * (sign(x) + 1)
precip_of_t = (t) -> eltype(t)(-((3.3e-4)/60)*heaviside(200*60-t))
function he(z)
    z_interface = -0.5
    ν = 0.4
    S_s = 5e-4
    α = 100.0
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
    (x,y) -> eltype(x)(-0.05),
    (x,y) -> eltype(x)(0.0),
    (x,y) -> eltype(x)(1);
    mannings = (x,y) -> eltype(x)(3.31e-3*60)
)


N_poly = 1;
xres = FT(80)
yres = FT(80)
zres = FT(0.2)
# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-5);
xmax = FT(400)
ymax = FT(320)

bc =  LandDomainBC(
    bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
    surface_bc = LandComponentBC(soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                precip_model = DrivenConstantPrecip{FT}(precip_of_t),
                                                runoff_model = CoarseGridRunoff{FT}(zres)
                                                                                   )),
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

# Create the soil model - the coupled soil water and soil heat models.
m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);
# Define the function that initializes the prognostic variables. This
# in turn calls the functions supplied to `soil_water_model`.
function init_land_model!(land, state, aux, localgeo, time)
    state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
    state.river.area = eltype(state)(0)
end
# # Define a warping function to build an analytic topography (we want a 2D slope, in 3D):

function warp_maxwell_slope(xin, yin, zin; topo_max = 0.2, zmin =  -5, xmax = 400)
    FT = eltype(xin)
    zmax = FT((xmax-xin)/xmax*topo_max)
    alpha = FT(1.0)- zmax/zmin
    zout = zmin+ (zin-zmin)*alpha
    x, y, z = xin, yin, FT(zout)
    return x, y, z
end

#=
precip(x, y, t) = t < (200 * 60) ? 5.5e-6 : 0.0

sources = (Precip{FT}(precip),)

# Create the land model - in this tutorial, it only includes the soil.
m = LandModel(
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
    m;
    zmin = zmin,
    numerical_flux_first_order = RusanovNumericalFlux(),
    meshwarp = (x...) -> warp_maxwell_slope(x...;topo_max = topo_max, zmin = zmin, xmax = xmax),
);


# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60* 300)
dt = FT(1);

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);


# Determine how often you want output.
const n_outputs = 500;
const every_x_simulation_time = ceil(Int, timeend / n_outputs);
mygrid = solver_config.dg.grid
Q = solver_config.Q

area_index =
    varsindex(vars_state(m, Prognostic(), FT), :river, :area)
moisture_index = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
dons = Dict([k => Dict() for k in 1:n_outputs]...)

iostep = [1]
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
) do (init = false)
    t = ODESolvers.gettime(solver_config.solver)
    area = Q[:, area_index, :]
    moisture = Q[:, moisture_index, :]
    all_vars = Dict{String, Array}(
        "t" => [t],
        "area" => area,
        "ϑ_l" => moisture,
    )
    dons[iostep[1]] = all_vars
    iostep[1] += 1
    return
end


ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));
aux = solver_config.dg.state_auxiliary;
x = aux[:,1,:]
y = aux[:,2,:]
z = aux[:,3,:]
mask = ((FT.(x .== 400.0)) + FT.(z .== 0.0)) .==2
n_outputs = length(dons)
# get prognostic variable area from nodal state (m^2)
area = [mean(Array(dons[k]["area"])[mask[:]]) for k in 1:n_outputs]
height = area
# get similation timesteps (s)

time_data = [dons[l]["t"][1] for l in 1:n_outputs]

alpha = sqrt(0.05)/(3.31e-3*60)
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

=#

sources = (SoilRunoff{FT}(),)

# Create the land model - in this tutorial, it only includes the soil.
m = LandModel(
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
    m;
    zmin = zmin,
    numerical_flux_first_order = RusanovNumericalFlux(),
    meshwarp = (x...) -> warp_maxwell_slope(x...;topo_max = topo_max, zmin = zmin, xmax = xmax),
);


# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60* 300)
dt = FT(0.3);

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);


# Determine how often you want output.
const n_outputs = 500;
const every_x_simulation_time = ceil(Int, timeend / n_outputs);
mygrid = solver_config.dg.grid
Q = solver_config.Q

area_index =
    varsindex(vars_state(m, Prognostic(), FT), :river, :area)
moisture_index = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
dons = Dict([k => Dict() for k in 1:n_outputs]...)

iostep = [1]
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
) do (init = false)
    t = ODESolvers.gettime(solver_config.solver)
    area = Q[:, area_index, :]
    moisture = Q[:, moisture_index, :]
    all_vars = Dict{String, Array}(
        "t" => [t],
        "area" => area,
        "ϑ_l" => moisture,
    )
    dons[iostep[1]] = all_vars
    iostep[1] += 1
    return
end


ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));
aux = solver_config.dg.state_auxiliary;
x = aux[:,1,:]
y = aux[:,2,:]
z = aux[:,3,:]
mask = ((FT.(x .== 400.0)) + FT.(z .== 0.0)) .==2
n_outputs = length(dons)
# get prognostic variable area from nodal state (m^2)
area = [mean(Array(dons[k]["area"])[mask[:]]) for k in 1:n_outputs]
height = area
# get similation timesteps (s)

time_data = [dons[l]["t"][1] for l in 1:n_outputs]

alpha = sqrt(0.05)/(3.31e-3*60)
i = 5.5e-6
L = xmax
m = 5/3
t_c = (L*i^(1-m)/alpha)^(1/m)
t_r = 200*60

q = height.^(m) .* alpha .* 320.0
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
# need to multiply by ymax to get integrated value
# also convert to be per minute
# does paper plot only 1/4 of domain?
plot(time_data ./ 60, analytic.(time_data, alpha, t_c, t_r, i, L, m) .* 60 .* 320.0 ./4)
plot!(time_data ./ 60, q .* 60 ./4)
plot!(ylim = [0,12])
plot!(yticks = [0,2,4,6,8,10,12])
plot!(xticks = [0,50,100,150,200,250,300])
