# add tmar in for river
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
using ClimateMachine.Filters
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

function warp_constant_slope(xin, yin, zin; topo_max = 0.2, zmin = -5, xmax = 400)
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


function warp_shift_up(xin, yin, zin; topo_max = 0.2, xmax = 400)
    FT = eltype(xin)
    zmax = FT((FT(1.0)-xin / xmax) * topo_max)
    zout = zin + zmax
    x, y, z = xin, yin, zout
    return x, y, z
end
    
function inverse_shift_up(xin, yin, zin; topo_max = 0.2, xmax = 400)
    FT = eltype(xin)
    zmax = FT((FT(1.0)-xin / xmax) * topo_max)
    zout = zin - zmax
    return zout
end

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

soil_heat_model = PrescribedTemperatureModel();

soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.4,
    Ksat = 6.94e-5 / 60,
    S_s = 5e-4,
);
heaviside(x) = 0.5 * (sign(x) + 1)
precip_of_t = (t) -> eltype(t)(-((3.3e-4)/60)*heaviside(200*60-t))
function he(z)
    z_interface = -1.0
    ν = 0.4
    S_s = 5e-4
    α = 1.0
    n = 2.0
    m = 0.5
    θ_r = 0.08
    if z < z_interface
        return -S_s * (z - z_interface) + ν
    else
        return (ν-θ_r) * (1 + (α * (z - z_interface))^n)^(-m)+θ_r
    end
end
ϑ_l0 = (aux) -> eltype(aux)(he(aux.z))

m_river = RiverModel(
    (x,y) -> eltype(x)(-0.0005),
    (x,y) -> eltype(x)(0.0),
    (x,y) -> eltype(x)(1);
    mannings = (x,y) -> eltype(x)(3.31e-4*60)
)

N_poly = 1
xres = FT(80)
yres = FT(1)
zres = FT(0.2)
# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-2);
xmax = FT(400)
ymax = FT(1)
Δz = FT(zres/2)

bc =  LandDomainBC(
bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
surface_bc = LandComponentBC(#soil_water = Dirichlet((aux,t)->eltype(aux)(0.4)),),
                             soil_water = CATHYWaterBoundaryConditions(FT;
                                                                               precip_model = DrivenConstantPrecip{FT}(precip_of_t),
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
    hydraulics = vanGenuchten{FT}(n = 2.0,  α = 1.0),
    initialϑ_l = ϑ_l0,
);

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);

function init_land_model!(land, state, aux, localgeo, time)
    state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
    state.river.area = eltype(state)(0)
    
end
#river_precip_of_t = (x,y,t) -> eltype(t)(0)
#sources = (Precip{FT}(river_precip_of_t),)
sources = (SoilRunoff{FT}(),)


model = LandModel(
    param_set,
    m_soil;
    river = m_river,
    boundary_conditions = bc,
    source = sources,
    init_state_prognostic = init_land_model!,
);

topo_max = FT(0.2)
driver_config = ClimateMachine.MultiColumnLandModel(
    "LandModel",
    N_poly,
    (xres,yres,zres),
    xmax,
    ymax,
    zmax,
    param_set,
    model;
    zmin = zmin,
    # uncomment this to turn off Rusanov
    numerical_flux_first_order = RusanovNumericalFlux(),
    meshwarp = (x...) -> warp_shift_up(x...;topo_max = topo_max,xmax = xmax),
);


# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60* 300)
dt = FT(1);
const n_outputs = 500;
const every_x_simulation_time = ceil(Int, timeend / n_outputs);
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);


mygrid = solver_config.dg.grid
Q = solver_config.Q
grad = solver_config.dg.state_gradient_flux;
aux = solver_config.dg.state_auxiliary;
area_index =
    varsindex(vars_state(model, Prognostic(), FT), :river, :area)
moisture_index = varsindex(vars_state(model, Prognostic(), FT), :soil, :water, :ϑ_l)
K∇h_index = varsindex(vars_state(model,GradientFlux(), FT), :soil, :water)


dons = Dict([k => Dict() for k in 1:n_outputs]...)
#precip_model = bc.surface_bc.soil_water.precip_model
#runoff_model = bc.surface_bc.soil_water.runoff_model
water = m_soil.water
Δz = Δz#runoff_model.Δz
hydraulics = water.hydraulics
ν = soil_param_functions.porosity
specific_storage = soil_param_functions.S_s
T = 0.0
θ_i = 0.0
ϑ_bc = ν

function get_bc1(inc, ic)
    if inc < -ic
        #ponding BC
        K∇h⁺ = min(ic,-inc)
    else
        K∇h⁺ = - inc
    end
    return K∇h⁺
end



iostep = [1]
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
) do (init = false)
    t = ODESolvers.gettime(solver_config.solver)
    area = Q[:, area_index, :]
    ϑ_below = Q[:, moisture_index, :]
    incident_water_flux = precip_of_t(t)

    ∂h∂z =
        FT(1) .+
        (
            pressure_head.(Ref(hydraulics), Ref(ν), Ref(specific_storage), ϑ_bc, Ref(θ_i)) .-
            pressure_head.(Ref(hydraulics), Ref(ν), Ref(specific_storage), ϑ_below, Ref(θ_i))
        ) ./ Δz
    K =
        soil_param_functions.Ksat .* hydraulic_conductivity.(
    Ref(water.impedance_factor),
    Ref(water.viscosity_factor),
    Ref(water.moisture_factor),
    Ref(hydraulics),
    Ref(θ_i),
    Ref(soil_param_functions.porosity),
    Ref(T),
    Ref(ϑ_bc / ν),
        )
    i_c = K .*  ∂h∂z
    bcvval1 = get_bc1.(Ref(incident_water_flux),i_c)
    
    all_vars = Dict{String, Array}(
        "t" => [t],
        "area" => area,
        "ϑ_l" => ϑ_below,
        "flux" => -bcvval1,
        "i_c" => i_c,
        "grad" => grad[:,K∇h_index,:][:,3,:],
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
#remove cbtmarfilter from list to stop TMAR
ClimateMachine.invoke!(solver_config; user_callbacks = (cbtmarfilter, callback,));

x = aux[:,1,:]
y = aux[:,2,:]
z = aux[:,3,:]
ztrue = inverse_shift_up.(x,y,z;topo_max = topo_max, xmax = xmax)
mask = ((FT.(abs.(x .-400.0) .< 1e-10)) + FT.(abs.(ztrue) .< 1e-10)) .==2
N = sum([length(dons[k]) !=0 for k in 1:n_outputs])
# get prognostic variable area from nodal state (m^2)
area = [mean(Array(dons[k]["area"])[mask[:]]) for k in 1:N]
water = [mean(Array(dons[k]["ϑ_l"])[mask[:]]) for k in 1:N]
flux =  [mean(Array(dons[k]["flux"])[mask[:]]) for k in 1:N]
i_c =  [mean(Array(dons[k]["i_c"])[mask[:]]) for k in 1:N]
gradf =  [mean(Array(dons[k]["grad"])[mask[:]]) for k in 1:N]
height = area
time_data = [dons[l]["t"][1] for l in 1:N]
#plot(time_data ./60, -precip_of_t.(time_data).+flux, label = "with infiltration")
#plot!(time_data ./60, -precip_of_t.(time_data), label = "without infiltration")

alpha = sqrt(0.0005)/(3.31e-4*60)
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


solution = analytic.(time_data, alpha, t_c, t_r, i, L, m)

#plot(time_data ./ 60,solution, label = "base case")
#plot!(time_data ./ 60, q , label = "sim")
#plot!(time_data ./ 60, q , label = "K = 1e-1m/d,Δ = 0.05m", color = "red", linestyle = :dash)
#plot!(ylim = [0,10.5])
#plot!(yticks = [0,1,2,3,4,5,6,7,8,9,10])
#plot!(xticks = [0,50,100,150,200,250,300])

#q_001_01, q_001_02

