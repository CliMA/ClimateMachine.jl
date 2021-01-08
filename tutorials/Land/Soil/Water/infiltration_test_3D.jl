using Plots
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
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
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state

const FT = Float64;

# - Initialize ClimateMachine for CPU
ClimateMachine.init(; disable_gpu = true);


soil_heat_model = PrescribedTemperatureModel();
soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.4,
    Ksat = 6.94e-6 / 60,
    S_s = 5e-4,
);


sigmoid(x, offset, width) = typeof(x)(exp((x-offset)/width)/(1+exp((x-offset)/width)))
precip_of_t = (t) -> eltype(t)(((-6.94e-6)/120)*(1-sigmoid(t,1250*60.0,20.0)) -((3.3e-4)/60) * (1-sigmoid(t, 1100*60.0,30.0)))
ϑ_l0 = (aux) -> eltype(aux)(0.399- 0.1 * sigmoid(aux.z, -1.0,0.02))#heaviside((-0.5)-aux.z))

# Specify the polynomial order and resolution.
N_poly = 1;
xres = FT(80)
yres = FT(80)
zres = FT(0.02) ## could change to be larger to match Maxwell, but this is used in calculating i_c
# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-3);## will change to 5?
xmax = FT(400)
ymax = FT(320)

bc =  LandDomainBC(
    bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
    surface_bc = LandComponentBC(soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                precip_model = DrivenConstantPrecip{FT}(precip_of_t),
                                                runoff_model = CoarseGridRunoff{FT}(zres)
                                                                                   )),
    lateral_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0)))
)


soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = 2.0,  α = 100.0),
    initialϑ_l = ϑ_l0
);

# Create the soil model - the coupled soil water and soil heat models.
m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);

# We are ignoring sources and sinks here, like runoff or freezing and thawing.
sources = ();

# Define the function that initializes the prognostic variables. This
# in turn calls the functions supplied to `soil_water_model`.
function init_soil_water!(land, state, aux, localgeo, time)
    state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
end


# Create the land model - in this tutorial, it only includes the soil.
m = LandModel(
    param_set,
    m_soil;
    boundary_conditions = bc,
    source = sources,
    init_state_prognostic = init_soil_water!,
);

# # Specify the numerical configuration and output data.
function warp_maxwell_slope(xin, yin, zin; topo_max = 0.2, zmin =  -3, xmax = 400)
    FT = eltype(xin)
    zmax = FT(xin/xmax*topo_max)
    alpha = FT(1.0)- zmax/zmin
    zout = zmin+ (zin-zmin)*alpha
    x, y, z = xin, yin, zout
    return x, y, z
end


function inverse_warp_maxwell_slope(xin, yin, zin; topo_max = 0.2, zmin =  -3, xmax = 400)
    FT = eltype(xin)
    zmax = FT(xin/xmax*topo_max)
    alpha = FT(1.0)- zmax/zmin
    zout = (zin-zmin)/alpha+zmin
    return zout
 end

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
    #numerical_flux_first_order = CentralNumericalFluxFirstOrder(),now the default for us
    meshwarp = (x...) -> warp_maxwell_slope(x...;topo_max = topo_max, zmin = zmin, xmax = xmax),
);


# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60*700)
dt = FT(2); #5

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# Determine how often you want output.
const n_outputs = 500;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);
mygrid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;
grads = solver_config.dg.state_gradient_flux

x_ind = varsindex(vars_state(m, Auxiliary(), FT), :x)
y_ind = varsindex(vars_state(m, Auxiliary(), FT), :y)
z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
K∇h_vert_ind = varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[3]
K∇h_y_ind = varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[2]
K∇h_x_ind = varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[1]

x = aux[:, x_ind, :]
y = aux[:, y_ind, :]
z = aux[:, z_ind, :]
ϑ_l = Q[:, ϑ_l_ind, :]
K∇h_vert = zeros(length(ϑ_l)) .+ FT(NaN)

all_data = [Dict{String, Array}("ϑ_l" => ϑ_l, "Khz" => K∇h_vert,"Khx" => K∇h_vert,"Khy" => K∇h_vert)]
time_data = FT[0] # store time data

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    ϑ_l = Q[:, ϑ_l_ind, :]
    K∇h_vert = grads[:, K∇h_vert_ind, :]
    Khy = grads[:, K∇h_y_ind, :]
    Khx = grads[:, K∇h_x_ind, :]

    dons = Dict{String, Array}("ϑ_l" => ϑ_l, "Khz" => K∇h_vert,"Khx" => Khx,"Khy" => Khy)
    push!(all_data, dons)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));



#function compute_at_surface(array, x, z)
#    fluxes = []
#    for i in unique(x)
#        q = [x .== i][1]
#        loc = round.(z[q] .* 10) .== maximum(round.(z[q].*10))
#        value = mean(array[q][loc])
#        push!(fluxes, value)
#    end
#    return fluxes
#end

N = length(all_data)

#i_c_of_t = [mean(compute_at_surface(all_data[k]["Khz"][:],x[:],z[:])) for k in 1:N]
precip = precip_of_t.(time_data[1:N])

ztrue = inverse_warp_maxwell_slope.(x,y,z; topo_max = 0.2, zmin = -3, xmax = 400)
surface_locs = round.(ztrue[:] .* 100) .== 0
####Get actual BC
function compute_bc(all_data, N,x,y,z, surface_locs)
    aux_structure = vars_state(m, Auxiliary(), FT)
    st_structure = vars_state(m, Prognostic(), FT)
    x1 = x[:][surface_locs]
    y1 = y[:][surface_locs]
    z1 = z[:][surface_locs]
    i_arr = zeros(N, length(z1))
    d_arr = zeros(N, length(z1))
    h_arr = zeros(N, length(z1))
    r_arr = zeros(N, length(z1))
    moisture = zeros(N, length(z1))
    for i in 1:N
        K∇h  = all_data[i]["Khz"][:][surface_locs]
        ϑ1 = all_data[i]["ϑ_l"][:][surface_locs]
        for k in 1:length(z1)
            
            aux_vals = [x1[k], y1[k], z1[k], FT(0.0), FT(0.0)]# h and k arent used in the functions we call.
            aux_m = MArray{Tuple{varsize(aux_structure)}, FT}(aux_vals)
            theaux = Vars{aux_structure}(aux_m)
            
            
            st_vals = [ϑ1[k], FT(0.0)]# ice is always zero
            st_m = MArray{Tuple{varsize(st_structure)}, FT}(st_vals)
            thest = Vars{st_structure}(st_m)
            t = time_data[i]
            infiltration = compute_surface_flux(m_soil,bc.surface_bc.soil_water.runoff_model,bc.surface_bc.soil_water.precip_model, theaux, thest, t)
            d = compute_dunne_runoff(m_soil,bc.surface_bc.soil_water.runoff_model,bc.surface_bc.soil_water.precip_model, theaux, thest, t)
            h = compute_horton_runoff(m_soil,bc.surface_bc.soil_water.runoff_model,bc.surface_bc.soil_water.precip_model, theaux, thest, t)
            r = compute_generalized_runoff(m_soil,bc.surface_bc.soil_water.runoff_model,bc.surface_bc.soil_water.precip_model, theaux, thest, t)
            h_arr[i,k] = h
            d_arr[i,k] = d
            r_arr[i,k] = r
            i_arr[i,k] = infiltration
            moisture[i,k] =  ϑ1[k]
            
        end
    end
    return x1,y1,z1, i_arr, moisture,d_arr,h_arr, r_arr
end
xv,yv,zv, iv,mmv, dv,hv, rv = compute_bc(all_data, N, x,y,z,surface_locs)

#iiii = [mean(iv[k,:]) for k in 1:N]
#plot(time_data ./ 60, -i_c_of_t, label  ="mean, measured from simulation")
#plot!(time_data ./ 60, iiii, label  = "mean flux BC", color = "black")
#plot!(xlabel = "Time (minutes)")
#plot!(ylabel = "Flux at surface (m/s)")
#plot!(legend = :bottomright)
#savefig("./tutorials/Land/Soil/Water/runoff_plots/mean_horton_infiltration_dunne_ic.png")

#top_moisture = [mean(mmv[k,:]) for k in 1:N]
#effective_s  = volumetric_liquid_fraction.(top_moisture,Ref(0.4)) ./ 0.4
#ψ = pressure_head.(Ref(soil_water_model.hydraulics), Ref(0.4), Ref(soil_param_functions.S_s), top_moisture, Ref(0.0))
#expected = soil_param_functions.Ksat .* (1 .- ψ/zres)
#iiii = [mean(iv[k,:]) for k in 1:N]
#plot(time_data ./ 60, log10.(abs.(i_c_of_t)), label  ="mean, measured from simulation")
#plot!(time_data ./ 60, log10.(expected), label  = "i_c", color = "red")
#plot!(time_data ./ 60, log10.(-precip), label  = "precip", color = "purple")
#plot!(time_data ./ 60, log10.(abs.(iiii)), label  = "mean flux BC", color = "black")
#plot!(time_data ./ 60,zeros(length(time_data)) .+ log10(soil_param_functions.Ksat), label = "Ksat")
#plot!(xlabel = "Time (minutes)")
#plot!(ylabel = "Flux at surface (m/s)")
#plot!(legend = :bottomleft)
#plot!(title = "Dunne ic")
#plot!(ylim = [-10,-5])
#savefig("./tutorials/Land/Soil/Water/runoff_plots/mean_horton_infiltration_dunne_ic_log.png")

#function f(k)
#    scatter(z[:], x[:],all_data[k]["ϑ_l"][:], zlim = [0.35,0.402], xlim = [-1, 0.2],label = "")
#end
#anim = @animate for i in 1:Int(50)
#    f(i*3)
#end
#(gif(anim, "./tutorials/Land/Soil/Water/horton_surface_moisture2_no_ic.gif", fps = 8))

#would be much better as a contour plot!!

function f2(k)
    locs = z .> -1
    θvals = all_data[k]["ϑ_l"][locs]
    xvals = x[locs]
    zvals  = z[locs]
    
    myvals =unique([[round(100*xvals[k])/100,round(100*zvals[k])/100] for k in 1:length(xvals)])
    θ2 = zeros(length(myvals))
    θ1 = zeros(length(myvals))
    X = [myvals[k][1] for k in 1:length(myvals)]
    Z = [myvals[k][2] for k in 1:length(myvals)]
    for i in 1:length(myvals)
        pair = myvals[i]
        if (pair[1] .* 100)./100 == 0.0
            loc = (Int.(round.(zvals .* 100)./100 .== pair[2]) .+ Int.(round.(xvals .* 100)./100 .== pair[1])) .== 2
            θ1[i] = mean(θvals[loc])
        elseif (pair[1] .* 100)./100 == 400.0
            loc = (Int.(round.(zvals .* 100)./100 .== pair[2]) .+ Int.(round.(xvals .* 100)./100 .== pair[1])) .== 2
            θ2[i] = mean(θvals[loc])
        end
        
    end
    plot(θ1[θ1.!=0]./0.4,Z[θ1 .!=0],xlim = [0.3,0.402]./0.4, label = "downslope", xlabel = "S_l", ylabel = "Depth")
    plot!(θ2[θ2.!=0]./0.4,Z[θ2 .!=0], xlim = [0.3,0.402]./0.4, label = "upslope")
    plot!(legend = :bottomleft)

end
anim = @animate for i in 1:Int(50)
    f2(i*10)
end
(gif(anim, "./tutorials/Land/Soil/Water/runoff_plots/horton_surface_moisture_dunne_ic.gif", fps = 8))



#####
function f3(k)
    locs = z .> -1
    θvals = all_data[k]["ϑ_l"][locs]
    xvals = x[locs]
    zvals  = z[locs]
    
    myvals =unique([[round(100*xvals[k])/100,round(100*zvals[k])/100] for k in 1:length(xvals)])
    θ2 = zeros(length(myvals))
    θ1 = zeros(length(myvals))
    X = [myvals[k][1] for k in 1:length(myvals)]
    Z = [myvals[k][2] for k in 1:length(myvals)]
    for i in 1:length(myvals)
        pair = myvals[i]
        if (pair[1] .* 100)./100 == 0.0
            loc = (Int.(round.(zvals .* 100)./100 .== pair[2]) .+ Int.(round.(xvals .* 100)./100 .== pair[1])) .== 2
            θ1[i] = mean(θvals[loc])
        elseif (pair[1] .* 100)./100 == 400.0
            loc = (Int.(round.(zvals .* 100)./100 .== pair[2]) .+ Int.(round.(xvals .* 100)./100 .== pair[1])) .== 2
            θ2[i] = mean(θvals[loc])
        end
        
    end
    return [mean(θ1[θ1 .!=0]), mean(θ2[θ2 .!=0])]
end
#Plot top_moisture and integrated vertical moisture vs time at upslope and downslope point
ds = [round.(xv) .== 0.0][1]
top_moisture_ds = [mean(mmv[k,ds[:]]) for k in 1:N]
us = [round.(xv) .== 400.0][1]
top_moisture_us = [mean(mmv[k,us][:]) for k in 1:N]
integrated = [f3(k) for k in 1:N]
int_ds = [integrated[k][1] for k in 1:N]
int_us = [integrated[k][2] for k in 1:N]
plot(time_data ./ 60, top_moisture_ds, label = "Top ϑ_l, downslope")
plot!(time_data./ 60, top_moisture_us, label = "Top ϑ_l, upslope")
plot!(time_data ./60, int_us, label = "Mean ϑ_l above 1m, upslope")
plot!(time_data ./ 60, int_us, label = "Mean ϑ_l above 1m, downslope")
plot!(xlabel = "Time (min)")
plot!(ylabel = "ϑ_l")
plot!(legend = :bottomleft)
savefig("./tutorials/Land/Soil/Water/runoff_plots/horton_surface_moisture_dunne_ic_time.png")
#Plot ic, D, H, flux, precip, R vs time at two spots on surface




bc_ds = [mean(iv[k,ds[:]]) for k in 1:N]
d_ds = [mean(dv[k,ds[:]]) for k in 1:N]
h_ds = [mean(hv[k,ds[:]]) for k in 1:N]
r_ds = [mean(rv[k,ds[:]]) for k in 1:N]
ψ = pressure_head.(Ref(soil_water_model.hydraulics), Ref(0.4), Ref(soil_param_functions.S_s), top_moisture_ds, Ref(0.0))

expected_ds_ic = soil_param_functions.Ksat .* (1 .- ψ/zres)

plot(time_data ./60, log10.(-bc_ds), label = "infiltration BC", linewidth = 3)
plot!(time_data ./60, log10.(r_ds), label = "Runoff", linewidth = 2)
#plot!(time_data ./60, log10.(d_ds), label = "Dunne Runoff", linewidth = 2)
plot!(time_data ./60, log10.(expected_ds_ic), label = "infiltration capacity", color = "black")
plot!(time_data ./60, log10.(-precip), label = "precip")
plot!(time_data ./60, zeros(N) .+ log10.(soil_param_functions.Ksat), label = "Ksat")
plot!(ylim=[-9,-5])
plot!(xlabel = "Time (min)")
plot!(ylabel = "-K∂_z h")
plot!(legend = :bottomright)
plot!(title = "Surface fluxes; downslope")
savefig("./tutorials/Land/Soil/Water/runoff_plots/horton_surface_flux_dunne_ic_time.png")

