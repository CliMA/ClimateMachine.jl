# # Zettl - site SV60
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using DelimitedFiles
using Plots
using Dates

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
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state
import ClimateMachine.DGMethods.FVReconstructions: FVLinear

const FT = Float64;

ClimateMachine.init(; disable_gpu = true);

#data = readdlm("/Users/katherinedeck/Desktop/data/huang_2011/Site 2D-60 Data/Enviroscan Recalb.-Table 1.csv",',')
data = readdlm("/Users/katherinedeck/Desktop/code/ClimateMachine.jl/tutorials/Land/Soil/SiteBenchmarks/data/zettl_60_inf.csv", '\t')
#tws_obs = FT.(data[46:46+16,23])
#columns = 6:1:16
#depth1 = [5,15,25,35,45,55,65,75,85,95,105]
#depth2 = [2.5,12.5,22.5,32.5,42.5,52.5,62.5,72.5,82.5,92.5,102.5]
#depth = -0.5 .* (depth1 .+ depth2) ./100
#T = FT.(data[46:75,4])
#data = transpose(FT.(data[46:75,:][:,columns])./100)
#ic = data[:, T .== 0.0]
depth = -1/100*data[2:end,1]
ic = data[2:end, 2]

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

# # Set up the soil model

soil_heat_model = PrescribedTemperatureModel();


function ks(z::F) where {F}
    factor = F(1/100/60)# given in cm/min
    if z >= F(-0.07)
        k = F(0.944)
    elseif z >= F(-0.1)
        k = F(0.71)
    elseif z >=  F(-0.17)
        k = F(0.703)
    elseif z>= F(-0.2)
        k = F(0.624)
    elseif z>=F(-0.23)
        k = F(0.521)
    elseif z>=F(-0.30)
        k = F(0.341)
    elseif z >=  F(-0.35)
        k = F(0.351)
    elseif z>= F(-0.45)
        k = F(0.456)
    elseif z>=F(-0.48)
        k = F(0.423)
    elseif z>=F(-0.53)
        k = F(0.260)
    elseif z >=  F(-0.59)
        k = F(0.382)
    elseif z>= F(-0.68)
        k = F(0.270)
    elseif z>=F(-0.74)
        k = F(0.187)
    elseif z>=F(-0.78)
        k = F(0.297)
    elseif z >=  F(-0.84)
        k = F(0.272)
    elseif z >=  F(-0.97)
        k = F(0.725)
    elseif z >=  F(-1.02)
        k = F(1.169)
    else
        k = F(0.457)
    end
    return k*factor
end

function vgα(z::F) where {F}
    factor = F(100)
   if z >= F(-0.07)
        k = F(0.07)
    elseif z >= F(-0.1)
        k = F(0.057)
    elseif z >=  F(-0.17)
        k = F(0.102)
    elseif z>= F(-0.2)
        k = F(0.093)
    elseif z>=F(-0.23)
        k = F(0.068)
    elseif z>=F(-0.30)
        k = F(0.034)
    elseif z >=  F(-0.35)
        k = F(0.058)
    elseif z>= F(-0.45)
        k = F(0.057)
    elseif z>=F(-0.48)
        k = F(0.057)
    elseif z>=F(-0.53)
        k = F(0.056)
    elseif z >=  F(-0.59)
        k = F(0.06)
    elseif z>= F(-0.68)
        k = F(0.049)
    elseif z>=F(-0.74)
        k = F(0.027)
    elseif z>=F(-0.78)
        k = F(0.132)
    elseif z >=  F(-0.84)
        k = F(0.083)
    elseif z >=  F(-0.97)
        k = F(0.092)
    elseif z >=  F(-1.02)
        k = F(0.046)
    else
        k = F(0.064)
    end
    return k*factor*FT(2)# they report αᵈ; we don't model hysteresis. use mean: αʷ = 2αᵈ, so use *1.5? in sims they use 2 for all non-hysteresis runs
end

function vgn(z::F) where {F}
   if z >= F(-0.07)
        k = F(1.717)
    elseif z >= F(-0.1)
        k = F(1.968)
    elseif z >=  F(-0.17)
        k = F(2.517)
    elseif z>= F(-0.2)
        k = F(2.736)
    elseif z>=F(-0.23)
        k = F(2.388)
    elseif z>=F(-0.30)
        k = F(2.302)
    elseif z >=  F(-0.35)
        k = F(2.136)
    elseif z>= F(-0.45)
        k = F(2.169)
    elseif z>=F(-0.48)
        k = F(2.095)
    elseif z>=F(-0.53)
        k = F(2.396)
    elseif z >=  F(-0.59)
        k = F(2.568)
    elseif z>= F(-0.68)
        k = F(2.455)
    elseif z>=F(-0.74)
        k = F(2.711)
    elseif z>=F(-0.78)
        k = F(1.488)
    elseif z >=  F(-0.84)
        k = F(2.109)
    elseif z >=  F(-0.97)
        k = F(2.206)
    elseif z >=  F(-1.02)
        k = F(2.362)
    else
        k = F(2.306)
    end
    return k
end


function θr(z::F) where {F}
    if z >= F(-0.07)
        k = F(0.0)
    elseif z >= F(-0.17)
        k = F(0.007)
    elseif z >=  F(-0.2)
        k = F(0.011)
    elseif z >=F(-0.3)
        k = F(0.0)
    elseif z >= F(-0.35)
        k = F(0.001)
    elseif z>=F(-0.84)
        k = F(0.0)
    elseif z>= F(-0.97)
        k = F(0.013)
    else
        k = F(0.00)
    end
    return k
end


function ν(z::F) where {F}
    if z >= F(-0.07)
        k = F(0.483)
    elseif z >= F(-0.1)
        k = F(0.508)
    elseif z >=  F(-0.17)
        k = F(0.536)
    elseif z>= F(-0.2)
        k = F(0.504)
    elseif z>=F(-0.23)
        k = F(0.452)
    elseif z>=F(-0.30)
        k = F(0.401)
    elseif z >=  F(-0.35)
        k = F(0.355)
    elseif z>= F(-0.45)
        k = F(0.356)
    elseif z>=F(-0.48)
        k = F(0.360)
    elseif z>=F(-0.53)
        k = F(0.356)
    elseif z >=  F(-0.59)
        k = F(0.352)
    elseif z>= F(-0.68)
        k = F(0.405)
    elseif z>=F(-0.74)
        k = F(0.385)
    elseif z>=F(-0.78)
        k = F(0.378)
    elseif z >=  F(-0.84)
        k = F(0.264)
    elseif z >=  F(-0.97)
        k = F(0.303)
    elseif z >=  F(-1.02)
        k = F(0.287)
    else
        k = F(0.34)
    end
    return k
end

N_poly = 1;
nelem_vert = 55;

# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-1.1);
Δ = FT((zmax-zmin)/nelem_vert/2)

S_s = 1e-3
wpf = WaterParamFunctions(FT; Ksat = (aux)->ks(aux.z), S_s = S_s, θ_r = (aux)->θr(aux.z))
soil_param_functions = SoilParamFunctions(FT; porosity = (aux)->ν(aux.z), water = wpf)
kstop = ks(FT(0.0))

#surface_state = (aux, t) -> eltype(aux)(S_s*0.1 + ν(aux.z))
surface_flux = (aux,t)-> eltype(aux)(t < 64*60 ? -kstop*((0.1+Δ)-aux.soil.water.h)/Δ : 0.0)
bottom_flux = (aux, t) -> aux.soil.water.K * eltype(aux)(-1)

bc = LandDomainBC(
    bottom_bc = LandComponentBC(
        soil_water = Neumann(bottom_flux)
    ),
    surface_bc = LandComponentBC(
       # soil_water = Dirichlet(surface_state)
    soil_water = Neumann(surface_flux)
    )
)
#icdata = readdlm("./tutorials/Land/Soil/SiteBenchmarks/data/huang_sv60_ic.csv",',')
θ = Spline1D(reverse(depth),reverse(ic),k =1)
ϑ_l0 = aux -> eltype(aux)(θ(aux.z))


soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten(FT; n = (aux) ->vgn(aux.z), α = (aux)->vgα(aux.z)),
    initialϑ_l = ϑ_l0,
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

# Create the driver configuration.
driver_config = ClimateMachine.SingleStackConfiguration(
    "LandModel",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m;
    zmin = zmin,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
    #fv_reconstruction = FVLinear()
);

# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60 *64)+t0
dt = FT(0.005);

# Create the solver configuration.
#ode_solver_type = ClimateMachine.ExplicitSolverType(
#    solver_method = SSPRK34SpiteriRuuth
#)
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt)#, ode_solver_type = ode_solver_type,)

dg = solver_config.dg
Q = solver_config.Q

ode_solver = SSPRK34SpiteriRuuth(
        dg,
        Q;
        dt = dt,
        t0 = t0,
)
solver_config.solver = ode_solver
#solver_config.solver = SSPRK34SpiteriRuuth()
#=
    dg = solver_config.dg
    Q = solver_config.Q

    vdg = DGModel(
        driver_config;
        state_auxiliary = dg.state_auxiliary,
        direction = VerticalDirection(),
    )



    linearsolver = BatchedGeneralizedMinimalResidual(
        dg,
        Q;
        max_subspace_size = 30,
        atol = -1.0,
        rtol = 1e-5,
    )

    """
    N(q)(Q) = Qhat  => F(Q) = N(q)(Q) - Qhat
    F(Q) == 0
    ||F(Q^i) || / ||F(Q^0) || < tol
    """
    nonlinearsolver =
        JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-5)

    ode_solver = ARK2GiraldoKellyConstantinescu(
        dg,
        vdg,
        NonLinearBackwardEulerSolver(
            nonlinearsolver;
            isadjustable = true,
            preconditioner_update_freq = 100,
        ),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = false,
        variant = NaiveVariant(),
    )

    solver_config.solver = ode_solver
=#
# Determine how often you want output.
n_outputs = 64;

every_x_simulation_time = ceil(Int, (timeend-t0) / n_outputs);

# Create a place to store this output.
state_types = (Prognostic(), Auxiliary(), GradientFlux())
dons_arr = Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
time_data = FT[0] # store time data

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

# # Run the integration
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));
dons = dict_of_nodal_states(solver_config, state_types; interp = true)
push!(dons_arr, dons)
push!(time_data, gettime(solver_config.solver))
z = get_z(solver_config.dg.grid; rm_dupes = true);

their_sim = readdlm("/Users/katherinedeck/Desktop/code/ClimateMachine.jl/tutorials/Land/Soil/SiteBenchmarks/data/zettl_60_sim.csv", '\t')


plot(dons_arr[1]["soil.water.ϑ_l"],z, label = "initial", color = "grey", aspect_ratio = 0.8)

plot!(dons_arr[9]["soil.water.ϑ_l"], z, label = "8min", color = "orange")
scatter!(data[2:end, 3], depth,label = "", color = "orange")
plot!(their_sim[2:end, 3],-their_sim[2:end,1]./100, label = "", linestyle = :dot, color = "orange")

plot!(dons_arr[17]["soil.water.ϑ_l"], z, label = "16min", color = "red")
scatter!( data[2:end, 4],depth, label = "", color = "red")
plot!(their_sim[2:end, 4],-their_sim[2:end,1]./100, label = "", linestyle = :dot, color = "red")

plot!(dons_arr[25]["soil.water.ϑ_l"], z, label = "24min", color = "teal")
scatter!( data[2:end, 5],depth, label = "", color = "teal")
plot!(their_sim[2:end, 5],-their_sim[2:end,1]./100, label = "", linestyle = :dot, color = "teal")

plot!(dons_arr[33]["soil.water.ϑ_l"], z, label = "32min", color = "blue")
scatter!( data[2:end, 6],depth, label = "", color = "blue")
plot!(their_sim[2:end, 6],-their_sim[2:end,1]./100, label = "", linestyle = :dot, color = "blue")

plot!(dons_arr[41]["soil.water.ϑ_l"], z, label = "40min", color = "purple")
scatter!( data[2:end, 7],depth, label = "", color = "purple")
plot!(their_sim[2:end, 7],-their_sim[2:end,1]./100, label = "", linestyle = :dot, color = "purple")

plot!(dons_arr[65]["soil.water.ϑ_l"], z, label = "64min", color = "green")
scatter!( data[2:end, 8], depth,label = "", color = "green")
plot!(their_sim[2:end, 8],-their_sim[2:end,1]./100, label = "", linestyle = :dot, color = "green")
plot!([0,0],[0,0], label = "CliMA", color = "black")
plot!([0,0],[0,0], label = "Hydrus", color = "black", linestyle = :dot)
scatter!([1,1],[1,1], label = "Data", color = "black")
plot!(legend = :bottomright)

plot!(xlim = [0,0.5])

plot!(xlim = [0,0.6])

plot!(ylim = [-1.1,0], yticks = [-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1])

plot!(ylabel = "Depth (m)")

plot!(xlabel  = "Volumeteric Water Content")

savefig("./sv60_alpha_2_inf_updated_data.png")
function compute_tws(N)
    m = data[2:end, N]
    foo = Spline1D(reverse(depth), reverse(m))
    q = foo.(z)
    return sum((q[2:end] .+ q[1:end-1])./2.0 .*Δ .*2)*100
end
tws = [sum((dons_arr[k]["soil.water.ϑ_l"][1:end-1] .+ dons_arr[k]["soil.water.ϑ_l"][2:end]) ./2) .* Δ .* 2 .* 100 for k in 1:65] #cm
plot(0:1:64, tws, label = "Simulated")
scatter!([0, 8,16,24,32,40,64], compute_tws.(Array(1:1:7).+1), label = "Observed")
#scatter!(T[1:17],tws_obs,  label = "Observed")
plot!(legend = :bottomright)
plot!(ylabel = "Total Water Storage in 1.1m (cm)")
plot!(xlabel = "Minutes since infiltration began")
savefig("./sv60_alpha_2_tws_inf_updated_data.png")

#=
#data = readdlm("./tutorials/Land/Soil/SiteBenchmarks/data/huang_sv60.csv", ',')
plot(dons_arr[1]["soil.water.ϑ_l"],z, label = "initial", color = "black", aspect_ratio = 0.8)
plot!(dons_arr[9]["soil.water.ϑ_l"], z, label = "8min", color = "orange")
scatter!(data[:, T .== 8], depth,label = "", color = "orange")

#scatter!(data[2:end, 1], data[2:end, 2], label = "", color = "orange")
plot!(dons_arr[17]["soil.water.ϑ_l"], z, label = "16min", color = "red")
scatter!( data[:, T .== 16],depth, label = "", color = "red")
#scatter!(data[2:end, 9], data[2:end, 10], label = "", color = "red")
plot!(dons_arr[25]["soil.water.ϑ_l"], z, label = "24min", color = "teal")
scatter!( data[:, T .== 24],depth, label = "", color = "teal")
#scatter!(data[2:end, 3], data[2:end, 4], label = "", color = "teal")
plot!(dons_arr[33]["soil.water.ϑ_l"], z, label = "32min", color = "blue")
scatter!( data[:, T .== 32],depth, label = "", color = "blue")
#scatter!(data[2:end, 11], data[2:end, 12], label = "", color = "blue")
plot!(dons_arr[41]["soil.water.ϑ_l"], z, label = "40min", color = "purple")
scatter!( data[:, T .== 40],depth, label = "", color = "purple")
#scatter!(data[2:end, 5], data[2:end, 6], label = "", color = "purple")
plot!(dons_arr[65]["soil.water.ϑ_l"], z, label = "64min", color = "green")
scatter!( data[:, T .== 64], depth,label = "", color = "green")
#scatter!(data[2:end, 7], data[2:end, 8], label = "", color = "green")
plot!(legend = :bottomright)

plot!(xlim = [0,0.5])

plot!(xlim = [0,0.6])

plot!(ylim = [-1.1,0], yticks = [-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1])

plot!(ylabel = "Depth (m)")

plot!(xlabel  = "Volumeteric Water Content")


savefig("./sv60_alpha_2_infiltration.png")

function compute_tws(t)
    m = data[:, T .== t]
    foo = Spline1D(reverse(depth), reverse(m[:]))
    q = foo.(z)
    return sum((q[2:end] .+ q[1:end-1])./2.0 .*Δ .*2)*100
end
tws = [sum((dons_arr[k]["soil.water.ϑ_l"][1:end-1] .+ dons_arr[k]["soil.water.ϑ_l"][2:end]) ./2) .* Δ .* 2 .* 100 for k in 1:65] #cm
plot(0:1:64, tws, label = "Simulated")
scatter!(T[1:17], compute_tws.(T[1:17]), label = "Observed")
#scatter!(T[1:17],tws_obs,  label = "Observed")
plot!(legend = :bottomright)
plot!(ylabel = "Total Water Storage in 1.1m (cm)")
plot!(xlabel = "Minutes since infiltration began")
savefig("./sv60_alpha_2_tws_inf.png")

=#
