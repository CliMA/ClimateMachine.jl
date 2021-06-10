# # Lamont
# to do - compare STAMP to swats, get soil characeteristics for the soil here. make sure IC are correct.
# # Preliminary setup

using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using DelimitedFiles
using Plots
using Dates
using NCDatasets
using JLD2, FileIO
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

const FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

# # Set up the soil model

soil_heat_model = PrescribedTemperatureModel();

# Meso - v2
#=
function ks(z::F) where {F}
    if z >= F(-0.075)
        k = F(1.9e-7)
    elseif z >= F(-0.175)
        k = F(1.6e-7)
    elseif z >=  F(-0.35)
        k = F(1.1e-7)
    elseif z>= F(-0.525)
        k = F(2.4e-7)
    elseif z>=F(-0.675)
        k = F(1.6e-7)
    else
        k = F(0.8e-7)
    end
    
    return k
end

function vgα(z::F) where {F}
    if z >= F(-0.075)
        k = F(0.71)
    elseif z >= F(-0.175)
        k = F(0.79)
    elseif z >=  F(-0.35)
        k = F(0.68)
    elseif z>= F(-0.525)
        k = F(1.0)
    elseif z>=F(-0.675)
        k = F(0.80)
    else
        k = F(0.46)
    end
    
    return k

end
function vgn(z::F) where {F}
    if z >= F(-0.075)
        k = F(1.37)
    elseif z >= F(-0.175)
        k = F(1.33)
    elseif z >=  F(-0.35)
        k = F(1.26)
    elseif z>= F(-0.525)
        k = F(1.16)
    elseif z>=F(-0.675)
        k = F(1.15)
    else
        k = F(1.15)
    end
    
    return k

end

function ν(z::F) where {F}
    if z >= F(-0.075)
        k = F(0.406)
    elseif z >= F(-0.175)
        k = F(0.412)
    elseif z >=  F(-0.35)
        k = F(0.446)
    elseif z>= F(-0.525)
        k = F(0.419)
    elseif z>=F(-0.675)
        k = F(0.379)
    else
        k = F(0.358)
    end
    
    return k

end


function θr(z::F) where {F}
    if z >= F(-0.075)
        k = F(0.078)
    elseif z >= F(-0.175)
        k = F(0.082)
    elseif z >=  F(-0.35)
        k = F(0.114)
    elseif z>= F(-0.525)
        k = F(0.13)
    elseif z>=F(-0.675)
        k = F(0.13)
    else
        k = F(0.121)
    end
    
    return k

end
=#


# Meso - v13
function ks(z::F) where {F}
    if z >= F(-0.15)
        k = F(3.5e-7)
    elseif z >=  F(-0.35)
        k = F(2.5e-7)
    elseif z>= F(-0.525)
        k = F(5.1e-7)
    elseif z>=F(-0.675)
        k = F(3.7e-7)
    else
        k = F(1.8e-7)
    end
    
    return k
end


function vgL(z::F) where {F}
    if z >= F(-0.15)
        k = F(-0.14)
    elseif z >=  F(-0.35)
        k = F(-0.34)
    elseif z>= F(-0.525)
        k = F(-1.67)
    elseif z>=F(-0.675)
        k = F(-3.29)
    else
        k = F(-5.1)
    end
    
    return k
end

function vgα(z::F) where {F}
    if z >= F(-0.15)
        k = F(0.9)
    elseif z >=  F(-0.35)
        k = F(0.85)
    elseif z>= F(-0.525)
        k = F(1.78)
    elseif z>=F(-0.675)
        k = F(1.44)
    else
        k = F(0.7)
    end
    
    return k

end
function vgn(z::F) where {F}
    if z >= F(-0.15)
        k = F(1.39)
    elseif z >=  F(-0.35)
        k = F(1.3)
    elseif z>= F(-0.525)
        k = F(1.19)
    elseif z>=F(-0.675)
        k = F(1.16)
    else
        k = F(1.14)
    end
    
    return k

end

function ν(z::F) where {F}
    if z >= F(-0.15)
        k = F(0.41)
    elseif z >=  F(-0.35)
        k = F(0.46)
    elseif z>= F(-0.525)
        k = F(0.44)
    elseif z>=F(-0.675)
        k = F(0.40)
    else
        k = F(0.41)
    end
    
    return k

end


function θr(z::F) where {F}
    if z >= F(-0.15)
        k = F(0.057)
    elseif z >=  F(-0.35)
        k = F(0.076)
    elseif z>= F(-0.525)
        k = F(0.075)
    elseif z>=F(-0.675)
        k = F(0.069)
    else
        k = F(0.061)
    end
    
    return k

end

#=
function mymap(z::F,  depths::Array{F,1},values::Array{F,1}) where {F}
    N = length(depths)
    v = F(0)
    @inbounds if -z < depths[1]
        @inbounds v =  values[1]
    elseif -z >= @inbounds depths[N]
        @inbounds v =  values[N]
    else
        for i in 2:1:N
            @inbounds   if -z < depths[i] && -z>= depths[i-1]
                @inbounds num = values[i-1]*(depths[i]+z)+values[i]*(-z-depths[i-1])
                @inbounds denom = depths[i]-depths[i-1]
                v =  num/denom
            end
        end
    end
    return v
    
end



function mymap_sorted(z::F,  depths::Array{F,1},values::Array{F,1}) where {F}
    N = length(depths)
    v = F(0)
    @inbounds if -z < depths[1]
        @inbounds v =  values[1]
    elseif -z >= @inbounds depths[N]
        @inbounds v =  values[N]
    else
        i1 = searchsorted(soil_depths, -z).stop
        i2 = searchsorted(soil_depths, -z).start
        if i1 != i2
            @inbounds num = values[i1]*(depths[i2]+z)+values[i2]*(-z-depths[i1])
            @inbounds denom = depths[i2]-depths[i1]
            v =  num/denom
        else
            @inbounds v = values[i1]
        end
    end
    return v
end
=#


wpf = WaterParamFunctions(FT; Ksat = (aux)->ks(aux.z), S_s = 1e-4, θ_r = (aux)->θr(aux.z))
soil_param_functions = SoilParamFunctions(FT; porosity = (aux)->ν(aux.z), water = wpf)

### Read in flux data
cutoff = DateTime(2016,03,20,0,30,0)
cutoff1 = DateTime(2016,03,29,0,30,0)
cutoff2 = DateTime(2016,06,01,0,30,0)
filepath = "data/lamont/arms_flux/sgparmbeatmC1.c1.20160101.003000.nc"
ds = Dataset(filepath)
times = ds["time"][:]
p = ((times .<=cutoff2) .+ (times .>= cutoff)) .== 2
precip_rate = ds["precip_rate_sfc"][p] # mm/hr
lhf_baebbr = ds["latent_heat_flux_baebbr"][p]
times = ds["time"][p]
close(ds)
keep1 = (typeof.(lhf_baebbr) .!= Missing)
lhf = lhf_baebbr[keep1]
keep2 = (typeof.(precip_rate) .!= Missing)
precip_rate = precip_rate[keep2] ./1000 ./ 3600
Lv = 2.5008e6
ρ = 1e3
evap_rate = lhf ./ Lv ./ρ

foo = (times .-times[times .== cutoff1])./1000
seconds = [k.value for k in foo]

# Create interpolating function for evap_rate. same for P. then we have a net water flux
E = Spline1D(seconds[keep1], evap_rate)
P  = Spline1D(seconds[keep2], precip_rate)
function net_water_flux(t::Real, P::Spline1D, E::Spline1D)
    net = -P(t) +E(t)
    return net
end
incident = (t) -> net_water_flux(t, P, E)


bottom_flux = (aux, t) -> aux.soil.water.K * eltype(aux)(-1)
surface_flux = (aux, t) -> incident(t)
surface_zero_flux = (aux, t) -> eltype(aux)(0)
N_poly = 1;
nelem_vert = 20;

# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-1.0);
Δ = FT((zmax-zmin)/nelem_vert/2)
bc = LandDomainBC(
    bottom_bc = LandComponentBC(
        soil_water = Neumann(bottom_flux)
    ),
    surface_bc = LandComponentBC(
        soil_water =SurfaceDrivenWaterBoundaryConditions(FT;
                                                         precip_model = DrivenConstantPrecip{FT}(incident),
                                                         runoff_model =CoarseGridRunoff{FT}(Δ),
                                                         )
    )
)

depths = [5, 10,20,50,75] .* (-0.01) # m
#data = readdlm("./data/lamont/swat_swc_depth.txt",'\t',String)
data = readdlm("./data/lamont/stamps_swc_depth.txt",'\t',String)
ts = DateTime.(data[:,1], "yyyymmdd")
soil_data = tryparse.(Float64, data[:, 2:end-1])
keep = ((ts .<= cutoff2) .+ (ts .>= cutoff1)) .==2
swc = FT.(soil_data[keep,:][1,:])*0.01
swc[5] = FT(0.15)
θ = Spline1D(reverse(depths), reverse(swc), k=2)

ϑ_l0 = aux -> θ(aux.z)


soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten(FT; n = (aux) ->vgn(aux.z), α = (aux)->vgα(aux.z), L = (aux)->vgL(aux.z)),
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
);

# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60 *60 * 24*61)+t0
dt = FT(25);

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

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

# Determine how often you want output.
n_outputs = 90;

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
filename = joinpath(@__DIR__, "data/lamont/meso_output_fixed_ic.jld2")
save(filename,"dons", dons_arr)


#=
#tfile = joinpath(@__DIR__, "data/lamont/t_meso.csv")
#open(tfile, "w") do io
#    writedlm(io, time_data)
#end

# Get z-coordinate
#z = get_z(solver_config.dg.grid; rm_dupes = true);
#N = length(dons_arr)

mask = z .== depths[1]
l1 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]
f1 = [dons_arr[k]["soil.water.K∇h[3]"][mask][1] for k in 1:N]
mask = z .== round(depths[3]*100)/100
l3 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]
f3 = [dons_arr[k]["soil.water.K∇h[3]"][mask][1] for k in 1:N]
mask = z .== depths[2]
l2 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]
f2 = [dons_arr[k]["soil.water.K∇h[3]"][mask][1] for k in 1:N]
mask = z .== round(depths[4]*100)/100
l4 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]
f4 = [dons_arr[k]["soil.water.K∇h[3]"][mask][1] for k in 1:N]
mask = z .== depths[5]
l5 = [dons_arr[k]["soil.water.ϑ_l"][mask][1] for k in 1:N]
f5 = [dons_arr[k]["soil.water.K∇h[3]"][mask][1] for k in 1:N]
T = typeof(cutoff2 - cutoff1)
steps = T.(time_data*1000)
times = cutoff1 .+ steps
soil_data = soil_data ./ 100

## also plot scan data for elreno site
#d2016 = readdlm("data/lamont/2022_27_YEAR=2016.csv",',')
#scan_data = d2016[2:end,:]
#columns = d2016[1,:]

#scan_date = DateTime.(scan_data[:,2])
#scan_keep = ((scan_date .<cutoff2) .+ (scan_date .> cutoff1)) .==2
#scan_swc = FT.(scan_data[scan_keep,:][:,4:8]) .* 0.01
#scan_depths = FT.([-2,-4,-8,-20, -40])*2.5/100.0




plot1 = plot(times,l1, label = "simulation;lamont", color = "red", title = "-5cm")
scatter!(ts[keep], soil_data[keep,1], ms = 2, color = "blue", label = "stamp;lamont")
#scatter!(scan_date[scan_keep], scan_swc[:,1], ms = 2, color = "green", label = "scan; elreno")
savefig("./data/lamont/layer1_meso.png")

plot2 = plot(times,l2, label = "", color = "red", title= "-10cm")
scatter!(ts[keep], soil_data[keep,2], ms = 2, color = "blue", label = "")
#scatter!(scan_date[scan_keep], scan_swc[:,2], ms = 2, color = "green", label = "")
savefig("./data/lamont/layer2_meso.png")
plot3 = plot(times,l3, label = "", color = "red", title = "-20cm")
scatter!(ts[keep], soil_data[keep,3], ms = 2, color = "blue", label = "")
#scatter!(scan_date[scan_keep], scan_swc[:,3], ms = 2, color = "green", label = "")
savefig("./data/lamont/layer3_meso.png")
plot4 = plot(times,l4, label = "", color = "red", title = "-50cm")
scatter!(ts[keep], soil_data[keep,4], ms = 2, color = "blue", label = "")
#scatter!(scan_date[scan_keep], scan_swc[:,4], ms = 2, color = "green", label = "")
savefig("./data/lamont/layer4_meso.png")
plot5 = plot(times,l5, label = "", color = "red", title = "-75cm")
scatter!(ts[keep], soil_data[keep,5], ms = 2, color = "blue", label = "")
#scatter!(scan_date[scan_keep], scan_swc[:,5], ms = 2, color = "green", label = "")
#plot!(ylim = [0.05,0.45])
savefig("./data/lamont/layer5_meso.png")
=#
