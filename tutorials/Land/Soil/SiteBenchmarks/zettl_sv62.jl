# # Zettl - site SV62
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
import ClimateMachine.DGMethods.FVReconstructions: FVLinear

const FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

# # Set up the soil model

soil_heat_model = PrescribedTemperatureModel();


function ks(z::F) where {F}
    factor = F(1/100/60)# given in cm/min
    if z >= F(-0.06)
        k = F(0.62)
    elseif z >= F(-0.15)
        k = F(0.216)
    elseif z >=  F(-0.27)
        k = F(0.47)
    elseif z>= F(-0.32)
        k = F(0.719)
    elseif z>=F(-0.37)
        k = F(0.576)
    elseif z>=F(-0.42)
        k = F(0.554)
    elseif z >=  F(-0.46)
        k = F(0.505)
    elseif z>= F(-0.54)
        k = F(1.311)
    elseif z>=F(-0.6)
        k = F(0.750)
    elseif z>=F(-0.67)
        k = F(0.789)
    elseif z >=  F(-0.71)
        k = F(1.059)
    elseif z>= F(-0.78)
        k = F(1.865)
    elseif z>=F(-0.81)
        k = F(2.109)
    elseif z>=F(-0.88)
        k = F(2.636)
    elseif z >=  F(-0.97)
        k = F(1.901)
    else
        k = F(1.268)
    end
    return k*factor
end

function vgα(z::F) where {F}
    factor = F(100)
    if z >= F(-0.06)
        k = F(0.129)
    elseif z >= F(-0.15)
        k = F(0.108)
    elseif z >=  F(-0.27)
        k = F(0.051)
    elseif z>= F(-0.32)
        k = F(0.114)
    elseif z>=F(-0.37)
        k = F(0.093)
    elseif z>=F(-0.42)
        k = F(0.095)
    elseif z >=  F(-0.46)
        k = F(0.152)
    elseif z>= F(-0.54)
        k = F(0.176)
    elseif z>=F(-0.6)
        k = F(0.119)
    elseif z>=F(-0.67)
        k = F(0.101)
    elseif z >=  F(-0.71)
        k = F(0.099)
    elseif z>= F(-0.78)
        k = F(0.105)
    elseif z>=F(-0.81)
        k = F(0.105)
    elseif z>=F(-0.88)
        k = F(0.106)
    elseif z >=  F(-0.97)
        k = F(0.107)
    else
        k = F(0.109)
    end
    return k*factor
end

function vgn(z::F) where {F}
    if z >= F(-0.06)
        k = F(1.986)
    elseif z >= F(-0.15)
        k = F(1.875)
    elseif z >=  F(-0.27)
        k = F(2.296)
    elseif z>= F(-0.32)
        k = F(1.736)
    elseif z>=F(-0.37)
        k = F(2.07)
    elseif z>=F(-0.42)
        k = F(1.994)
    elseif z >=  F(-0.46)
        k = F(2.044)
    elseif z>= F(-0.54)
        k = F(1.711)
    elseif z>=F(-0.6)
        k = F(1.955)
    elseif z>=F(-0.67)
        k = F(1.982)
    elseif z >=  F(-0.71)
        k = F(1.931)
    elseif z>= F(-0.78)
        k = F(1.970)
    elseif z>=F(-0.81)
        k = F(1.851)
    elseif z>=F(-0.88)
        k = F(1.997)
    elseif z >=  F(-0.97)
        k = F(2.019)
    else
        k = F(2.024)
    end
    return k
end


function θr(z::F) where {F}
    if z >= F(-0.67)
        k = F(0.0)
    elseif z >= F(-0.81)
        k = F(0.005)
    elseif z >=  F(-0.88)
        k = F(0.003)
    else
        k = F(0.004)
    end
    return k
end


function ν(z::F) where {F}
    if z >= F(-0.06)
        k = F(0.467)
    elseif z >= F(-0.15)
        k = F(0.3)
    elseif z >=  F(-0.27)
        k = F(0.379)
    elseif z>= F(-0.32)
        k = F(0.388)
    elseif z>=F(-0.37)
        k = F(0.389)
    elseif z>=F(-0.42)
        k = F(0.405)
    elseif z >=  F(-0.46)
        k = F(0.4)
    elseif z>= F(-0.54)
        k = F(0.324)
    elseif z>=F(-0.6)
        k = F(0.311)
    elseif z>=F(-0.67)
        k = F(0.305)
    elseif z >=  F(-0.71)
        k = F(0.302)
    elseif z>= F(-0.78)
        k = F(0.305)
    elseif z>=F(-0.81)
        k = F(0.302)
    elseif z>=F(-0.88)
        k = F(0.289)
    elseif z >=  F(-0.97)
        k = F(0.316)
    else
        k = F(0.332)
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

S_s = 1e-3
wpf = WaterParamFunctions(FT; Ksat = (aux)->ks(aux.z), S_s = S_s, θ_r = (aux)->θr(aux.z))
soil_param_functions = SoilParamFunctions(FT; porosity = (aux)->ν(aux.z), water = wpf)
kstop = FT(0.62/60/100)
surface_flux = (aux,t)-> eltype(aux)(-kstop*(0.1-aux.soil.water.h)/0.0275)
bottom_flux = (aux, t) -> aux.soil.water.K * eltype(aux)(-1)

N_poly = 1;
nelem_vert = 55;

# Specify the domain boundaries.
zmax = FT(0);
zmin = FT(-1.1);
Δ = FT((zmax-zmin)/nelem_vert/2)
bc = LandDomainBC(
    bottom_bc = LandComponentBC(
        soil_water = Neumann(bottom_flux)
    ),
surface_bc = LandComponentBC(
    soil_water = Neumann(surface_flux)
    )
)

ϑ_l0 = aux -> eltype(aux)(0.04)


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
    fv_reconstruction = FVLinear()
);

# Choose the initial and final times, as well as a timestep.
t0 = FT(0)
timeend = FT(60 *60)+t0
dt = FT(0.01);

# Create the solver configuration.
solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
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
n_outputs = 60;

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
z = get_z(solver_config.dg.grid; rm_dupes = true);


 plot(dons_arr[1]["soil.water.ϑ_l"],z, label = "initial")
 plot!(dons_arr[9]["soil.water.ϑ_l"], z, label = "8min")

 plot!(dons_arr[17]["soil.water.ϑ_l"], z, label = "16min")

 plot!(dons_arr[25]["soil.water.ϑ_l"], z, label = "24min")

 plot!(dons_arr[33]["soil.water.ϑ_l"], z, label = "32min")

 plot!(dons_arr[41]["soil.water.ϑ_l"], z, label = "40min")

 plot!(dons_arr[60]["soil.water.ϑ_l"], z, label = "59min")

 plot!(legend = :bottomright)

 plot!(xlim = [0,0.5])

 plot!(xlim = [0,0.6])

 plot!(ylim = [-1.1,0])

 plot!(ylabel = "Depth (m)")

 plot!(xlabet  = "Volumeteric Water Content")

 plot!(xlabel  = "Volumeteric Water Content")

plot!(title = "SV62; infiltration")
savefig("./sv62.png")
