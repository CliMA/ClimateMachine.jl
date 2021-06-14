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
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state

#@testset "Richard's equation - Sand van Genuchten test" begin
    ClimateMachine.init()
    FT = Float64

    function init_soil_water!(land, state, aux, localgeo, time)
        myfloat = eltype(aux)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myfloat(land.soil.water.initialθ_i(aux))
    end

    soil_heat_model = PrescribedTemperatureModel()
#    wpf_vg = WaterParamFunctions(
#        FT;
#        Ksat = 7.8 / (60 * 100),
#        θ_r = 0.005,
#        S_s = 1e-3,
#    )

    wpf_bc = WaterParamFunctions(
        FT;
        Ksat = 1.3e-5,
        θ_r = 0.0,
        S_s = 1e-3,
    )
    soil_param_functions = SoilParamFunctions(FT; porosity = 0.367, water = wpf_bc)

irr1(t) = t<FT(3600*5) ? FT(-170/5/60/60/1000) : FT(0)
irr2(t) = t>FT(3600*125) ? FT(-25/2/60/60/1000) : FT(0)
irr3(t) = t>FT(3600*127) ? FT(25/2/60/60/1000) : FT(0)

    surface_flux = (aux, t) -> eltype(aux)(irr1(t)+irr2(t)+irr3(t))
#bottom_flux = (aux, t) -> aux.soil.water.K * eltype(aux)(-1)

Δ = FT(0.0092)
lower_bc(aux, t) = aux.soil.water.h-aux.z < FT(-0.6) ? FT(0.0) : -aux.soil.water.K*(aux.soil.water.h-aux.z-FT(-0.6))/Δ
bottom_flux = (aux,t) -> lower_bc(aux,t)
    ϑ_l0 = (aux) -> eltype(aux)(0.07)

    bc = LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann(bottom_flux)),
        surface_bc = LandComponentBC(soil_water = Neumann(surface_flux)),
    )
#vg_hydraulics =  vanGenuchten(FT; n = 7.5, α = 5.5)
bc_hydraulics =  BrooksCorey(FT; m = 1.0/1.19, ψb = 0.315)

    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = bc_hydraulics,
        initialϑ_l = ϑ_l0,
    )

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    N_poly = 1
    nelem_vert = 50

    # Specify the domain boundaries
    zmax = FT(0)
    zmin = FT(-0.92)
    Δz = abs(FT((zmin - zmax) / nelem_vert / 2))

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
    timeend = FT(60 * 60*250)

    dt = FT(15)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
n_outputs = 750;

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
f_bottom = [dons_arr[k]["soil.water.K∇h[3]"][1] for k in 1:n_outputs]
t = time_data ./ 60


l64 = [dons_arr[k]["soil.water.ϑ_l"][16] for k in 1:length(dons_arr)]
l14 = [dons_arr[k]["soil.water.ϑ_l"][43] for k in 1:length(dons_arr)]


plot1 = plot(dons_arr[1]["soil.water.ϑ_l"],z, label = "initial")
plot!(dons_arr[11]["soil.water.ϑ_l"],z, label = "t = 1h")
plot!(dons_arr[21]["soil.water.ϑ_l"],z, label = "t = 2h")
plot!(dons_arr[31]["soil.water.ϑ_l"],z, label = "t = 2h")
plot!(dons_arr[41]["soil.water.ϑ_l"],z, label = "t = 4h")
plot!(dons_arr[51]["soil.water.ϑ_l"],z, label = "t = 5h")
plot!(xlim = [0.0, 0.27])
plot!(legend = :bottomright)
plot!(ylabel = "Depth [m]")
plot!(xlabel = "Volumetric water content")
plot!(ylim = [-0.92, 0])
plot!(xticks = [0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18])
plot2 = plot(t[2:end]./60, f_bottom *60 *1000, ylabel = "Discharge (mm/min)", xlabel = "Time after infiltration start (hr)", xguide_position = :top, xmirror = true, label="", yguidefontsize = 6, yticks = [0,0.1,0.3,0.5])
plot(plot2,plot1, layout = grid(2,1, heights = [0.2,0.8]))
#end

plot(time_data[2:end] ./3600,-cumsum(f_bottom).*(time_data[2]-time_data[1]).*100)
