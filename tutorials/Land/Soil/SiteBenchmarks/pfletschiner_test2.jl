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
    wpf_bc = WaterParamFunctions(
        FT;
        Ksat = 7.2e-5,
        θ_r = 0.067,
        S_s = 1e-3,
    )
    soil_param_functions = SoilParamFunctions(FT; porosity = 0.31, water = wpf_bc)

irr1(t) = ((t<FT(3600*7.5)) & (t>FT(3600*2.5))) ? FT(-170/5/60/60/1000) : FT(0)
irr2(t) = ((t>FT(3600*122.5)) & (t<FT(3600*124.5))) ? FT(-25/2/60/60/1000) : FT(0)
evap1(t) = ((t<FT(3600*50.5)) & (t>FT(3600*2.5))) ? FT(0.009/60/60/100) : FT(0)
evap2(t) = ((t>FT(3600*50.5)) & (t<FT(3600*122.5))) ? FT(0.005/60/60/100) : FT(0)
evap3(t) = ((t>FT(3600*122.5)) & (t<FT(3600*200.5))) ? FT(0.015/60/60/100) : FT(0)
#evap3(t) = t>FT(3600*200.5) ? FT(0.01/60/60/100) : FT(0)

    surface_flux = (aux, t) -> eltype(aux)(irr1(t)+irr2(t)+evap1(t)+evap2(t) + evap3(t))

Δ = FT(0.0092)
function lower_bc(aux, t)
    pressure_head = aux.soil.water.h-aux.z
    if pressure_head < FT(-0.62)
        flux = FT(0.0)
    else
        proposed_flux = -aux.soil.water.K*(aux.soil.water.h-aux.z-FT(-0.62)+Δ)/Δ
        if proposed_flux < FT(0)
            flux = proposed_flux
        else
            flux = FT(0.0)
        end
    end
    return(flux)
end

bottom_flux = (aux,t) -> lower_bc(aux,t)
    ϑ_l0 = (aux) -> eltype(aux)(0.07)

    bc = LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann(bottom_flux)),
        surface_bc = LandComponentBC(soil_water = Neumann(surface_flux)),
    )
#vg_hydraulics =  vanGenuchten(FT; n = 7.5, α = 5.5)
bc_hydraulics =  BrooksCorey(FT; m = 1.0/2.91, ψb = 0.421)

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

    dt = FT(1.1)

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
f_bottom = [dons_arr[k]["soil.water.K∇h[3]"][1] for k in 1:length(dons_arr)]
t = time_data ./ 60


l64 = [dons_arr[k]["soil.water.ϑ_l"][16] for k in 1:length(dons_arr)]
l14 = [dons_arr[k]["soil.water.ϑ_l"][43] for k in 1:length(dons_arr)]


plot1 = plot(dons_arr[1]["soil.water.ϑ_l"],z, label = "initial")
plot!(dons_arr[50]["soil.water.ϑ_l"],z, label = "t = 3.3h")
plot!(dons_arr[100]["soil.water.ϑ_l"],z, label = "t = 6.6h")
plot!(dons_arr[150]["soil.water.ϑ_l"],z, label = "t = 10h")

plot!(xlim = [0.0, 0.27])
plot!(legend = :bottomright)
plot!(ylabel = "Depth [m]")
plot!(xlabel = "Volumetric water content")
plot!(ylim = [-0.92, 0])
plot!(xticks = [0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18])
plot2 = plot(t./60, f_bottom *60 *1000, ylabel = "Discharge (mm/min)", xlabel = "Time after infiltration start (hr)", xguide_position = :top, xmirror = true, label="", yguidefontsize = 6, yticks = [0,0.1,0.3,0.5])
plot(plot2,plot1, layout = grid(2,1, heights = [0.2,0.8]))
#end

plot(time_data ./3600,-cumsum(f_bottom).*(time_data[2]-time_data[1]).*100)
