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

import ClimateMachine.DGMethods.FVReconstructions: FVLinear
using Printf
FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));
dts = [6,1.5,0.75,0.2]
p = [10,20,40,80]
for i in [1,2,3,4]
    nelem_vert  = p[i]
    dt = dts[i]
    N_poly = 1
    # Specify the domain boundaries.
    zmax = FT(0);
    zmin = FT(-2);
    
    Δz = abs(FT((zmin - zmax) / nelem_vert / 2))
    
    # # Parameters
    νp = 0.4
    Ksat = 6.94e-5/60
    S_s = 5e-4
    vg_α = 1.0
    vg_n = 2.0
    vg_m = 1.0-1.0/vg_n
    θ_r = 0.08
    wt_depth = -1.0
    precip_rate = (3.3e-4)/60
    precip_time = 200*60
    topo_max = 0.2
    soil_heat_model = PrescribedTemperatureModel();
    
    soil_param_functions = SoilParamFunctions{FT}(
        porosity = νp,
        θ_r = θ_r,
        Ksat = Ksat,
        S_s = S_s,
    );
    heaviside(x) = 0.5 * (sign(x) + 1)
    precip_of_t = (t) -> eltype(t)(-precip_rate*heaviside(precip_time-t))
    function he(z, z_interface, ν, Ss, vga, vgn, θ_r)
        m = 1.0-1.0/vgn
        if z < z_interface
            return -Ss * (z - z_interface) + ν
        else
            return (ν-θ_r) * (1 + (vga * (z - z_interface))^vgn)^(-m)+θ_r
        end
    end
    ϑ_l0 = (aux) -> eltype(aux)(he(aux.z, wt_depth, νp, S_s, vg_α, vg_n, θ_r))
    
    bc =  LandDomainBC(
        bottom_bc = LandComponentBC(soil_water = Neumann((aux,t)->eltype(aux)(0.0))),
        surface_bc = LandComponentBC(#soil_water = Neumann((aux,t)->eltype(aux)(0.0)),
                                     #soil_water = Dirichlet((aux,t)->eltype(aux)(0.4)),),
                                     soil_water = SurfaceDrivenWaterBoundaryConditions(FT;
                                                                                       precip_model = DrivenConstantPrecip{FT}(precip_of_t),
                                                                                       runoff_model = CoarseGridRunoff{FT}(Δz)
                                                                                       )
                                     ),
    )
    
    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = vanGenuchten{FT}(n = vg_n,  α = vg_α),
        initialϑ_l = ϑ_l0,
    );
    
    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);
    
    function init_land_model!(land, state, aux, localgeo, time)
        state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
        
    end
    model = LandModel(
        param_set,
        m_soil;
        boundary_conditions = bc,
        init_state_prognostic = init_land_model!,
    );
    
    driver_config = ClimateMachine.SingleStackConfiguration(
        "LandModel",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        model;
        zmin = zmin,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
        fv_reconstruction = FVLinear(),
    )
    
    
    # Choose the initial and final times, as well as a timestep.
    t0 = FT(0)
    timeend = FT(60* 200)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    n_outputs = 1000;
    mygrid = solver_config.dg.grid
    every_x_simulation_time = ceil(Int, timeend / n_outputs)
    state_types = (Prognostic(), Auxiliary(), GradientFlux())
    dons_arr =
        Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0] # store time data
    ol_source = FT[0]
    
    # We specify a function which evaluates `every_x_simulation_time` and returns
    # the state vector, appending the variables we are interested in into
    # `dons_arr`.
    
    callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
        dons = dict_of_nodal_states(solver_config, state_types; interp = true)
        push!(dons_arr, dons)
        t = gettime(solver_config.solver)
        push!(time_data, t)
        flux = -dons["soil.water.K∇h[3]"][end]
        precip = precip_of_t(t)
        push!(ol_source, -(precip-flux))
        nothing
    end
    
    # # Run the integration
    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))
    z = get_z(solver_config.dg.grid; rm_dupes = true)
    
    output_step = time_data[2]-time_data[1]
    OL = output_step * sum(ol_source[2:end])
    Input = -output_step * sum(precip_of_t.(time_data[2:end]))
    init = 0.5 .* (he.(z[2:end], Ref(wt_depth), Ref(νp), Ref(S_s), Ref(vg_α), Ref(vg_n), Ref(θ_r)) .+he.(z[1:end-1], Ref(wt_depth), Ref(νp), Ref(S_s), Ref(vg_α), Ref(vg_n), Ref(θ_r)))
    soil_t0 = sum(init)*Δz*2
    final = 0.5 .* (dons_arr[end]["soil.water.ϑ_l"][2:end] .+ dons_arr[end]["soil.water.ϑ_l"][1:end-1])
    soil_tf = sum(final)*Δz*2
    soil_delta = soil_tf - soil_t0
    @printf("%lf %le\n", Δz*2, Input-OL-soil_delta)
end
