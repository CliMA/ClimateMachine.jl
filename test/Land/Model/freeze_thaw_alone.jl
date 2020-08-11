# Test that freeze thaw alone reproduces expected behavior: exponential behavior
# for liquid water content, ice content, and total water conserved
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
using CLIMAParameters.Planet: ρ_cloud_liq
using CLIMAParameters.Planet: ρ_cloud_ice

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.SoilHeatParameterizations
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

@testset "Freeze thaw alone" begin
    FT = Float64;
    #This shouldnt be defined in multiple places, but is hacky anyways?
    struct tmp_model <: BalanceLaw end
    struct tmp_param_set <: AbstractParameterSet end

    function get_grid_spacing(N_poly::Int64, nelem_vert::Int64, zmax::FT, zmin::FT)
        test_config = ClimateMachine.SingleStackConfiguration(
            "TmpModel",
            N_poly,
            nelem_vert,
            zmax,
            tmp_param_set(),
            tmp_model();
            zmin = zmin,
        );

        Δ = min_node_distance(test_config.grid)
        return Δ
    end
  
    function init_soil_water!(land, state, aux, coordinates, time)
        myf = eltype(state)
        state.soil.water.ϑ_l = myf(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_ice = myf(land.soil.water.initialθ_ice(aux))
    end;


    ClimateMachine.init();

    N_poly = 5;
    nelem_vert = 50;
    zmax = FT(0);
    zmin = FT(-1)
    t0 = FT(0)
    timeend = FT(30)
    dt = FT(0.05)

    n_outputs = 30;
    every_x_simulation_time = ceil(Int, timeend / n_outputs);
    Δ = get_grid_spacing(N_poly, nelem_vert, zmax, zmin)
    cs = FT(3e6)
    κ = FT(1.5)
    τft = FT(cs * Δ^FT(2.0) / κ)
    determine_τft = (land, state, aux, t) -> τft
    freeze_thaw_source = FreezeThaw(τft = determine_τft)
    
    soil_param_functions =
        SoilParamFunctions{FT}(porosity = 0.75, Ksat = 0.0, S_s = 1e-3,)

    zero_value = FT(0.0)
    bottom_flux = (aux, t) -> zero_value
    surface_flux = (aux, t) -> zero_value
    surface_state = nothing
    bottom_state = nothing
    initial_value = FT(0.33)
    ϑ_l0 = (aux) -> initial_value
    soil_water_model = SoilWaterModel(
        FT;
        initialϑ_l = ϑ_l0,
        dirichlet_bc = Dirichlet(
            surface_state = surface_state,
            bottom_state = bottom_state,
        ),
        neumann_bc = Neumann(
            surface_flux = surface_flux,
            bottom_flux = bottom_flux,
        ),
    )

    temperature_value = FT(267.15)
    soil_heat_model = PrescribedTemperatureModel((aux,t) -> temperature_value)

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = (freeze_thaw_source,)
    m = LandModel(
        param_set,
        m_soil;
        source = sources,
        init_state_prognostic = init_soil_water!,
    )

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



    solver_config =
        ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
    mygrid = solver_config.dg.grid;
    Q = solver_config.Q;
    
    ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
    θ_ice_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :θ_ice)

    all_data = Dict([k => Dict() for k in 1:n_outputs]...)
    step = [1];
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        ϑ_l = Q[:, ϑ_l_ind, :]
        θ_ice = Q[:, θ_ice_ind, :]
        all_vars = Dict{String, Array}(
            "t" => [t],
            "ϑ_l" => ϑ_l,
            "θ_ice" => θ_ice,
        )
        all_data[step[1]] = all_vars
        step[1] += 1
        nothing
    end

    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

    t = ODESolvers.gettime(solver_config.solver)
    ϑ_l = Q[:, ϑ_l_ind, :]
    θ_ice = Q[:, θ_ice_ind, :]
    all_vars = Dict{String, Array}(
        "t" => [t],
        "ϑ_l" => ϑ_l,
        "θ_ice" => θ_ice,
    )
    
    all_data[n_outputs] = all_vars

    m_liq = [ρ_cloud_liq(param_set)*mean(all_data[k]["ϑ_l"]) for k in 1:n_outputs]
    m_ice = [ρ_cloud_ice(param_set)*mean(all_data[k]["θ_ice"]) for k in 1:n_outputs]
    t = [all_data[k]["t"][1] for k in 1:n_outputs]
    total_water = m_ice+m_liq
    m_liq_of_t = m_liq[1]*exp.(-1.0.*(t.-t[1])./τft)
    m_ice_of_t = -m_liq_of_t .+ (m_ice[1]+m_liq[1])

    @test mean(abs.(m_ice+m_liq .- total_water)) < 1e-9
    @test mean(abs.(m_liq .- m_liq_of_t)) < 1e-9
    @test mean(abs.(m_ice .- m_ice_of_t)) < 1e-9
end


#    function determine_τft(land, state, aux, t, spacing)
#        soil = land.soil
#        water = soil.water
#        heat = soil.heat
#        myf = eltype(state)
#        _T_ref = myf(T_0(land.param_set))
#        _LH_f0 = myf(LH_f0(land.param_set))#
#
#        _ρ_i = myf(ρ_cloud_ice(land.param_set))
#        _cp_i = myf(cp_i(land.param_set) * _ρ_i)
# 
#        _ρ_l = myf(ρ_cloud_liq(land.param_set))
#        _cp_l = myf(cp_l(land.param_set) * _ρ_l)##

#        T = get_temperature(heat,aux,t,state)
#        ϑ_l, θ_ice = get_water_content(soil.water, aux, state, t)
#        θ_l = volumetric_liquid_fraction(ϑ_l, soil.param_functions.porosity)
        #c_ds = soil.param_functions.c_ds
        #cs = volumetric_heat_capacity(θ_l, θ_ice, c_ds, _cp_l, _cp_i)
#        cs = myf(3e6)
        #κ_dry = soil.param_functions.κ_dry
        #S_r = relative_saturation(θ_l, θ_ice, soil.param_functions.porosity)
        #kersten  = kersten_number(θ_ice, S_r, soil.param_functions.a, soil.param_functions.b,
        #                      soil.param_functions.ν_om, soil.param_functions.ν_sand,
        #                      soil.param_functions.ν_gravel)
        #κ_sat = saturated_thermal_conductivity(θ_l, θ_ice, soil.param_functions.κ_sat_unfrozen,
        #                                   soil.param_functions.κ_sat_frozen)
        #κ = thermal_conductivity(κ_dry, kersten, κ_sat)
 #       κ = myf(1.5)
 #       τLTE = cs * spacing^2.0 / κ
 #       m_water = _ρliq*θ_l*heaviside(_T_ref - T) +_ρice*θ_ice*heaviside(T - _T_ref)
 #       div_flux = get_divergence_of_flux(land, state, aux, t)
 #       τPT = m_water*_LH_f0/div_flux
 #       τft = max(τLTE, τPR)
 #       return τft

