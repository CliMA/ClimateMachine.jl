using MPI
using OrderedCollections
using StaticArrays
using Test
using Statistics
using DelimitedFiles
using Plots
using CLIMAParameters.Planet: cp_i, LH_f0
using Dierckx


using CLIMAParameters
using CLIMAParameters.Planet: cp_i
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SnowModel
using ClimateMachine.Land.SnowModelParameterizations
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
using ClimateMachine.ArtifactWrappers

@testset "Snow Model" begin

    ClimateMachine.init()
    FT = Float64

    soil_water_model = PrescribedWaterModel()
    soil_heat_model = PrescribedTemperatureModel()
    soil_param_functions = nothing

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    depth = FT(1.0)
    snow_parameters = SnowParameters{FT,FT,FT,FT}(0.05,100,depth)
    Q_surf = (t) -> eltype(t)(-9.0*sin(2.0*π/3600/24*t))
    #Q_bott = (t) -> eltype(t)(-1.0*sin(2.0*π/3600/24*t))
    forcing = PrescribedForcing(FT;Q_surf = Q_surf)
    Tave_0 = FT(263.0)
    l_0 = FT(0.0)
    ρe_int0 = volumetric_internal_energy(Tave_0, snow_parameters.ρ_snow, l_0, param_set)
    ic = (aux) -> eltype(aux)(ρe_int0)
    m_snow = SingleLayerSnowModel{typeof(snow_parameters), typeof(forcing),typeof(ic)}(
        snow_parameters,
        forcing,
        ic
    )


    function init_land_model!(land, state, aux, localgeo, time)
        state.snow.ρe_int = land.snow.initial_ρe_int(aux)
    end

    sources = (FluxDivergence{FT}(),)

    m = LandModel(
        param_set,
        m_soil;
        snow = m_snow,
        source = sources,
        init_state_prognostic = init_land_model!,
    )

    N_poly = 1
    nelem_vert = 1

    # Specify the domain boundaries
    zmax = FT(1)
    zmin = FT(0)

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
    timeend = FT(60 * 60 * 48)
    dt = FT(60*60)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    n_outputs = 100;
    
    every_x_simulation_time = ceil(Int, (timeend-t0) / n_outputs);
    
    # Create a place to store this output.
    state_types = (Prognostic(),)
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
   # z = get_z(solver_config.dg.grid; rm_dupes = true);
    N = length(dons_arr)
    ρe_int = [dons_arr[k]["snow.ρe_int"][1] for k in 1:N]
    T_ave = T_snow_ave.(ρe_int, Ref(0.0), Ref(100.0),Ref(param_set))
    coeffs = compute_profile_coefficients.(Q_surf.(time_data), Ref(0.0), Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
    t_profs = get_temperature_profile.(Q_surf.(time_data), Ref(0.0), Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
    z = 0:0.01:depth    
    
    function analytic(p, Q, a, tbar,z,t)
        κ = p.κ_snow
        ρ = p.ρ_snow
        z_snow = p.z_snow
        ν = 2.0*π/24/3600
        c_i = cp_i(param_set)
        ρc_snow = ρ*c_i # assumes no liquid
        D1 = (2*κ/ν/ρc_snow)^0.5
        Q1 = Q
        α = a+ κ/D1
        β = κ/D1
        θ  = ν*t-(z_snow-z)/D1
        mag = sqrt(α^2+β^2)
        δ = atan(α/mag, β/mag)
        
        temp = tbar - Q1/mag*sin(θ+δ)*exp(-(z_snow-z)/D1)
        return temp
    end
    
    
    
    anim = @animate for i ∈ 1:N
        plot(t_profs[i].(z),z, ylim = [0,depth+0.5], xlim = [240,300], label = "our T profile")
        t = time_data[i]
        plot!(analytic.(Ref(snow_parameters), Ref(-9.0), Ref(0.0), Ref(mean(T_ave)),z,Ref(t)),z, label = "analytic profile")
    end
    gif(anim, "snow.gif", fps = 6)
    
end



@testset "driven snow ρe_int" begin

    ClimateMachine.init()
    FT = Float64

    data = readdlm("/Users/katherinedeck/Downloads/ERAwyo_2017_hourly_revised.csv",',')
    #data = readdlm("/Users/shuangma/Downloads/ERAwyo_2017_hourly.csv",',')

    Qsurf = data[:, data[1,:] .== "Qsurf"][2:end,1] ./ 3600 ./ 24 # per second
    Qsurf = -Qsurf
    Tsurf = FT.(data[:, data[1,:] .== "surtsn(K)"][2:end,:])
    # 300:440 - no snowfall.
    start = 300
    range = 300:440
    ρ_snow = FT(mean(data[:, data[1,:] .== "rhosn(kgm-3)"][2:end,:][range]))
    κ_air = FT(0.023)
    κ_ice = FT(2.29)
    κ_snow = FT(κ_air + (7.75*1e-5 *ρ_snow + 1.105e-6*ρ_snow^2)*(κ_ice-κ_air))
    z_snow = FT(mean(data[:,data[1,:] .== "depsn(m)"][2:end,:][range]))
    t0 = start*3600
    t = FT.(0:3600:length(Tsurf)*3600-1) .- t0
    soil_water_model = PrescribedWaterModel()
    soil_heat_model = PrescribedTemperatureModel()
    soil_param_functions = nothing

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

    snow_parameters = SnowParameters{FT,FT,FT,FT}(κ_snow,ρ_snow,z_snow)
    Qsurf_spline = Spline1D(t, Qsurf)
    function Q_surf(t::FT, Q::Spline1D) where {FT}
        return Q(t)
    end
    

    forcing = PrescribedForcing(FT;Q_surf = (t) -> eltype(t)(Q_surf(t,Qsurf_spline)))
    Tave_0 = mean(Tsurf)
    l_0 = FT(0.0)
    ρe_int0 = volumetric_internal_energy(Tave_0, snow_parameters.ρ_snow, l_0, param_set)
    ic = (aux) -> eltype(aux)(ρe_int0)
    m_snow = SingleLayerSnowModel{typeof(snow_parameters), typeof(forcing),typeof(ic)}(
        snow_parameters,
        forcing,
        ic
    )


    function init_land_model!(land, state, aux, localgeo, time)
        state.snow.ρe_int = land.snow.initial_ρe_int(aux)
    end

    sources = (FluxDivergence{FT}(),)

    m = LandModel(
        param_set,
        m_soil;
        snow = m_snow,
        source = sources,
        init_state_prognostic = init_land_model!,
    )

    N_poly = 1
    nelem_vert = 1

    # Specify the domain boundaries
    zmax = FT(1)
    zmin = FT(0)

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
    timeend = FT((range[end]-start)*3600)
    dt = FT(60*60)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    n_outputs = length(range);
    
    every_x_simulation_time = ceil(Int, (timeend-t0) / n_outputs);
    
    # Create a place to store this output.
    state_types = (Prognostic(),)
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
    z = 0:0.01:FT(snow_parameters.z_snow)
    N = length(dons_arr)
    ρe_int = [dons_arr[k]["snow.ρe_int"][1] for k in 1:N]
    T_ave = T_snow_ave.(ρe_int, Ref(0.0), Ref(100.0),Ref(param_set))
    coeffs = compute_profile_coefficients.(Q_surf.(time_data, Ref(Qsurf_spline)), Ref(0.0), Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))
    t_profs = get_temperature_profile.(Q_surf.(time_data, Ref(Qsurf_spline)), Ref(0.0), Ref(snow_parameters.z_snow), Ref(snow_parameters.ρ_snow), Ref(snow_parameters.κ_snow), ρe_int, Ref(param_set))

"""tsurf_pw = Array{FT,1}(undef, N)
    tsurf_era5 = Array{FT,1}(undef, N)
    snowf_era5 = Array{FT,1}(undef, N)
    anim = @animate for i ∈ 1:N
        plot(t_profs[i].(z),z, ylim = [0,1.5], xlim = [240,300], label = "Piecewise model")
        t = time_data[i]
        scatter!([Tsurf[Int64.(time_data[i]/3600)+1]], [FT(snow_parameters.z_snow)], label = "ERA5 Tsurf")
        
        tsurf_pw[i]   = t_profs[i].(FT(snow_parameters.z_snow))
        tsurf_era5[i] = Tsurf[Int64.(time_data[i]/3600)+1]
        snowf_era5[i] = snowf[Int64.(time_data[i]/3600)+1]
        println(tsurf_pw[i])
        println(tsurf_era5[i])"""
        
    tsurf_pw = [coeffs[k][1] for k in 1:N]
    tsurf_era5 = Tsurf[range]
    anim = @animate for i ∈ 1:N
        plot(t_profs[i].(z),z, ylim = [0,1.5], xlim = [240,300], label = "Piecewise model")
        t = time_data[i]
        scatter!([Tsurf[range[i]]], [FT(snow_parameters.z_snow)], label = "ERA5 Tsurf")
    end
    gif(anim, "snow2.gif", fps = 6)


    RMSEv  = round(sqrt(mean((tsurf_pw - tsurf_era5).^2)),digits = 2) 
    titlev = "snow_scatter_rho+100"
    snowscatter = plot(tsurf_pw, tsurf_era5, seriestype = :scatter, reg = true,
    title = titlev,
    legend = false,
    ylim = [220,280],xlim=[220,280],
    xlab = "Tsurf Piecewise model", ylab = "Tsurf ERA5")
    annotate!(260,270,text(string("RMSE = ", RMSEv), :left, 12))
    Plots.abline!(1, 1, line=:dash)
    savefig(snowscatter,string(titlev,".png"))

    snowts = plot(1:28,tsurf_pw)

    scatter(range, tsurf_pw, label = "piecewise model")
    scatter!(range, tsurf_era5, label = "ERA5")
    plot!(xlabel = "hour", ylabel = "T surf (K)")
    savefig("snow_scatter2.png")
end 
    
