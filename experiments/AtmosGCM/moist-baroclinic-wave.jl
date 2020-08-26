#!/usr/bin/env julia --project

using ArgParse
using LinearAlgebra
using StaticArrays
using Test

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.TurbulenceClosures
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics:
    air_density, air_temperature, total_energy, internal_energy, PhasePartition, air_pressure
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet:
    day, grav, R_d, Omega, planet_radius, MSLP, T_triple, press_triple, LH_v0, R_v, LH_s0, LH_f0
using CLIMAParameters.Atmos.SubgridScale:
    C_drag

struct EarthParameterSet <: AbstractEarthParameterSet end

using Printf

#  ---------------------------
#press_triple(::EarthParameterSet) = 610.78      # from Thatcher and Jablonowski (2016)
#C_drag(::EarthParameterSet) = 0.0044            # "
import CLIMAParameters.Planet.LH_v0
import CLIMAParameters.Planet.LH_s0
import CLIMAParameters.Planet.LH_f0
#LH_v0(::EarthParameterSet) = 0.0
#LH_s0(::EarthParameterSet) = 0.0
#LH_f0(::EarthParameterSet) = 0.0

# driver-specific parameters added here
T_sfc_pole(::EarthParameterSet) = 271.0

remove_q = false
#    |
#    |
#  \ | /
#    V
rq = remove_q
if rq == "0micro"
  function get_source()
    _list = (Gravity(), Coriolis(), NudgeToSaturation(), RemovePrecipitation(), )
    return _list
  end
elseif rq == "none"  
  function get_source()
    _list = (Gravity(), Coriolis(), NudgeToSaturation(), )
    return _list
  end
end

diffn = false
#    |
#    |
#  \ | /
#    V
rq = diffn
if rq == "SmagLil"
  function get_diffn(FT)
    c_smag = FT(0.21);   # Smagorinsky constant
    _list = SmagorinskyLilly(c_smag) 
    return _list
  end
elseif rq == "ThatJab"
  function get_diffn(FT)
    _list = ( define_later  )
    return _list
  end
elseif rq == "none"  
  function get_diffn(FT)
    _list =  ConstantViscosityWithDivergence(FT(0) )
    return _list
  end
end
#  ---------------------------


const param_set = EarthParameterSet()

include("init_funcs.jl")
include("bc_funcs.jl")

function config_baroclinic_wave(FT, param_set, poly_order, resolution, with_moisture)
    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(275), FT(75), FT(45e3))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up the atmosphere model
    exp_name = "MBcW"
    domain_height::FT = 30e3 # distance between surface and top of atmosphere (m)
    if with_moisture
        hyperdiffusion = EquilMoistBiharmonic(FT(8 * 3600))
        moisture = EquilMoist{FT}()
    else
        hyperdiffusion = DryBiharmonic(FT(8 * 3600))
        moisture = DryModel()
    end

    _C_drag = C_drag(param_set)
    bulk_flux = Varying_SST_TJ16(param_set, SphericalOrientation(), moisture)
    problem = AtmosProblem(
        boundarycondition = (
            AtmosBC(),
            AtmosBC(),
        ),
        init_state_prognostic = init_baroclinic_wave!,
    )

    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        problem = problem,
        ref_state = ref_state,
        turbulence = ConstantViscosityWithDivergence(FT(0) ),
        hyperdiffusion = hyperdiffusion,
        moisture = moisture,
        source = (Gravity(), Coriolis(), NudgeToSaturation(), ),
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_baroclinic_wave!;
        model = model,
    )

    return config
end

function main()
    # add a command line argument to specify whether to use a moist setup
    # TODO: this will move to the future namelist functionality
    bw_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(bw_args, "BaroclinicWave")
    @add_arg_table! bw_args begin
        "--with-moisture"
        help = "use a moist setup"
        action = :store_const
        constant = false
        default = true
    end

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = bw_args)
    with_moisture = cl_args["with_moisture"]

    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = 3                           # discontinuous Galerkin polynomial order
    n_horz = 12                              # horizontal element number
    n_vert = 6                               # vertical element number
    n_days::FT = 10
    timestart::FT = 0                        # start time (s)
    timeend::FT = n_days * day(param_set)    # end time (s)

    # Set up driver configuration
    driver_config =
        config_baroclinic_wave(FT, param_set, poly_order, (n_horz, n_vert), with_moisture)

    # Set up experiment
    ode_solver_type = ClimateMachine.IMEXSolverType(
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
        split_explicit_implicit = true,
        discrete_splitting = false,
    )

    CFL = FT(0.1) # target acoustic CFL number

    # time step is computed such that the horizontal acoustic Courant number is CFL
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        ode_solver_type = ode_solver_type,
        CFL_direction = HorizontalDirection(),
        diffdir = HorizontalDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    filterorder = 20
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            solver_config.dg.grid,
            filter,
            state_auxiliary = solver_config.dg.state_auxiliary,
        )
        nothing
    end

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot",),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbfilter,),
        #user_callbacks = (cbtmarfilter, cbfilter),
        check_euclidean_distance = true,
    )
end

function config_diagnostics(FT, driver_config)
    interval = "40000steps" # chosen to allow a single diagnostics collection

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info
    boundaries = [
        FT(-90.0) FT(-180.0) _planet_radius
        FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(1), FT(1), FT(1000)) # in (deg, deg, m)
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )

    dgngrp = setup_atmos_default_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )

    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

main()
