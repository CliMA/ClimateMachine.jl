#
#
# Run against 
#
# origin	https://github.com/clima/climatemachine.jl (fetch)
# origin	https://github.com/clima/climatemachine.jl (push)
#
# commit e674f19012704c16c8e99740cb133031b563f202 (HEAD -> master, origin/staging, origin/master, origin/HEAD)
# Merge: 199cb7ab a6c15d5c
# Author: bors[bot] <26634292+bors[bot]@users.noreply.github.com>
# Date:   Wed Jul 8 01:27:58 2020 +0000
#
#

# LinearHBModel - solver_config.solver.rhs_implicit!()

# State vector  - typeof(solver_config.Q).parameters[2]
# NamedTuple{(:u, :η, :θ),Tuple{StaticArrays.SArray{Tuple{2},Float64,1,2},Float64,Float64}}

# Setting values and evaluating

# cd /Users/chrishill/projects/github.com/christophernhill/cm-20200708
# /Applications/Julia-1.4.app/Contents/Resources/julia/bin/julia --project=@.
	using Pkg
	Pkg.instantiate()

using MPI

using StaticArrays

using Random

using ClimateMachine
using ClimateMachine.VariableTemplates

using ClimateMachine.MPIStateArrays

using ClimateMachine.GenericCallbacks

using ClimateMachine.StateCheck

ClimateMachine.init()

	using ClimateMachine.ODESolvers
	using ClimateMachine.Mesh.Filters
	using ClimateMachine.Mesh.Grids: polynomialorder
	using ClimateMachine.Ocean.HydrostaticBoussinesq
	using CLIMAParameters
	using CLIMAParameters.Planet: grav
	struct EarthParameterSet <: AbstractEarthParameterSet end
	const param_set = EarthParameterSet()

using ClimateMachine.BalanceLaws
using ClimateMachine.VariableTemplates

using LinearAlgebra: dot, Diagonal


import ...BalanceLaws:
    vars_state_conservative,
    vars_state_auxiliary,
    vars_state_gradient,
    vars_state_gradient_flux,
    init_state_conservative!,
    init_state_auxiliary!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!,
    update_auxiliary_state!,
    update_auxiliary_state_gradient!,
    vars_integrals,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    indefinite_stack_integral!,
    vars_reverse_integrals,
    reverse_indefinite_stack_integral!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import ClimateMachine.Ocean.HydrostaticBoussinesq: diffusivity_tensor
import ClimateMachine.Ocean.HydrostaticBoussinesq: ocean_boundary_state!
import ClimateMachine.Ocean.HydrostaticBoussinesq: boundary_state!



include("mymodel-repl.jl")
include("mylhbm-repl.jl")

	function config_simple_box(FT, N, resolution, dimensions; BC = nothing)
	    if BC == nothing
	        problem = OceanGyre{FT}(dimensions...)
	    else
	        problem = OceanGyre{FT}(dimensions...; BC = BC)
	    end
	
	    _grav::FT = grav(param_set)
	    cʰ = sqrt(_grav * problem.H) # m/s
	    model = HydrostaticBoussinesqModel{FT}(param_set, problem, cʰ = cʰ)
	
	    config = ClimateMachine.OceanBoxGCMConfiguration(
	        "ocean_gyre",
	        N,
	        resolution,
	        param_set,
	        model,
	    )
	
	    return config
	end

	include("test/Ocean/refvals/test_ocean_gyre_refvals.jl")
	nt = 1
#==
	boundary_conditions = [
	    (
	        OceanBC(Impenetrable(NoSlip()), Insulating()),
	        OceanBC(Impenetrable(NoSlip()), Insulating()),
	        OceanBC(Penetrable(KinematicStress()), TemperatureFlux()),
	    ),
	]
==#
	boundary_conditions = [
	    (
	        OceanBC(Impenetrable(NoSlip()), Insulating()),
	        OceanBC(Impenetrable(NoSlip()), Insulating()),
	        OceanBC(Impenetrable(NoSlip()), Insulating()),
	    ),
	]
	BC=boundary_conditions[1]
	Δt=60
	nt=0
	FT = Float64

	# DG polynomial order
	N = Int(4)
        Np1 = N+1
	# Domain resolution and size
	Nˣ = Int(4)
	Nʸ = Int(4)
	Nᶻ = Int(5)
	resolution = (Nˣ, Nʸ, Nᶻ)
	Lˣ = 4e6    # m
	Lʸ = 4e6    # m
	H = 1000   # m
	dimensions = (Lˣ, Lʸ, H)
	timestart = FT(0)    # s
	timeend = FT(36000) # s
	solver_type =
	    ClimateMachine.IMEXSolverType(implicit_model = LinearHBModel)
	Courant_number = 0.1
	driver_config = config_simple_box(FT, N, resolution, dimensions; BC = BC)
	grid = driver_config.grid
	vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
	exp_filter = ExponentialFilter(grid, 1, 8)
	modeldata = (vert_filter = vert_filter, exp_filter = exp_filter)
	solver_config = ClimateMachine.SolverConfiguration(
	     timestart,
	     timeend,
	     driver_config,
	     init_on_cpu = true,
	     ode_solver_type = solver_type,
	     ode_dt = FT(Δt),
	     modeldata = modeldata,
	     Courant_number = Courant_number,
	 )

         Qin=solver_config.Q;
         dQout=deepcopy(Qin);

         # Calc d/dt for one step using LinearHBM
         solver_config.solver.rhs_implicit!(dQout,Qin,nothing,0);

         using Plots
         zc=reshape(solver_config.solver.rhs!.grid.vgeo[:,15,:],(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]

         # QP=Qin;
         QP=Qin;
         tz=reshape(QP.θ,(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
         savefig( scatter( tz, zc ), "fooQin.png" )
         QP=dQout;
         tz=reshape(QP.θ,(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
         uz=reshape(QP.u[:,1,:],(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
         vz=reshape(QP.u[:,2,:],(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
         savefig( scatter( tz, zc ), "fooLHBM.png" )

         # Try setting up an IVDC version
         bl=solver_config.solver.rhs!;
         using ClimateMachine.DGMethods: DGModel, init_ode_state
         ivdc_model=IVDCModel{Float64}(nothing, nothing)
         ivdc_dg=DGModel(ivdc_model,bl.grid,bl.numerical_flux_first_order,bl.numerical_flux_second_order,bl.numerical_flux_gradient; 
                         direction=ClimateMachine.Mesh.Grids.VerticalDirection )
         dQout2=deepcopy(Qin);
         ivdc_dg(dQout2,Qin,nothing,0);
         QP=dQout2;
         tz=reshape(QP.θ,(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
         savefig( scatter( tz, zc ), "fooIVDC.png" )

         mylhb_model=myLHBModel(solver_config.solver.rhs!.balance_law)
         mylbh_dg=DGModel(mylhb_model,bl.grid,bl.numerical_flux_first_order,bl.numerical_flux_second_order,bl.numerical_flux_gradient; direction=ClimateMachine.Mesh.Grids.VerticalDirection )
         dQout3=deepcopy(Qin);
         mylbh_dg(dQout3,Qin,nothing,0);
         QP=dQout3;
         tz=reshape(QP.θ,(Np1*Np1,Np1,Nᶻ,Nˣ*Nʸ) )[1,:,:,1]
         savefig( scatter( tz, zc ), "fooMYLHBM.png" )


