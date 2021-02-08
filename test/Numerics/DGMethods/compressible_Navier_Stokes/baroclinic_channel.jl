using Test
using Dates
using LinearAlgebra
using MPI
using Printf
using StaticArrays
using UnPack

using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.GenericCallbacks
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.Orientations
using ClimateMachine.VariableTemplates
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VTK

import ClimateMachine.BalanceLaws: source

using CLIMAParameters
using CLIMAParameters.Planet: e_int_v0, grav, day, cp_d, cv_d, R_d, grav, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMAParameters
# Assume zero reference temperature
CLIMAParameters.Planet.T_0(::EarthParameterSet) = 0

"""
    baroclinic_instability_cube(...)
Initialisation helper for baroclinic-wave (channel flow) test case for iterative
determination of η = p/pₛ coordinate for given z-altitude. 
"""
function baroclinic_instability_cube!(eta, temp, tolerance, (x,y,z),f0,beta0,u0,T0,gamma_lapse,gravity, R_gas, cp)
  for niter = 1:200
    FT    = eltype(y)
    b     = FT(2)
    Ly    = FT(6e6)  
    y0    = FT(Ly/2)
    b2    = b*b
    #Get Mean Temperature
    exp1  = R_gas*gamma_lapse/gravity
    Tmean = T0*eta^exp1
    phimean = T0*gravity/gamma_lapse * (FT(1) - eta^exp1)
    logeta = log(eta)
    fac1   = (f0-beta0*y0)*(y - FT(1/2)*Ly - Ly/2π * sin(2π*y/Ly))  
    fac2   = FT(1/2)*beta0*(y^2 - Ly*y/π*sin(2π*y/Ly) - 
                          FT(1/2)*(Ly/π)^2*cos(2π*y/Ly) - 
                          Ly^2/3 - FT(1/2)*(Ly/π)^2)
    fac3 = exp(-logeta*logeta/b2)
    fac4 = exp(-logeta/b) 
    ## fac4 applies a correction based on the required temp profile from Ullrich's paper
    ## TODO: Verify if paper contains TYPO, use fac3 or fac4
    ## Check for consistency 
    phi_prime=FT(1/2)*u0*(fac1 + fac2)
    geo_phi = phimean + phi_prime*fac3*logeta
    temp = Tmean + phi_prime/R_gas*fac3*(2/b2*logeta*logeta - 1)
    num  = -gravity*z + geo_phi
    den  = -R_gas/(eta)*temp
    deta = num/den
    eta  = eta - deta
    if (abs(deta) <= FT(tolerance))
      break
    elseif (abs(deta) > FT(tolerance)) && niter==200
      #@error "Initialisation: η convergence failure."
      break
    end
  end
  return (eta, temp)
end 

function init_baroclinicwave!(problem, bl, state, aux, localgeo, t)

  (x,y,z) = localgeo.coord
  ### Problem float-type
  FT = eltype(state)
  param_set = bl.param_set
  ### Unpack CLIMAParameters
  _planet_radius = FT(planet_radius(param_set))
  gravity        = FT(grav(param_set))
  cp             = FT(cp_d(param_set))
  R_gas          = FT(R_d(param_set))
  ### Global Variables
  up    = FT(1)                ## See paper: Perturbation peak value
  Lp    = FT(6e5)              ## Perturbation parameter (radius)
  Lp2   = Lp*Lp              
  xc    = FT(2e6)              ## Streamwise center of perturbation
  yc    = FT(2.5e6)            ## Spanwise center of perturbation
  gamma_lapse = FT(5/1000)     ## Γ Lapse Rate
  Ω     = FT(7.292e-5)         ## Rotation rate [rad/s]
  f0    = 2Ω/sqrt(2)           ## 
  beta0 = f0/_planet_radius    ##  
  beta0 = -zero(FT)
  b     = FT(2)
  b2    = b*b
  u0    = FT(35)
  Ly    = FT(6e6)
  T0    = FT(288)
  T_ref = T0                   
  x0    = FT(2e7)
  p00   = FT(1e5)              ## Surface pressure

  ## Step 1: Get current coordinate value by unpacking nodal coordinates from aux state
  eta = FT(1e-7)
  temp = FT(300)
  ## Step 2: Define functions for initial condition temperature and geopotential distributions
  ## These are written in terms of the pressure coordinate η = p/pₛ

  ### Unpack initial conditions (solved by iterating for η)
  tolerance = FT(1e-10)
  eta, temp = baroclinic_instability_cube!(eta, 
                                           temp, 
                                           tolerance, 
                                           (x,y,z),
                                           f0, 
                                           beta0, 
                                           u0, 
                                           T0, 
                                           gamma_lapse, 
                                           gravity, 
                                           R_gas, 
                                           cp)
  eta = min(eta,FT(1))
  eta = max(eta,FT(0))
  ### η = p/p_s
  logeta = log(eta)
  T=FT(temp)
  press = p00*eta
  theta = T *(p00/press)^(R_gas/cp)
  rho = press/(R_gas*T)
  thetaref = T_ref * (1 - gamma_lapse*z/T0)^(-gravity/(cp*gamma_lapse))
  rhoref = p00/(T0*R_gas) * (1 - gamma_lapse*z/T0)^(gravity/(R_gas*gamma_lapse) - 1)

  ### Balanced Flow
  u = -u0*(sinpi(y/Ly))^2  * logeta * exp(-logeta*logeta/b2)

  ### Perturbation of the balanced flow
  rc2 = (x-xc)^2 + (y-yc)^2
  du = up*exp(-rc2/Lp2)
    
  ### Primitive variables
  u⃗ = SVector{3,FT}(u+du,0,0)
  e_kin = FT(1/2)*sum(abs2.(u⃗))
  e_pot = gravity * z

  ### Assign state variables for initial condition
  state.ρ = rho
  state.ρu = rho .* u⃗
  state.energy.ρe = rho * total_energy(param_set, e_kin, e_pot, T)
end

function test_run(mpicomm, ArrayType, topl, N, FT, brickrange)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

        problem = AtmosProblem(
            boundaryconditions = (InitStateBC(),),
            init_state_prognostic = init_baroclinicwave!,
        )
        model = AtmosModel{FT}(
            AtmosLESConfigType,
            param_set;
            problem = problem,
            orientation = FlatOrientation(),
            ref_state = NoReferenceState(),
            #turbulence = ConstantKinematicViscosity(FT(sqrt(1e14)), WithDivergence()),
            turbulence = ConstantKinematicViscosity(FT(100), WithDivergence()),
            moisture = DryModel(),
            source = (Gravity(),),
        )
    show_tendencies(model)

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )
    
    timeend = FT(86400 * 15)
    elementsize = minimum(step.(brickrange))
    dt =
        elementsize / FT(330) / N^2
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0); init_on_cpu = true)

    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            @info @sprintf(
                """Update
                simtime = %.16e
                runtime = %s
                norm(Q) = %.16e""",
                ODESolvers.gettime(lsrk),
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end
    
    function do_output(
        mpicomm,
        vtkdir,
        vtkstep,
        dg,
        Q,
        Qe,
        model,
        testname = "baroclinic_channel",
    )
        ## name of the file that this MPI rank will write
        filename = @sprintf(
            "%s/%s_mpirank%04d_step%04d",
            vtkdir,
            testname,
            MPI.Comm_rank(mpicomm),
            vtkstep
        )

        statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
        exactnames = statenames .* "_exact"

        writevtk(filename, Q, dg, statenames, Qe, exactnames)

        ## Generate the pvtu file for these vtk files
        if MPI.Comm_rank(mpicomm) == 0
            ## name of the pvtu file
            pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)
            ## name of each of the ranks vtk files
            prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
                @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
            end
            writepvtu(
                pvtuprefix,
                prefixes,
                (statenames..., exactnames...),
                eltype(Q),
            )
            @info "Done writing VTK: $pvtuprefix"
        end
    end

        # create vtk dir
        vtkdir =
            "vtk_baroclinicwave" *
            "_poly$(N)_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

        # setup the output callback
        outputtime = timeend
        cbvtk = EveryXSimulationSteps(50.0/ dt) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(lsrk), setup)
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
        end

    solve!(Q, lsrk; timeend = timeend, callbacks = (cbinfo, cbvtk))

    # Print some end of the simulation information
    engf = norm(Q)
    Qe = init_ode_state(dg, FT(timeend))

    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ engf engf / eng0 engf - eng0 errf errf / engfe
    errf
end

let
    ClimateMachine.init(parse_clargs=true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    # DG polynomial order
    N = 5
    FT = Float64
    # Domain resolution and size
    Δx = FT(200e3) 
    Δy = FT(120e3)
    Δz = FT(1e3)
    # Prescribe domain parameters
    xmax = FT(27000e3) 
    ymax = FT(6000e3)
    zmax = FT(30e3)

    Nex = cld(xmax , (Δx * (N+1)))
    Ney = cld(ymax , (Δy * (N+1)))
    Nez = cld(zmax , (Δz * (N+1)))


    resolution = (Δx, Δy, Δz)
    @testset "mms_bc_atmos" begin
                Ne = [23,9,5]
                brickrange = (
                    range(FT(0); length = Ne[1] + 1, stop = xmax),
                    range(FT(0); length = Ne[2] + 1, stop = ymax),
                    range(FT(0); length = Ne[3] + 1, stop = zmax),
                )
                topl = BrickTopology(
                    mpicomm,
                    brickrange,
                    periodicity = (false, false, false),
                )

            result = test_run(
                mpicomm,
                ArrayType,
                topl,
                N,
                FT,
                brickrange
            )
    end
end
