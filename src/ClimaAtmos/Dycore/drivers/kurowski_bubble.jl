using MPI

using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.CLIMAAtmosDycore.VanillaAtmosDiscretizations
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.GenericCallbacks
using CLIMA.CLIMAAtmosDycore
using CLIMA.MoistThermodynamics
using LinearAlgebra
using DelimitedFiles
using Dierckx
using Printf

const HAVE_CUDA = try
    using CuArrays
    using CUDAdrv
    using CUDAnative
    true
catch
    false
end

macro hascuda(ex)
    return HAVE_CUDA ? :($(esc(ex))) : :(nothing)
end

using CLIMA.ParametersType
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0


# {{{

function read_sounding()
    #read in the original squal sounding
    fsounding  = open(joinpath(@__DIR__, "./soundings/sounding_GC1991.dat"))
    sounding = readdlm(fsounding)
    close(fsounding)
    (nzmax, ncols) = size(sounding)
    if nzmax == 0
       # error("SOUNDING ERROR: The Sounding file is empty!")
    end
    return (sounding, nzmax, ncols)
end

function kurowski_bubble(x...; ntrace=0, nmoist=0, dim=2, Ne, N)

  DFloat = eltype(x)
  p0::DFloat = MSLP
  c_p::DFloat = cp_d 
  c_v::DFloat = cv_d
  gravity::DFloat = grav
  q_tot::DFloat = 0

  # ----------------------------------------------------
  # GET DATA FROM INTERPOLATED ARRAY ONTO VECTORS
  # DRIVER FOR 6 COLUMN SOUNDING DATA
  # ----------------------------------------------------
  (sounding, nz_sound, ncols) = read_sounding()
  #height  theta  qv     u      v      press
  zinit,   tinit, qvinit, uinit, vinit, pinit = sounding[:, 1],sounding[:, 2],sounding[:, 3],sounding[:, 4],sounding[:, 5],sounding[:, 6]
  #------------------------------------------------------
  # GET DESCRIPTION OF INTERPOLATED SPLINE 
  #------------------------------------------------------
  spl_tinit    = Spline1D(zinit, tinit)
  spl_qvinit    = Spline1D(zinit, qvinit)
  spl_uinit    = Spline1D(zinit, uinit)
  spl_vinit    = Spline1D(zinit, vinit)
  if (ncols == 6)
      spl_pinit    = Spline1D(zinit, pinit)
  end
   
  # --------------------------------------------------
  # INITIALISE ARRAYS FOR INTERPOLATED VALUES
  # --------------------------------------------------
  
  θ = spl_tinit(x[dim])
  dataqv = spl_qvinit(x[dim])
  u = spl_uinit(x[dim])
  v = spl_vinit(x[dim])
  P = spl_pinit(x[dim])
  if(x[dim]> 14000.0)
      dataqv = 0.0
  end
  
  q_tot        = 0.014 #*exp(-x[dim]/500)
  R_gas        = gas_constant_air(q_tot, 0.0, 0.0)
  R_d          = gas_constant_air(0.0, 0.0, 0.0)
  c_p          = cp_m(q_tot, 0.0, 0.0)
  c_v          = cv_m(q_tot, 0.0, 0.0)
  cpoverR      = c_p/R_gas
  cvoverR      = c_v/R_gas
  Rovercp      = R_gas/c_p
  cvovercp     = c_v/c_p

  rvapor = 461.0
  levap  = 2.5e+6
  es0    = 6.1e+2
  pi0    =     1.0    
  theta0 =   283.0
  p0     = 85000.0
  
  # Virtual potential temperature from \theta sounding data 
  θ = θ * (1.0 + 0.608*dataqv)
  rho0   = p0/(R_gas * theta0) * (pi0)^cvoverR
  ρ = ((p0^Rovercp)*P^cvovercp)/(R_gas * θ)
  
  # convert qv from g/kg to g/g
  dataqv = dataqv.*1.0e-3
  
  # calculate the hydrostatically balanced exner potential and pressure
  P_exner = exner(P, q_tot, 0.0, 0.0)
  T = P / (ρ * R_gas)
  # Grabowski
  es      = 611.0*exp(2.52e6/461.0*(1.0/273.16 - 1.0/T))
  qvs     = 287.04/461.0 * es/(P - es)   # saturation mixing ratio
  # Get qv from RH: #20% of relative humidity in the background
  RH_0    = 20.0
  qv_k    = qvs * RH_0/100.0
  # Perturbation    
  thetac = 2.0
  dtheta = 0.0
  dqr    = 0.0
  dqc    = 0.0
  dqv    = 0.0
  dRH    = 0.0
  sigma  = 6.0
  rc     = 300.0
  r      = sqrt( (x[1] - 700)^2 + (x[dim] - 400.0)^2 )
  dtheta = thetac*exp(-(r/rc)^sigma)    
  dRH    = 80.0 * exp(-(r/rc)^sigma)
  dqv    = dRH * qvs/100.0
  q_tot  = qv_k + dqv
  u      = u
  v      = v
  w      = 0
  U      = ρ * u
  V      = ρ * v
  W      = ρ * w
  # Calculation of energy per unit mass
  e_kin = (u^2 + v^2 + w^2) / 2  
  e_pot = gravity * x[dim]
  e_int = MoistThermodynamics.internal_energy(T, q_tot, 0.0, 0.0)
  # Total energy 
  E = ρ * MoistThermodynamics.total_energy(e_kin, e_pot, T, q_tot, 0.0, 0.0)
  (ρ=ρ, U=U, V=V, W=V, E=E, Qmoist=(ρ * q_tot, 0.0, 0.0)) 

end

function main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, Ne, 
              timeend; gravity=true, viscosity=0, dt=nothing,
              exact_timeend=true) 
    dim = length(brickrange)
    topl = BrickTopology(# MPI communicator to connect elements/partition
                         mpicomm,
                         # tuple of point element edges in each dimension
                         # (dim is inferred from this)
                         brickrange,
                         periodicity=(true, ntuple(j->false, dim-1)...))

    grid = DiscontinuousSpectralElementGrid(topl,
                                            # Compute floating point type
                                            FloatType = DFloat,
                                            # This is the array type to store
                                            # data: CuArray = GPU, Array = CPU
                                            DeviceArray = ArrayType,
                                            # polynomial order for LGL grid
                                            polynomialorder = N,
                                            # how to skew the mesh degrees of
                                            # freedom (for instance spherical
                                            # or topography maps)
                                            # warp = warpgridfun
                                            )

    # spacedisc = data needed for evaluating the right-hand side function
    spacedisc = VanillaAtmosDiscretization(grid,
                                           gravity=gravity,
                                           viscosity=viscosity,
                                           ntrace=ntrace,
                                           nmoist=nmoist)

    # This is a actual state/function that lives on the grid    
    #vgeo = grid.vgeo
    #initial_sounding       = interpolate_sounding(dim, N, Ne, vgeo, nmoist, ntrace)
    initialcondition(x...) = kurowski_bubble(x...;
                                             ntrace=ntrace,
                                             nmoist=nmoist,
                                             dim=dim,
                                             Ne=Ne,
                                             N=N)
    Q = MPIStateArray(spacedisc, initialcondition)

    # Determine the time step
    (dt == nothing) && (dt = VanillaAtmosDiscretizations.estimatedt(spacedisc, Q))
    if exact_timeend
        nsteps = ceil(Int64, timeend / dt)
        dt = timeend / nsteps
    end

    # Initialize the Method (extra needed buffers created here)
    # Could also add an init here for instance if the ODE solver has some
    # state and reading from a restart file

    # TODO: Should we use get property to get the rhs function?
    lsrk = LowStorageRungeKutta(getrhsfunction(spacedisc), Q; dt = dt, t0 = 0)

    # Get the initial energy
    io = MPI.Comm_rank(mpicomm) == 0 ? stdout : open("/dev/null", "w")
    eng0 = norm(Q)
    @printf(io, "||Q||₂ (initial) =  %.16e\n", eng0)

    # Set up the information callback
    timer = [time_ns()]
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
        if s
            timer[1] = time_ns()
        else
            run_time = (time_ns() - timer[1]) * 1e-9
            (min, sec) = fldmod(run_time, 60)
            (hrs, min) = fldmod(min, 60)
            @printf(io,
                    "-------------------------------------------------------------\n")
            @printf(io, "simtime =  %.16e\n", ODESolvers.gettime(lsrk))
            @printf(io, "runtime =  %03d:%02d:%05.2f (hour:min:sec)\n", hrs, min, sec)
            @printf(io, "||Q||₂  =  %.16e\n", norm(Q))
        end
        nothing
    end

    #= Paraview calculators:
    P = (0.4) * (E  - (U^2 + V^2 + W^2) / (2*ρ) - 9.81 * ρ * coordsZ)
    theta = (100000/287.0024093890231) * (P / 100000)^(1/1.4) / ρ
    =#
    step = [0]
    mkpath("vtk")
    cbvtk = GenericCallbacks.EveryXSimulationSteps(1000) do (init=false)
        outprefix = @sprintf("vtk/RTB_%dD_step%04d", dim, step[1])
        @printf(io,
                "-------------------------------------------------------------\n")
        @printf(io, "doing VTK output =  %s\n", outprefix)
        VanillaAtmosDiscretizations.writevtk(outprefix, Q, spacedisc)
        step[1] += 1
        nothing
    end

    solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

    # Print some end of the simulation information
    engf = norm(Q)
    @printf(io, "-------------------------------------------------------------\n")
    @printf(io, "||Q||₂ ( final ) =  %.16e\n", engf)
    @printf(io, "||Q||₂ (initial) / ||Q||₂ ( final ) = %+.16e\n", engf / eng0)
    @printf(io, "||Q||₂ ( final ) - ||Q||₂ (initial) = %+.16e\n", eng0 - engf)
end

let
    MPI.Initialized() || MPI.Init()

    Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
    mpicomm = MPI.COMM_WORLD

    @hascuda device!(MPI.Comm_rank(mpicomm) % length(devices()))

    viscosity = 0.0
    nmoist = 3
    ntrace = 0
    Ne = (20, 60)
    N = 4
    dim = 2
    timeend = 100
    viscosity = 50.0

    xmin = 0.0
    xmax = 1400.0
    zmin = 0.0
    zmax = 4000.0
    
    DFloat = Float64
    for ArrayType in (HAVE_CUDA ? (CuArray, Array) : (Array,))
        brickrange = (range(DFloat(xmin); length=Ne[1]+1, stop=xmax),
                      range(DFloat(zmin); length=Ne[2]+1, stop=zmax))

        main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, Ne, timeend)
    end
end

isinteractive() || MPI.Finalize()
