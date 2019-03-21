using MPI

using CLIMAAtmosDycore.Topologies
using CLIMAAtmosDycore.Grids
using CLIMAAtmosDycore.VanillaAtmosDiscretizations
using CLIMAAtmosDycore.AtmosStateArrays
using CLIMAAtmosDycore.LSRKmethods
using CLIMAAtmosDycore.GenericCallbacks
using CLIMAAtmosDycore
using Utilities.MoistThermodynamics
using LinearAlgebra
using DelimitedFiles
using Dierckx
using Printf

const HAVE_CUDA = try
  using CUDAdrv
  using CUDAnative
  true
catch
  false
end

macro hascuda(ex)
  return HAVE_CUDA ? :($(esc(ex))) : :(nothing)
end

using ParametersType
using PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0
@parameter gamma_d cp_d/cv_d "Heat capcity ratio of dry air"
@parameter gdm1 R_d/cv_d "(equivalent to gamma_d-1)"

# {{{
#        SOUNDING operations:
# read_sound()
# interpolate_sounding()
# Interpolate the sounding along the FIRST column of the grid.

function read_sounding()
    #read in the original squal sounding
    fsounding  = open(joinpath(@__DIR__, "../soundings/sounding_GC1991.dat"))
    sound_data = readdlm(fsounding)
    close(fsounding)
    (nzmax, ncols) = size(sound_data)
    if nzmax == 0
        error(" SOUNDING ERROR: The Sounding file is empty!")
    end
    return (sound_data, nzmax, ncols)
end

function interpolate_sounding(dim, N, Ne, vgeo, nmoist, ntrace)
    # !!!WARNING!!! This function can only work for sturctured grids with vertical boundaries!!!
    # !!!TO BE REWRITTEN FOR THE GENERAL CODE!!!!
    # {{{ FIXME: remove this after we've figure out how to pass through to kernel
    γ::Float64       = gamma_d
    p0::Float64      = MSLP
    R_gas::Float64   = R_d
    c_p::Float64     = cp_d
    c_v::Float64     = cv_d
    gravity::Float64 = grav
    Ne_v = Ne[dim]
    
    #Get sizes
    (sound_data, nmax, ncols) = read_sounding()
  
    #read in the original squal sounding
    (nzmax, ncols) = size(sound_data)

    Np = (N+1)^dim
    Nq = N + 1
    (~, ~, nelem) = size(vgeo)
    Ne_v = Ne[dim]

    #Reshape vgeo:
    if(dim == 2)
        _x, _z = 12, 13
        vgeo = reshape(vgeo, Nq, Nq, size(vgeo,2), nelem)      
    elseif(dim == 3)
        _x, _z = 12, 14
        vgeo= reshape(vgeo, Nq, Nq, Nq, size(vgeo,2), nelem)
    end
    
    if ncols == 6
        #height  theta  qv     u      v      press
        zinit,   tinit, qinit, uinit, vinit, pinit = sound_data[:, 1], sound_data[:, 2], sound_data[:, 3], sound_data[:, 4], sound_data[:, 5], sound_data[:, 6]
    elseif ncols == 5
        #height  theta  qv     u      v
        zinit,   tinit, qinit, uinit, vinit = sound_data[:, 1], sound_data[:, 2], sound_data[:, 3], sound_data[:, 4], sound_data[:, 5]
    end
    
    # create vector with all the z-values of the current processor
    # (avoids using column structure for better domain decomposition when no rain is used. AM)
    nz         = Ne_v*N + 1
    
    dataz      = zeros(Float64, nz)
    datat      = zeros(Float64, nz)
    dataq      = zeros(Float64, nz)
    datau      = zeros(Float64, nz)
    datav      = zeros(Float64, nz)
    datap      = zeros(Float64, nz)
    thetav     = zeros(Float64, nz)
    datapi     = zeros(Float64, nz)
    ρ      = zeros(Float64, nz)
    U      = zeros(Float64, nz)
    V      = zeros(Float64, nz)
    P      = zeros(Float64, nz)
    T      = zeros(Float64, nz)
    E      = zeros(Float64, nz)
    datarho    = zeros(Float64, nz)
    ini_data_interp = zeros(Float64, nz, 10)

    # WARNING:
    # FIXME
    # 1) REWRITE THIS TO WORK in PARALELL. NOW ONLY WORJKS IN SERIAL
    # 2) REWRITE THIS FOR 3D
    z          = vgeo[1, 1, _z, 1]
    zmax       = maximum(vgeo[:, :, _z, :])
    dataz[1]   = z
    zprev      = z
    xmin       = 1.0e-8; #Take this value from the grid if xmin != 0.0
    nzmax      = 2
    @inbounds for e = 1:nelem
        for j = 1:Nq
            x = vgeo[1, j, _x, e]
            z = vgeo[1, j, _z, e]
            if abs(x - xmin) <= 1.0e-5
                if (abs(z - zprev) > 1.0e-5 && z <= zmax+1.0e-5) #take un-repeated values
                    dataz[nzmax]   = z
                    zprev          = z                   
                    nzmax          = nzmax + 1
                end
            end       
        end
    end
    nzmax = nzmax - 1
    if(nzmax != nz)
        error(" function interpolate_sounding(): 1D INTERPOLATION: ops, something is wrong: nz is wrong!\n")
    end 
   
    #------------------------------------------------------
    # interpolate to the actual LGL points in vertical
    # dataz is given
    #------------------------------------------------------
    spl_tinit = Spline1D(zinit, tinit; k=1)
    spl_qinit = Spline1D(zinit, qinit; k=1)
    spl_uinit = Spline1D(zinit, uinit; k=1)
    spl_vinit = Spline1D(zinit, vinit; k=1)
    spl_pinit = Spline1D(zinit, pinit; k=1)
    if ncols == 5
        for k = 1:nz
            datat[k] = spl_tinit(dataz[k])
            dataq[k] = spl_qinit(dataz[k])
            datau[k] = spl_uinit(dataz[k])
            datav[k] = spl_vinit(dataz[k])
            if(dataz[k] > 14000.0)
                dataq[k] = 0.0
            end
        end
    elseif ncols == 6
        for k = 1:nz
            datat[k] = spl_tinit(dataz[k])
            dataq[k] = spl_qinit(dataz[k])
            datau[k] = spl_uinit(dataz[k])
            datav[k] = spl_vinit(dataz[k])
            datap[k] = spl_pinit(dataz[k])
            if(dataz[k] > 14000.0)
                dataq[k] = 0.0
            end
        end
    end
    
    #------------------------------------------------------
    # END interpolate to the actual LGL points in vertical
    #------------------------------------------------------

    c = cv_d/R_d
    rvapor = 461.0
    levap  = 2.5e+6
    es0    = 6.1e+2
    c2     = R_d/cp_d
    c1     = 1.0/c2
    g      = grav
    pi0    = 1.0
    theta0 = 300.5
    theta0 = 283
    p0     = 85000.0
            
    # convert qv from g/kg to g/g
    dataq = dataq.*1.0e-3

    # calculate the hydrostatically balanced exner potential and pressure
    if(ncols == 5)
        datapi[1] = 1.0
        datap[1]  = p0
    end
    thetav[1] = datat[1]*(1.0 + 0.608*dataq[1])
    
    for k = 2:nzmax
        thetav[k] = datat[k]*(1.0 + 0.608*dataq[k])
        if(ncols == 5)
            datapi[k] = datapi[k-1] - (gravity/(c_p*0.5*(thetav[k]+thetav[k-1])))*(dataz[k] - dataz[k-1])
            #Pressure is computed only if it is NOT passed in the sounding file
            datap[k] = p0*datapi[k]^(c_p/R_gas)
        end
    end

    if(ncols == 6)
        for k = 1:nzmax
            datapi[k] = (datap[k]/MSLP)^c2
        end
    end
    
    for k = 1:nzmax
        datarho[k] = datap[k]/(R_gas * datapi[k] * thetav[k])
        e          = dataq[k] * datap[k] * rvapor/(dataq[k] * rvapor + R_gas)
    end        
    
    for k = 1:nzmax
        ini_data_interp[k, 1] = dataz[k]   #z
        ini_data_interp[k, 2] = datat[k]   #theta
        ini_data_interp[k, 3] = dataq[k]   #qv
        ini_data_interp[k, 4] = datau[k]   #u
        ini_data_interp[k, 5] = datav[k]   #v
        ini_data_interp[k, 6] = datap[k]   #p
        ini_data_interp[k, 7] = datarho[k] #rho
        ini_data_interp[k, 8] = datapi[k]  #exner
        ini_data_interp[k, 9] = thetav[k]  #thetav
    end
    
    return ini_data_interp
end

# FIXME: Will these keywords args be OK?

function risingthermalbubble(x...; initial_sounding::Array, ntrace=0, nmoist=0, dim=2)
            γ::Float64       = gamma_d
            p0::Float64      = MSLP
            R_gas::Float64   = R_d
            c_p::Float64     = cp_d
            c_v::Float64     = cv_d
            gravity::Float64 = grav
            (nzmax, ~) = size(initial_sounding)
            # Moist bubble from Kurowski et al. 2013
            u0, v0 , w0 = 0.0, 0.0, 0.0
            pi0      = 1.0
            theta0   = 283.0
            p0       = 85000.0
            rho0   = p0/(R_gas * theta0) * (pi0)^(c_v/R_gas)
            #20% of relative humidity in the background
            RH0    = 20.0                                 
            # find the matching height
            maxt  = 0.0
            count = 1
            for k = 1:nzmax
                dataz = initial_sounding[k, 1]
                z     = x[dim]
                z2test = Float64(floor(100.0 * dataz))/100.0
                z1test = Float64(floor(100.0 * z))/100.0
                if ( abs(z1test - z2test) <= 0.2)
                    count=k
                    break
                end
            end

            dataz   = initial_sounding[count, 1] #z
            datat   = initial_sounding[count, 2] #theta
            dataq   = initial_sounding[count, 3] #qv
            datau   = initial_sounding[count, 4] #u
            datav   = initial_sounding[count, 5] #v
            datap   = initial_sounding[count, 6] #p
            datarho = initial_sounding[count, 7] #rho
            datapi  = initial_sounding[count, 8] #exner
            thetav  = initial_sounding[count, 9] #thetav

            theta_k = thetav
            pi_k    = datapi

            rho_k   = datarho
            press_k = datap
            tempe_k = pi_k*theta_k
           
            # Grabowski:
           es      = 611.0*exp(2.52e6/461.0*(1.0/273.16 - 1.0/tempe_k))
           qvs     = 287.04/461.0 * es/(press_k - es)                               # saturation mixing ratio

            qv_k    = qvs * RH0/100.0
              
            rc  =  300.0
            r   = sqrt( (x[1] - 800)^2 + (x[dim] - 800.0)^2 )
            
            R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
            
            thetac = 2.0
            dtheta = 0.0
            dqr    = 0.0
            dqc    = 0.0
            dqv    = 0.0
            dRH    = 0.0
            sigma  = 6.0
            dtheta = thetac*exp(-(r/rc)^sigma)
            
            if (dtheta > maxt)
                maxt = dtheta
            end
            theta_k = thetav + dtheta
            pi_k    = datapi
            rho_k   = datap/(R_gas * pi_k * theta_k)   
            tempe_k = pi_k * theta_k
            ρ = rho_k 
            T = tempe_k
            # Grabowski:
            es      = 611.0*exp(2.52e6/461.0*(1.0/273.16 - 1.0/tempe_k))
            qvs     = 287.04/461.0 * es/(press_k - es)  # saturation mixing ratio

            #formula from Joe Klemp's kessler.f:
            qv_k    = qvs * RH0/100.0

            dRH    = 80.0*exp(-(r/rc)^sigma)
            dqv    = dRH*qvs/100.0
            q_t  = qv_k + dqv
            U    = u0
            V    = v0
            W    = w0
            
            # System of equations only requires total specific humidity (q_t) to be solved for
            # Saturation adjustment used to compute q_v and q_l
            Qmoist = (q_t)
            # Internal energy is zero at the triple point temperature T_0
            E = ρ * (c_v * (T-T_0) + (u0^2 + v0^2 + w0^2) / 2 + gravity * x[dim])
            
            (ρ=ρ, U=U, V=V, W=W, E=E,
             Qmoist= ρ * Qmoist)
            # Note that this modification renders the assertion in line 458 invalid. 
            # TODO What do we do with this ?
            
            # Currently there are no tracers in this except the moisture vars
            #, Qtrace=ntuple(j->(-j*ρ), ntrace))

end

#function main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, Ne, 
#              initialcondition, timeend; gravity=true, dt=nothing, exact_timeend=true)

function main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, Ne, 
              timeend; gravity=true, dt=nothing,
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
                                         # Use gravity?
                                         gravity = gravity,
                                         # How many tracer variables
                                         ntrace=ntrace,
                                         # How many moisture variables
                                         nmoist=nmoist)


    
  #Read and interpolate external sounding
  vgeo = grid.vgeo
  initial_sounding = interpolate_sounding(dim, N, Ne, vgeo, nmoist, ntrace)
  initialcondition(x...) = risingthermalbubble(x...; initial_sounding=initial_sounding, ntrace=ntrace, nmoist=nmoist, dim=dim)
  
  # This is a actual state/function that lives on the grid
  Q = AtmosStateArray(spacedisc, initialcondition)

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
  lsrk = LSRK(getrhsfunction(spacedisc), Q; dt = dt, t0 = 0)

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
      @printf(io, "simtime =  %.16e\n", CLIMAAtmosDycore.gettime(lsrk))
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
  cbvtk = GenericCallbacks.EveryXSimulationSteps(10) do (init=false)
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

  h_Q = ArrayType == Array ? Q.Q : Array(Q.Q)
  for (j, n) = enumerate(spacedisc.moistrange)
    #@assert j*(@view h_Q[:, spacedisc.ρid, :]) ≈ (@view h_Q[:, n, :])
  end
  for (j, n) = enumerate(spacedisc.tracerange)
    @assert -j*(@view h_Q[:, spacedisc.ρid, :]) ≈ (@view h_Q[:, n, :])
  end
end

let
  MPI.Initialized() || MPI.Init()

  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
  mpicomm = MPI.COMM_WORLD

  @hascuda device!(MPI.Comm_rank(mpicomm) % length(devices()))

  dim = 2
  nmoist = 1
  # nmoist = 1 since q_t is the only moist prognostic variable
  ntrace = 0
  Ne = (10, 10, 10)
  N = 4
  timeend = 1.0

  for DFloat in (Float64, Float32)  
  for ArrayType in (HAVE_CUDA ? (CuArray, Array) : (Array,))
      brickrange = ntuple(j->range(DFloat(0); length=Ne[j]+1, stop=2400), dim)
      main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, Ne, timeend)
  end
  end
end

isinteractive() || MPI.Finalize()
