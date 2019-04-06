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
#        SOUNDING operations:
# read_sound()
# interpolate_sounding()
# Interpolate the sounding along the FIRST column of the grid.

function read_sounding()
    #read in the original squal sounding
    fsounding  = open(joinpath(@__DIR__, "./soundings/sounding_GC1991.dat"))
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
function kurowski_bubble(x...; initial_sounding::Array, ntrace=0, nmoist=0, dim=2)
    DFloat = eltype(x)
    
    p0::DFloat      = 85000.0 
    gravity::DFloat = grav
    q_tot::DFloat   = 0.0

    (nzmax, ~) = size(initial_sounding)

    q_tot        = 0.014*exp(-x[dim]/500)
    
    R_gas        = gas_constant_air(q_tot, 0.0, 0.0)
    c_p          = cp_m(q_tot, 0.0, 0.0)
    c_v          = cv_m(q_tot, 0.0, 0.0)
    cpoverR      = c_p/R_gas
    cvoverR      = c_v/R_gas
    Rovercp      = R_gas/c_p
    cvovercp     = c_v/c_p

    
    pi0    =     1.0
    theta0 =   283.0
    p0     = 85000.0
    rho0   = p0/(R_gas * theta0) * (pi0)^cvoverR

    #20% of relative humidity in the background
    RH0    = 20.0                                 
    # find the matching height

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

    θ = thetav
    π = datapi
    ρ = datarho
    P = datap
    #T = π * θ
    T = P / (ρ * R_gas)

    
    # Grabowski:
    es      = 611.0*exp(2.52e6/461.0*(1.0/273.16 - 1.0/T))
    qvs     = 287.04/461.0 * es/(P - es)   # saturation mixing ratio

    # Get qv from RH:
    qv_k    = qvs * RH0/100.0

    # Perturbation    
    thetac = 2.0
    dtheta = 0.0
    dqr    = 0.0
    dqc    = 0.0
    dqv    = 0.0
    dRH    = 0.0
    sigma  = 6.0
    
    rc     =  300.0
    r      = sqrt( (x[1] - 800)^2 + (x[dim] - 800.0)^2 )
    dtheta = thetac*exp(-(r/rc)^sigma)    
    dRH    = 80.0 * exp(-(r/rc)^sigma)
    dqv    = dRH * qvs/100.0
    
    q_tot  = qv_k + dqv
    u      = 0
    v      = 0
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

    (ρ=ρ, U=U, V=V, W=W, E=E, Qmoist=(ρ * q_tot, 0.0, 0.0)) 


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
    vgeo = grid.vgeo
    initial_sounding       = interpolate_sounding(dim, N, Ne, vgeo, nmoist, ntrace)
    initialcondition(x...) = kurowski_bubble(x...;
                                             initial_sounding=initial_sounding, 
                                             ntrace=ntrace,
                                             nmoist=nmoist,
                                             dim=dim)
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
    Ne = (10, 10, 10)
    N = 4
    dim = 2
    timeend = 100
    viscosity = 75.0
    
    DFloat = Float64
    for ArrayType in (HAVE_CUDA ? (CuArray, Array) : (Array,))
        brickrange = ntuple(j->range(DFloat(0); length=Ne[j]+1, stop=1000), dim)
        main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, Ne, timeend)
    end
end
end

isinteractive() || MPI.Finalize()
