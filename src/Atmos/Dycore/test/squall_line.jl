using MPI


using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.AtmosDycore.VanillaAtmosDiscretizations
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.GenericCallbacks
using CLIMA.AtmosDycore
using CLIMA.MoistThermodynamics
using LinearAlgebra
using DelimitedFiles
using Dierckx
using Printf

using CLIMA.ParametersType
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0


# {{{

function read_sounding()
    #read in the original squal sounding
    fsounding  = open(joinpath(@__DIR__, "./soundings/sounding_JCP2013_with_pressure.dat"))
    sounding = readdlm(fsounding)
    close(fsounding)
    (nzmax, ncols) = size(sounding)
    if nzmax == 0
        error("SOUNDING ERROR: The Sounding file is empty!")
    end
    return (sounding, nzmax, ncols)
end

function squall_line(x...; ntrace=0, nmoist=0, dim=3)
    dim = 2
    DFloat 	    = eltype(x)
    p0::DFloat 	    = MSLP
    
    # ----------------------------------------------------
    # GET DATA FROM INTERPOLATED ARRAY ONTO VECTORS
    # This driver accepts data in 6 column format
    # ----------------------------------------------------
    (sounding, _, ncols) = read_sounding()
    
    # WARNING: Not all sounding data is formatted/scaled 
    # the same. Care required in assigning array values
    # height theta qv    u     v     pressure
    zinit, tinit, qinit, uinit, vinit, pinit  = sounding[:, 1],
    sounding[:, 2],
    sounding[:, 3],
    sounding[:, 4],
    sounding[:, 5],
    sounding[:, 6]

    #------------------------------------------------------
    # GET SPLINE FUNCTION
    #------------------------------------------------------
    spl_tinit    = Spline1D(zinit, tinit; k=1)
    spl_qinit    = Spline1D(zinit, qinit; k=1)
    spl_uinit    = Spline1D(zinit, uinit; k=1)
    spl_vinit    = Spline1D(zinit, vinit; k=1)
    spl_pinit    = Spline1D(zinit, pinit; k=1)
    # --------------------------------------------------
    # INITIALISE ARRAYS FOR INTERPOLATED VALUES
    # --------------------------------------------------
    datat          = spl_tinit(x[dim])
    dataq          = spl_qinit(x[dim])
    datau          = spl_uinit(x[dim])
    datav          = spl_vinit(x[dim])
    datap          = spl_pinit(x[dim])
    dataq          = dataq * 1.0e-3
    
    R_gas::DFloat   = gas_constant_air(dataq, 0.0, 0.0)
    c_p::DFloat     = cp_m(dataq,0.0,0.0)
    c_v::DFloat     = cv_m(dataq,0.0,0.0)
    cvoverR         = c_v/R_gas
    gravity::DFloat = grav
    
    #TODO Driver constant parameters need references
    rvapor        = 461.0
    levap         = 2.5e6
    es0           = 611.0
    pi0           = 1.0
    p0            = MSLP
    theta0        = 300.4675
    c2            = R_gas / c_p
    c1            = 1.0 / c2
    
    # Convert dataq to kg/kg
    datapi        = (datap / MSLP) ^ (c2)                         # Exner pressure from sounding data
    thetav        = datat * (1.0 + 0.61 * dataq)                  # Liquid potential temperature

    # theta perturbation
    dtheta        = 0.0
    thetac        = 5.0
    rx            = 10000.0
    ry            =  1500.0
    r		  = sqrt( (x[1]/rx )^2 + ((x[dim] - 2000.0)/ry)^2)
    if (r <= 1.0)
        dtheta	  = thetac * (cos(0.5*π*r))^2
    end
    θ             = thetav + dtheta
    datarho       = datap / (R_gas * datapi *  θ)
    e             = dataq * datap * rvapor/(dataq * rvapor + R_gas)
    
    q_tot         = dataq
    P             = datap                                         # Assumed known from sounding
    ρ             = datarho
    T             = P / (ρ * R_gas)
    u, v, w       = 0.0, 0.0, 0.0
    U      	  = ρ * u
    V      	  = ρ * v
    W      	  = ρ * w
    
    # Calculation of energy per unit mass
    e_kin = (u^2 + v^2 + w^2) / 2  
    e_pot = gravity * x[dim]
    e_int = internal_energy(T, q_tot, 0.0, 0.0)
    # Total energy 
    E = ρ * total_energy(e_kin, e_pot, T, q_tot, 0.0, 0.0)
    (ρ=ρ, U=U, V=V, W=V, E=E, Qmoist=(ρ * q_tot, 0.0, 0.0)) 

end


function main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, Ne, 
              timeend; gravity=true, viscosity=2.5, dt=nothing,
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
    initialcondition(x...) = squall_line(x...;
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
            @printf(io, "||Q_t||infty, ||Q_l||infty  =  %.16e; %.16e\n", maximum(Q[:, 6, :]), maximum(Q[:, 7, :]))
        end
        nothing
    end

    step = [0]
    mkpath("vtk_squall")
    cbvtk = GenericCallbacks.EveryXSimulationSteps(25000) do (init=false)
        outprefix = @sprintf("vtk_squall/RTB_%dD_step%04d_mpirank%04d", dim, step[1],MPI.Comm_rank(mpicomm))
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
    
    viscosity = 75
    nmoist = 3
    ntrace = 0
    Ne = (13, 24)
    N = 4
    dim = 2
    timeend = 20000.0

    xmin =  -12000.0
    xmax =   12000.0
    zmin =       0.0
    zmax =   24000.0
    
    DFloat = Float64
    for ArrayType in (Array,)
        brickrange = (range(DFloat(xmin); length=Ne[1]+1, stop=xmax),
                      range(DFloat(zmin); length=Ne[2]+1, stop=zmax))

        main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, Ne, timeend) 
    end
end

isinteractive() || MPI.Finalize()
