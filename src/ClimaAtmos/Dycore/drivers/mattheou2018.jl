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

# FIXME: Will these keywords args be OK?
function squall_line(x...; ntrace=0, nmoist=0, dim=3)
    DFloat = eltype(x)

    p0::DFloat      = MSLP 
    gravity::DFloat = grav
    q_tot::DFloat   = 0.0

    zi = 840.0
    if (x[dim] < zi)
        q_tot = 0.009
        θ_ref = 289.0
    else
        q_tot = 0.0015
        θ_ref = 297.5 + (x[dim] - zi)^1/3
    end
    
    R_gas           = gas_constant_air(q_tot, 0.0, 0.0)
    c_p             = cp_m(q_tot, 0.0, 0.0)
    c_v             = cv_m(q_tot, 0.0, 0.0)
    cpoverR         = c_p/R_gas  
    Rovercp         = R_gas/c_p
    cvovercp        = c_v/c_p
    
    r = sqrt((x[1] - 300)^2 + (x[dim] - 250)^2)
    rc::DFloat    = 200
    θ_c::DFloat   = 1.0
    Δθ::DFloat    = 0.0
    
    if r <= rc
        Δθ = θ_c #* (1 + cos(π * r / rc)) / 2
    end
    θ = θ_ref + Δθ
    
    π_k = 1 - gravity / (c_p * θ) * x[dim]
    ρ = p0 / (R_gas * θ) * (π_k)^ (c_v / R_gas)
    P = p0 * (R_gas * (ρ * θ) / p0)^(c_p / c_v)
    
    u = zero(DFloat)
    v = zero(DFloat)
    w = zero(DFloat)
    
    #  P = p0*(1.0 - grav * x[dim]/(c_p*θ_ref))^cpoverR
    #  ρ = ((p0^Rovercp)*P^cvovercp)/(R_gas*θ_ref)
    T = P / (ρ * R_gas)

    U = ρ * u
    V = ρ * v
    W = ρ * w
    
    # Calculation of energy per unit mass
    e_kin = (u^2 + v^2 + w^2) / 2  
    e_pot = gravity * x[dim]
    e_int = MoistThermodynamics.internal_energy(T, q_tot, 0.0, 0.0)
    
    # Total energy 
    E = ρ * MoistThermodynamics.total_energy(e_kin, e_pot, T, q_tot, 0.0, 0.0)

    (ρ=ρ, U=U, V=V, W=W, E=E, Qmoist=(ρ * q_tot, 0.0, 0.0)) 
end

function main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, 
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
    Ne = (6, 24, 10)
    N = 4
    dim = 2
    timeend = 100
    viscosity = 75.0

    xmin, xmax = 0.0, 600.0
    zmin, zmax = 0.0, 2400.0
    
    DFloat = Float64
    for ArrayType in (HAVE_CUDA ? (CuArray, Array) : (Array,))
        brickrange = (range(DFloat(xmin); length=Ne[1]+1, stop=xmax),
                      range(DFloat(zmin); length=Ne[2]+1, stop=zmax))
        main(mpicomm, DFloat, ArrayType, brickrange, nmoist, ntrace, N, timeend)
    end
end
end

isinteractive() || MPI.Finalize()
