using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.Vtk
using DelimitedFiles
using Dierckx

@static if haspkg("CuArrays")
    using CUDAdrv
    using CUDAnative
    using CuArrays
    CuArrays.allowscalar(false)
    const ArrayType = CuArray
else
    const ArrayType = Array
end

if !@isdefined integration_testing
    const integration_testing =
        parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end


Npoly = 4

#
# Read topography files:
#
if isfile("TopographyFiles")
    Base.run(`mkdir TopographyFiles`);
end


# Physical domain extents
const Ne = (1, 1, 20)
const (xmin, xmax) = (0, 1)
const (ymin, ymax) = (0, 1)
const (zmin, zmax) = (0, 10)
Lx = abs(xmax - xmin)
Ly = abs(ymax - ymin)
Lz = abs(zmax - zmin)

(Nex, Ney, Nez) = (Ne[1], Ne[2], Ne[3])
Δx = Lx / ((Nex * Npoly) + 1)
Δy = Ly / ((Ney * Npoly) + 1)
Δz = Lz / ((Nez * Npoly) + 1)

@info @sprintf """ ------------------------------- """
@info @sprintf """ External grid parameters """
@info @sprintf """ Nex %d""" Nex
@info @sprintf """ Ney %d""" Ney
@info @sprintf """ Nez %d""" Nez
@info @sprintf """ xmin-max %.16e %.16e""" xmin xmax
@info @sprintf """ ymin-max %.16e %.16e""" ymin ymax
@info @sprintf """ zmin-max %.16e %.16e""" zmin zmax
@info @sprintf """ ------------------------------- """
@info @sprintf """ Grids.jl: Importing topography file to CLIMA ... DONE"""

include("mms_solution_generated.jl")

function mms2_init_state!(state::Vars, aux::Vars, (x,y,z), t)
  state.ρ = ρ_g(t, x, y, z, Val(2))
  state.ρu = SVector(U_g(t, x, y, z, Val(2)),
                     V_g(t, x, y, z, Val(2)),
                     W_g(t, x, y, z, Val(2)))
  state.ρe = E_g(t, x, y, z, Val(2))
end

function mms2_source!(source::Vars, state::Vars, aux::Vars, t::Real)
  x,y,z = aux.coord.x, aux.coord.y, aux.coord.z
  source.ρ  = Sρ_g(t, x, y, z, Val(2))
  source.ρu = SVector(SU_g(t, x, y, z, Val(2)),
                      SV_g(t, x, y, z, Val(2)),
                      SW_g(t, x, y, z, Val(2)))
  source.ρe = SE_g(t, x, y, z, Val(2))
end

function mms3_init_state!(state::Vars, aux::Vars, (x,y,z), t)
  state.ρ = ρ_g(t, x, y, z, Val(3))
  state.ρu = SVector(U_g(t, x, y, z, Val(3)),
                     V_g(t, x, y, z, Val(3)),
                     W_g(t, x, y, z, Val(3)))
  state.ρe = E_g(t, x, y, z, Val(3))
end

function mms3_source!(source::Vars, state::Vars, aux::Vars, t::Real)
  x,y,z = aux.coord.x, aux.coord.y, aux.coord.z
  source.ρ  = Sρ_g(t, x, y, z, Val(3))
  source.ρu = SVector(SU_g(t, x, y, z, Val(3)),
                      SV_g(t, x, y, z, Val(3)),
                      SW_g(t, x, y, z, Val(3)))
  source.ρe = SE_g(t, x, y, z, Val(3))
end


function run(mpicomm, ArrayType, dim, topl, warpfun, N, timeend, DFloat, dt)
    
    grid = DiscontinuousSpectralElementGrid(topl,
                                            FloatType = DFloat,
                                            DeviceArray = ArrayType,
                                            polynomialorder = N)
    
    model = AtmosModel(ConstantViscosityWithDivergence(DFloat(1)),DryModel(),NoRadiation(),
                           mms3_source!, InitStateBC(), mms3_init_state!)
   
    dg = DGModel(model,
                 grid,
                 Rusanov(),
                 DefaultGradNumericalFlux())

    param = init_ode_param(dg)

    Q = init_ode_state(dg, param, DFloat(0))
    
    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
    
    mkpath("./vtk-mesh")
    step = [0]
    statenames = ("RHO", "U", "V", "W", "E")
    cbvtk = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
        outprefix = @sprintf("./vtk-mesh/stretched_grid")
        
        @debug "doing VTK output" outprefix
        writevtk(outprefix, Q, dg, statenames)
        #pvtuprefix = @sprintf("_%dD_step%04d", dim, step[1])
        #prefixes = ntuple(i->
        #                  @sprintf("./vtk-mesh/stretched_grid"), MPI.Comm_size(mpicomm))
        #writepvtu(pvtuprefix, prefixes, statenames)
        step[1] += 1
        nothing
    end
    solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbvtk, ))
end
using Test

let
    MPI.Initialized() || MPI.Init()
    mpicomm = MPI.COMM_WORLD
    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
        ll == "WARN"  ? Logging.Warn  :
        ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))
    @static if haspkg("CUDAnative")
        device!(MPI.Comm_rank(mpicomm) % length(devices()))
    end

    polynomialorder = 4
    base_num_elem = 4

    DFloat = Float64
    dim = 3

    
    #-----------------------------------------------------------------
    # build physical range to be stratched
    #-----------------------------------------------------------------
    x_range = range(xmin, length=Ne[1]   + 1, xmax)
    y_range = range(ymin, length=Ne[2]   + 1, ymax)
    z_range = range(zmin, length=Ne[3]   + 1, zmax)
    
    #-----------------------------------------------------------------
    # Build grid stretching along whichever direction
    # (ONLY Z for now. We need to decide what function we want to use for x and y)
    #-----------------------------------------------------------------
    z_range = grid_stretching_1d(zmin, zmax, Ne[end], "boundary_stretching")
    
    #-----------------------------------------------------------------
    # END grid stretching 
    #-----------------------------------------------------------------
    
    brickrange = (x_range, y_range, z_range)
    #-----------------------------------------------------------------
    #Build grid:
    #-----------------------------------------------------------------
    
    topl = BrickTopology(mpicomm, brickrange,
                         periodicity = (false, false, false))
    dt = 0.00001
    
    timeend = dt
    nsteps = ceil(Int64, timeend / dt)
    
    @info (ArrayType, DFloat, dim)
    run(mpicomm, ArrayType, dim, topl, nothing,
        polynomialorder, timeend, DFloat, dt)
   

end


#nothing
