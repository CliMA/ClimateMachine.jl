using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Topography
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

#
# USER DEFINE REGION NAME BASED ON THE GRID FILE TO BE DOWNLOADED
#
region_name = "monterey"

joinpath(@__DIR__,

mkpath(joinpath(@__DIR__, "test/DGmethods")
header_file_in   = string(region_name, ".hdr")
header_file_path = string("./TopographyFiles/", header_file_in);
body_file_in     = string(region_name, ".xyz")
body_file_path   = string("./TopographyFiles/", body_file_in);
if !isfile(header_file_path)
    mypath = string("https://web.njit.edu/~smarras/TopographyFiles/NOAA/", region_name, ".hdr");
    Base.run(`wget $mypath`)
    Base.run(`mv $header_file_in $header_file_path`);
end
if !isfile(body_file_path)
    mypath = string("https://web.njit.edu/~smarras/TopographyFiles/NOAA/", region_name, ".xyz");
    Base.run(`wget $mypath`)
    Base.run(`mv $body_file_in $body_file_path`);
end

(nlon, nlat, lonmin, lonmax, latmin, latmax, dlon, dlat) = ReadExternalHeader(header_file_in)
(xTopo, yTopo, zTopo)                                    = ReadExternalTxtCoordinates(body_file_in, "topo", nlon, nlat)
TopoSpline                                               = Spline2D(xTopo, yTopo, zTopo)

#
# Set Δx < 0 and define  Nex, Ney, Nez:
#
(Nex, Ney, Nez) = (nlon-1, nlat-1, 1)
Ne = (Nex, Ney, Nez)
if lonmin < 0
    lonminaux = lonmin - lonmin
    lonmaxaux = lonmax - lonmin
    lonmin, lonmax = lonminaux, lonmaxaux
end

if latmin < 0
    latminaux = latmin - latmin
    latmaxaux = latmax - latmin
    latmin, latmax = latminaux, latmaxaux
end


# Physical domain extents 
const (xmin, xmax) = (lonmin, lonmax) #(lonmin - lonmin, lonmax - lonmin)
const (ymin, ymax) = (latmin, latmax)
const (zmin, zmax) = (0, 10000)
Lx = abs(xmax - xmin)
Ly = abs(ymax - ymin)
Lz = abs(zmax - zmin)

Δx = Lx / ((Nex * Npoly) + 1)
Δy = Ly / ((Ney * Npoly) + 1)
Δz = Lz / ((Nez * Npoly) + 1)

@info @sprintf """ ------------------------------- """
@info @sprintf """ External grid parameters """
@info @sprintf """ Nex %d""" nlon-1
@info @sprintf """ Ney %d""" nlat-1
@info @sprintf """ Nez %d""" Nez
@info @sprintf """ xmin-max %.16e %.16e""" xmin xmax
@info @sprintf """ ymin-max %.16e %.16e""" ymin ymax
@info @sprintf """ ------------------------------- """
@info @sprintf """ Grids.jl: Importing topography file to CLIMA ... DONE"""

#
# Warp topography read from file
#
warp_external_topography(xin, yin, zin) = warp_external_topography(xin, yin, zin; SplineFunction=TopoSpline)
function warp_external_topography(xin, yin, zin; SplineFunction=TopoSpline)
    """
       Given the input set of spatial coordinates based on the DG transform
       Interpolate using the 2D spline to get the mesh warp on the entire grid,
       pointwise. 
    """
    x     = xin
    y     = yin
    z     = zin
    zdiff = TopoSpline(x, y) * (zmax - zin)/zmax
    x, y, z + zdiff
end
#}}}



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
                                            polynomialorder = N,
                                            meshwarp = warpfun,
                                            )
    
    model = AtmosModel(ConstantViscosityWithDivergence(DFloat(μ_exact)),DryModel(),NoRadiation(),
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
    
    brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                  range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)),
                  range(DFloat(zmin), length=Ne[3]+1, DFloat(zmax)))
    
    topl = BrickTopology(mpicomm, brickrange,
                         periodicity = (false, false, false))
    dt = 0.00001
    warpfun = warp_external_topography
   
    timeend = dt
    nsteps = ceil(Int64, timeend / dt)
    
    @info (ArrayType, DFloat, dim)
    result[l] = run(mpicomm, ArrayType, dim, topl, warpfun,
                    polynomialorder, timeend, DFloat, dt)
    
    
end

end
#nothing
