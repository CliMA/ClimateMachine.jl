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
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.Atmos: vars_state, vars_aux
using DelimitedFiles
using GPUifyLoops
using Random
using CLIMA.IOstrings
using Test

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,) 
else
  const ArrayTypes = (Array,)
end
FT=Float64
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

for ArrayType in ArrayTypes

  warpfun = (ξ1, ξ2, ξ3) -> begin
    x1 = ξ1 + (ξ1 - 1/2) * cos(2 * π * ξ2 * ξ3) / 4
    x2 = ξ2  + (ξ2 - 1/2) * cos(2 * π * ξ2 * ξ3) / 4
    x3 = ξ3 + ξ1 / 4 + ξ2^2 / 2 + sin(ξ1 * ξ2 * ξ3) + exp(sin(2π * (ξ1 * ξ2 + ξ3)))/20
    return (x1, x2, x3)
  end

  for N in 4:4
    Ne = 1
      
      brickrange = (range(FT(0); length=Ne+1, stop=1),
              range(FT(0); length=Ne+1, stop=1),
              range(FT(0); length=2, stop=1))
      topl = StackedBrickTopology(mpicomm, brickrange,
                            periodicity = (false, false, false))
      
      grid = DiscontinuousSpectralElementGrid(topl,
                                        FloatType = FT,
                                        DeviceArray = ArrayType,
                                        polynomialorder = N,
                                        meshwarp = warpfun,
                                       )
     
      N = polynomialorder(grid)
      vgeo=grid.vgeo
      Nq = N + 1
      Nqk = dimensionality(grid) == 2 ? 1 : Nq
      nrealelem = length(topl.realelems)
      host_array = Array ∈ typeof(vgeo).parameters
      localvgeo = host_array ? vgeo : Array(vgeo)
      S = zeros(Nqk)
      S1 = zeros(Nqk)
      
      for e in 1:nrealelem
        for k in 1:Nqk
          for j in 1:Nq
            for i in 1:Nq
              ijk = i + Nq * ((j-1)+ Nq * (k-1))
              S[k] += localvgeo[ijk,grid.x1id,e] * localvgeo[ijk,grid.MHid,e]
              S1[k] += localvgeo[ijk,grid.MHid,e]
            end
          end
        end
      end
      
      Stot = zeros(Nqk)
      S1tot = zeros(Nqk)
      Err = 0
      
      for k in 1:Nqk
        Stot[k] = MPI.Reduce(S[k], +, 0, MPI.COMM_WORLD)
        S1tot[k] = MPI.Reduce(S1[k], +, 0, MPI.COMM_WORLD)
        Err += (0.5 - Stot[k] / S1tot[k])^2
      end
  
      Err = sqrt(Err / Nqk)
      
      @test 2e-15 > Err
    
  end

  warpfun1 = (ξ1, ξ2, ξ3) -> begin
    x1 = sin(2 * π * ξ3)/16 + ξ1 #+ (ξ1 - 1/2) * cos(2 * π * ξ2 * ξ3) / 4
    x2 = ξ2  + (ξ2 - 1/2) * cos(2 * π * ξ2 * ξ3) / 4
    x3 = ξ3 # + ξ1 / 4 + ξ2^2 / 2 + sin(ξ1 * ξ2 * ξ3) + exp(sin(2π * (ξ1 * ξ2 + ξ3)))/20
    return (x1, x2, x3)
  end

  for N in 4:4
    Ne = 1
      brickrange1 = (range(FT(0); length=Ne+1, stop=1),
              range(FT(0); length=Ne+1, stop=1),
              range(FT(0); length=2, stop=1))
      topl1 = StackedBrickTopology(mpicomm, brickrange1,
                            periodicity = (false, false, false))

      grid1 = DiscontinuousSpectralElementGrid(topl1,
                                        FloatType = FT,
                                        DeviceArray = ArrayType,
                                        polynomialorder = N,
                                        meshwarp = warpfun1,
                                       )

      N = polynomialorder(grid1)
      vgeo1 = grid1.vgeo
      Nq = N + 1
      Nqk = dimensionality(grid1) == 2 ? 1 : Nq
      nrealelem = length(topl1.realelems)
      host_array = Array ∈ typeof(vgeo1).parameters
      localvgeo = host_array ? vgeo1 : Array(vgeo1)
      S = zeros(Nqk)
      S1 = zeros(Nqk)
      K = zeros(Nqk)
      for e in 1:nrealelem
        for k in 1:Nqk
          for j in 1:Nq
            for i in 1:Nq
              ijk = i + Nq * ((j-1)+ Nq * (k-1))
              S[k] += localvgeo[ijk,grid1.x1id,e] * localvgeo[ijk,grid1.MHid,e]
              S1[k] += localvgeo[ijk,grid1.MHid,e]
	      K[k] = localvgeo[ijk,grid1.x3id,e]
             end
          end
        end
      end

      Stot = zeros(Nqk)
      S1tot = zeros(Nqk)
      Err = 0

      for k in 1:Nqk
        Stot[k] = MPI.Reduce(S[k], +, 0, MPI.COMM_WORLD)
        S1tot[k] = MPI.Reduce(S1[k], +, 0, MPI.COMM_WORLD)
        Err += (0.5 + sin(2 * π * K[k]) / 16 - Stot[k] / S1tot[k])^2
	end

      Err = sqrt(Err / Nqk)
      @test 2e-15 > Err
    
  end
  N = 4
  base_Nhorz = 4
  base_Nvert = 2
  Rinner = 1 // 2  
  Router = 1
  for l in 1:3
      Nhorz = 2^(l-1) * base_Nhorz
      Nvert = 2^(l-1) * base_Nvert
      Rrange = grid1d(FT(Rinner), FT(Router); nelem=Nvert)
      topl2 = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
      grid2 = DiscontinuousSpectralElementGrid(topl2,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp,
                                         )
      N = polynomialorder(grid2)
      vgeo2 = grid2.vgeo
      Nq = N + 1
      Nqk = dimensionality(grid2) == 2 ? 1 : Nq
      nrealelem = length(topl2.realelems)
      host_array = Array ∈ typeof(vgeo2).parameters
      localvgeo = host_array ? vgeo2 : Array(vgeo2)
      topology = grid2.topology
      nvertelem = topology.stacksize
      nhorzelem = div(nrealelem, nvertelem)
      Surfout = 0
      Surfin = 0
      SV = 0
      maxim = 0
      minim = 0
      for e in 1:nrealelem
      #for ev in 1:nvertelem
      ##for eh in 1:nhorzelem
        #eout = nvertelem + (eh - 1) * nvertelem
        #ein = 1 + (eh - 1) * nvertelem
	#e = ev + (eh - 1) * nvertelem
	for i in 1:Nq
	  for j in 1:Nq
	    for k in 1:Nqk
	    #=if ((i == 1 || i == Nq) && (j ==1 || j==Nq))
                    n = 1/4 
                elseif (i == 1 || i == Nq || j==1 || j==Nq)
                    n = 1/2 
                else
                    n = 1  
                end=#
		n = 1
	    ijk = i + Nq * ((j-1) + Nqk * (k-1))
	    x, y, z = localvgeo[ijk,grid2.x1id,e], localvgeo[ijk,grid2.x2id,e], localvgeo[ijk,grid2.x3id,e]
	    r = hypot(x,y,z)
	    #maxim = max(r,maxim)
	    #minim = min(r,minim)
	    if abs(r -1 )< 1e-16
	    #if (k == Nqk && ev == nvertelem) || (abs(r-1)<= 5e-16)  
	      Surfout += n * localvgeo[ijk,grid2.MHid,e]
	    end
	    if abs(r - 0.5) <= 1e-16
	     #if (k == 1 && ev == 1 && abs(r - 0.5) <= 1e-17 ) 
	      Surfin += n * localvgeo[ijk,grid2.MHid,e]
	    end

	  end
	end
      end
      end
      #end
      Surfouttot = MPI.Reduce(Surfout, +, 0, MPI.COMM_WORLD)
      Surfintot = MPI.Reduce(Surfin, +, 0, MPI.COMM_WORLD)
      @info Surfout
      @info Surfin
    end
end


