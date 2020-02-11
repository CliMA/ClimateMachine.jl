"""
    CLIMA driver configurations

Use to run CLIMA using the CLIMA driver's `CLIMA.invoke()`. User-customized
configurations can use these as templates.
"""

using MPI

using ..AdditiveRungeKuttaMethod
using ..Atmos
using ..DGmethods
using ..DGmethods.NumericalFluxes
using ..LowStorageRungeKuttaMethod
using ..Mesh.Topologies
using ..Mesh.Grids
using ..ODESolvers
using ..PlanetParameters

abstract type AbstractSolverType end
struct ExplicitSolverType <: AbstractSolverType
    solver_method::Function
    ExplicitSolverType(;solver_method=LSRK54CarpenterKennedy) = new(solver_method)
end
struct IMEXSolverType <: AbstractSolverType
    linear_model::Type
    linear_solver::Type
    solver_method::Function
    function IMEXSolverType(;linear_model=AtmosAcousticGravityLinearModel,
                            linear_solver=ManyColumnLU,
                            solver_method=ARK2GiraldoKellyConstantinescu)
        new(linear_model, linear_solver, solver_method)
    end
end
DefaultSolverType = IMEXSolverType

struct DriverConfiguration{FT}
    name::String
    N::Int
    array_type
    solver_type::AbstractSolverType
    #
    # AtmosModel details
    bl::BalanceLaw
    #
    # execution details
    mpicomm::MPI.Comm
    #
    # mesh details
    grid::DiscontinuousSpectralElementGrid
    #
    # DGModel details
    numfluxnondiff::NumericalFluxNonDiffusive
    numfluxdiff::NumericalFluxDiffusive
    gradnumflux::NumericalFluxGradient

    function DriverConfiguration(name::String, N::Int, FT, array_type,
                                 solver_type::AbstractSolverType,
                                 bl::BalanceLaw,
                                 mpicomm::MPI.Comm,
                                 grid::DiscontinuousSpectralElementGrid,
                                 numfluxnondiff::NumericalFluxNonDiffusive,
                                 numfluxdiff::NumericalFluxDiffusive,
                                 gradnumflux::NumericalFluxGradient)
        new{FT}(name, N, array_type, solver_type, bl, mpicomm, grid, numfluxnondiff, numfluxdiff, gradnumflux)
    end
end

function LES_Configuration(name::String,
                           N::Int,
                           (Δx, Δy, Δz)::NTuple{3,FT},
                           xmax::Int, ymax::Int, zmax::Int,
                           init_LES!;
                           xmin           = 0,
                           ymin           = 0,
                           zmin           = 0,
                           array_type     = CLIMA.array_type(),
                           solver_type    = IMEXSolverType(linear_solver=SingleColumnLU),
                           model          = AtmosModel{FT}(AtmosLESConfiguration;
                                                           init_state=init_LES!),
                           mpicomm        = MPI.COMM_WORLD,
                           boundary       = ((0,0), (0,0), (1,2)),
                           periodicity    = (true, true, false),
                           meshwarp       = (x...)->identity(x),
                           numfluxnondiff = Rusanov(),
                           numfluxdiff    = CentralNumericalFluxDiffusive(),
                           gradnumflux    = CentralNumericalFluxGradient()
                          ) where {FT<:AbstractFloat}

    brickrange = (grid1d(xmin, xmax, elemsize=Δx*N),
                  grid1d(ymin, ymax, elemsize=Δy*N),
                  grid1d(zmin, zmax, elemsize=Δz*N))
    topology = StackedBrickTopology(mpicomm, brickrange,
                                    periodicity=periodicity,
                                    boundary=boundary)

    grid = DiscontinuousSpectralElementGrid(topology,
                                            FloatType=FT,
                                            DeviceArray=array_type,
                                            polynomialorder=N,
                                            meshwarp=meshwarp)

    return DriverConfiguration(name, N, FT, array_type, solver_type, model, mpicomm, grid,
                               numfluxnondiff, numfluxdiff, gradnumflux)
end

function GCM_Configuration(name::String,
                           N::Int,
                           (nelem_horz, nelem_vert)::NTuple{2,Int},
                           domain_height::FT,
                           init_GCM!;
                           array_type         = CLIMA.array_type(),
                           solver_type        = DefaultSolverType(),
                           model              = AtmosModel{FT}(AtmosGCMConfiguration;
                                                                init_state=init_GCM!),
                           mpicomm            = MPI.COMM_WORLD,
                           meshwarp::Function = cubedshellwarp,
                           numfluxnondiff     = Rusanov(),
                           numfluxdiff        = CentralNumericalFluxDiffusive(),
                           gradnumflux        = CentralNumericalFluxGradient()
                          ) where {FT<:AbstractFloat}

    vert_range = grid1d(FT(planet_radius), FT(planet_radius+domain_height), nelem=nelem_vert)

    topology = StackedCubedSphereTopology(mpicomm, nelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(topology,
                                            FloatType=FT,
                                            DeviceArray=array_type,
                                            polynomialorder=N,
                                            meshwarp=meshwarp)

    return DriverConfiguration(name, N, FT, array_type, solver_type, model, mpicomm, grid,
                               numfluxnondiff, numfluxdiff, gradnumflux)
end

