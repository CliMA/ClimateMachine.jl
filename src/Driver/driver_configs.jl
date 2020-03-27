# CLIMA driver configurations
#
# Contains helper functions to establish simulation configurations to be
# used with the CLIMA driver. Currently:
# - AtmosLESConfiguration
# - AtmosGCMConfiguration
# - OceanBoxGCMConfiguration
#
# User-customized configurations can use these as templates.

using ..Parameters
const clima_dir = dirname(pathof(CLIMA))
include(joinpath(clima_dir, "..", "Parameters", "Parameters.jl"))
param_set = ParameterSet()

abstract type AbstractSolverType end

struct ExplicitSolverType <: AbstractSolverType
    solver_method::Function
    ExplicitSolverType(; solver_method = LSRK54CarpenterKennedy) =
        new(solver_method)
end

struct IMEXSolverType <: AbstractSolverType
    linear_model::Type
    linear_solver::Type
    solver_method::Function
    # FIXME: this is Atmos-specific
    function IMEXSolverType(;
        linear_model = AtmosAcousticGravityLinearModel,
        linear_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
    )
        return new(linear_model, linear_solver, solver_method)
    end
end

struct MultirateSolverType <: AbstractSolverType
    linear_model::Type
    solver_method::Type
    slow_method::Function
    fast_method::Function
    timestep_ratio::Int
    function MultirateSolverType(;
        linear_model = AtmosAcousticGravityLinearModel,
        solver_method = MultirateRungeKutta,
        slow_method = LSRK54CarpenterKennedy,
        fast_method = LSRK54CarpenterKennedy,
        timestep_ratio = 100,
    )
        return new(
            linear_model,
            solver_method,
            slow_method,
            fast_method,
            timestep_ratio,
        )
    end
end

DefaultSolverType = IMEXSolverType

abstract type ConfigSpecificInfo end
struct AtmosLESSpecificInfo <: ConfigSpecificInfo end
struct AtmosGCMSpecificInfo{FT} <: ConfigSpecificInfo
    domain_height::FT
    nelem_vert::Int
    nelem_horz::Int
end
struct OceanBoxGCMSpecificInfo <: ConfigSpecificInfo end

"""
    CLIMA.DriverConfiguration

Collects all parameters necessary to set up a CLIMA simulation.
"""
struct DriverConfiguration{FT}
    config_type::CLIMAConfigType

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
    #
    # configuration-specific info
    config_info::ConfigSpecificInfo

    function DriverConfiguration(
        config_type,
        name::String,
        N::Int,
        FT,
        array_type,
        solver_type::AbstractSolverType,
        bl::BalanceLaw,
        mpicomm::MPI.Comm,
        grid::DiscontinuousSpectralElementGrid,
        numfluxnondiff::NumericalFluxNonDiffusive,
        numfluxdiff::NumericalFluxDiffusive,
        gradnumflux::NumericalFluxGradient,
        config_info::ConfigSpecificInfo,
    )
        return new{FT}(
            config_type,
            name,
            N,
            array_type,
            solver_type,
            bl,
            mpicomm,
            grid,
            numfluxnondiff,
            numfluxdiff,
            gradnumflux,
            config_info,
        )
    end
end

function AtmosLESConfiguration(
    name::String,
    N::Int,
    (Δx, Δy, Δz)::NTuple{3, FT},
    xmax::FT,
    ymax::FT,
    zmax::FT,
    init_LES!;
    xmin = zero(FT),
    ymin = zero(FT),
    zmin = zero(FT),
    array_type = CLIMA.array_type(),
    solver_type = IMEXSolverType(linear_solver = SingleColumnLU),
    model = AtmosModel{FT}(
        AtmosLESConfigType;
        init_state = init_LES!,
        param_set = param_set,
    ),
    mpicomm = MPI.COMM_WORLD,
    boundary = ((0, 0), (0, 0), (1, 2)),
    periodicity = (true, true, false),
    meshwarp = (x...) -> identity(x),
    numfluxnondiff = Rusanov(),
    numfluxdiff = CentralNumericalFluxDiffusive(),
    gradnumflux = CentralNumericalFluxGradient(),
) where {FT <: AbstractFloat}

    @info @sprintf(
        """Establishing Atmos LES configuration for %s
        precision        = %s
        polynomial order = %d
        domain           = %.2fx%.2fx%.2f
        resolution       = %dx%dx%d
        MPI ranks        = %d""",
        name,
        FT,
        N,
        xmax,
        ymax,
        zmax,
        Δx,
        Δy,
        Δz,
        MPI.Comm_size(mpicomm)
    )

    brickrange = (
        grid1d(xmin, xmax, elemsize = Δx * N),
        grid1d(ymin, ymax, elemsize = Δy * N),
        grid1d(zmin, zmax, elemsize = Δz * N),
    )
    topology = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = N,
        meshwarp = meshwarp,
    )

    return DriverConfiguration(
        AtmosLESConfigType(),
        name,
        N,
        FT,
        array_type,
        solver_type,
        model,
        mpicomm,
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux,
        AtmosLESSpecificInfo(),
    )
end

function AtmosGCMConfiguration(
    name::String,
    N::Int,
    (nelem_horz, nelem_vert)::NTuple{2, Int},
    domain_height::FT,
    init_GCM!;
    array_type = CLIMA.array_type(),
    solver_type = DefaultSolverType(),
    model = AtmosModel{FT}(
        AtmosGCMConfigType;
        init_state = init_GCM!,
        param_set = param_set,
    ),
    mpicomm = MPI.COMM_WORLD,
    meshwarp::Function = cubedshellwarp,
    numfluxnondiff = Rusanov(),
    numfluxdiff = CentralNumericalFluxDiffusive(),
    gradnumflux = CentralNumericalFluxGradient(),
) where {FT <: AbstractFloat}
    @info @sprintf(
        """Establishing Atmos GCM configuration for %s
        precision        = %s
        polynomial order = %d
        #horiz elems     = %d
        #vert_elems      = %d
        domain height    = %.2e
        MPI ranks        = %d""",
        name,
        FT,
        N,
        nelem_horz,
        nelem_vert,
        domain_height,
        MPI.Comm_size(mpicomm)
    )

    vert_range = grid1d(
        FT(planet_radius),
        FT(planet_radius + domain_height),
        nelem = nelem_vert,
    )

    topology = StackedCubedSphereTopology(mpicomm, nelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = N,
        meshwarp = meshwarp,
    )

    return DriverConfiguration(
        AtmosGCMConfigType(),
        name,
        N,
        FT,
        array_type,
        solver_type,
        model,
        mpicomm,
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux,
        AtmosGCMSpecificInfo(domain_height, nelem_vert, nelem_horz),
    )
end

function OceanBoxGCMConfiguration(
    name::String,
    N::Int,
    (Nˣ, Nʸ, Nᶻ)::NTuple{3, Int},
    model::HydrostaticBoussinesqModel;
    FT = Float64,
    array_type = CLIMA.array_type(),
    solver_type = ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    ),
    mpicomm = MPI.COMM_WORLD,
    numfluxnondiff = Rusanov(),
    numfluxdiff = CentralNumericalFluxDiffusive(),
    gradnumflux = CentralNumericalFluxGradient(),
    periodicity = (false, false, false),
    boundary = ((1, 1), (1, 1), (2, 3)),
)

    brickrange = (
        range(FT(0); length = Nˣ + 1, stop = model.problem.Lˣ),
        range(FT(0); length = Nʸ + 1, stop = model.problem.Lʸ),
        range(FT(-model.problem.H); length = Nᶻ + 1, stop = 0),
    )

    topology = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = N,
    )

    return DriverConfiguration(
        OceanBoxGCMConfigType(),
        name,
        N,
        FT,
        array_type,
        solver_type,
        model,
        mpicomm,
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux,
        OceanBoxGCMSpecificInfo(),
    )
end
