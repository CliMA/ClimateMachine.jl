# ClimateMachine driver configurations
#
# Contains helper functions to establish simulation configurations to be
# used with the ClimateMachine driver. Currently:
# - AtmosLESConfiguration
# - AtmosGCMConfiguration
# - OceanBoxGCMConfiguration
#
# User-customized configurations can use these as templates.

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius

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
    ClimateMachine.DriverConfiguration

Collects all parameters necessary to set up a ClimateMachine simulation.
"""
struct DriverConfiguration{FT}
    config_type::ClimateMachineConfigType

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
    numerical_flux_first_order::NumericalFluxFirstOrder
    numerical_flux_second_order::NumericalFluxSecondOrder
    numerical_flux_gradient::NumericalFluxGradient
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
        numerical_flux_first_order::NumericalFluxFirstOrder,
        numerical_flux_second_order::NumericalFluxSecondOrder,
        numerical_flux_gradient::NumericalFluxGradient,
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
            numerical_flux_first_order,
            numerical_flux_second_order,
            numerical_flux_gradient,
            config_info,
        )
    end
end

function print_model_info(model)
    msg = "AtmosModel composition\n"
    for key in fieldnames(typeof(model))
        msg =
            msg * @sprintf(
                "    %s = %s\n",
                string(key),
                string((getproperty(model, key)))
            )
    end
    @info msg
end

function AtmosLESConfiguration(
    name::String,
    N::Int,
    (Δx, Δy, Δz)::NTuple{3, FT},
    xmax::FT,
    ymax::FT,
    zmax::FT,
    param_set::AbstractParameterSet,
    init_LES!;
    xmin = zero(FT),
    ymin = zero(FT),
    zmin = zero(FT),
    array_type = ClimateMachine.array_type(),
    solver_type = IMEXSolverType(linear_solver = SingleColumnLU),
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_conservative = init_LES!,
    ),
    mpicomm = MPI.COMM_WORLD,
    boundary = ((0, 0), (0, 0), (1, 2)),
    periodicity = (true, true, false),
    meshwarp = (x...) -> identity(x),
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
) where {FT <: AbstractFloat}

    print_model_info(model)

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

    @info @sprintf(
        """
Establishing Atmos LES configuration for %s
    precision        = %s
    polynomial order = %d
    domain           = %.2f m x%.2f m x%.2f m
    resolution       = %dx%dx%d
    MPI ranks        = %d
    min(Δ_horz)      = %.2f m
    min(Δ_vert)      = %.2f m""",
        name,
        FT,
        N,
        xmax,
        ymax,
        zmax,
        Δx,
        Δy,
        Δz,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
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
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        AtmosLESSpecificInfo(),
    )
end

function AtmosGCMConfiguration(
    name::String,
    N::Int,
    (nelem_horz, nelem_vert)::NTuple{2, Int},
    domain_height::FT,
    param_set::AbstractParameterSet,
    init_GCM!;
    array_type = ClimateMachine.array_type(),
    solver_type = DefaultSolverType(),
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_conservative = init_GCM!,
    ),
    mpicomm = MPI.COMM_WORLD,
    meshwarp::Function = cubedshellwarp,
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
) where {FT <: AbstractFloat}

    print_model_info(model)

    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + domain_height),
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

    @info @sprintf(
        """
Establishing Atmos GCM configuration for %s
    precision        = %s
    polynomial order = %d
    #horiz elems     = %d
    #vert elems      = %d
    domain height    = %.2e m
    MPI ranks        = %d
    min(Δ_horz)      = %.2f m
    min(Δ_vert)      = %.2f m""",
        name,
        FT,
        N,
        nelem_horz,
        nelem_vert,
        domain_height,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
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
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        AtmosGCMSpecificInfo(domain_height, nelem_vert, nelem_horz),
    )
end

function OceanBoxGCMConfiguration(
    name::String,
    N::Int,
    (Nˣ, Nʸ, Nᶻ)::NTuple{3, Int},
    model::HydrostaticBoussinesqModel;
    FT = Float64,
    array_type = ClimateMachine.array_type(),
    solver_type = ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    ),
    mpicomm = MPI.COMM_WORLD,
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
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
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        OceanBoxGCMSpecificInfo(),
    )
end
