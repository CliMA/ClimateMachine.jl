using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies:
    StackedCubedSphereTopology, cubedshellwarp, grid1d
using ClimateMachine.Mesh.Grids:
    DiscontinuousSpectralElementGrid, VerticalDirection, min_node_distance
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods: ESDGModel, init_ode_state
using ClimateMachine.DGMethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics: soundspeed_air
using ClimateMachine.TemperatureProfiles: IsothermalProfile
using ClimateMachine.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cv_d
import CLIMAParameters

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

include("DryAtmos.jl")

const output_vtk = false
#CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 6.371e6 / 120.0

Base.@kwdef struct AcousticWave{FT} <: AbstractDryAtmosProblem
    domain_height::FT = 10e3
    T_ref::FT = 300
    α::FT = 3
    γ::FT = 100
    nv::Int = 1
end

function init_state_prognostic!(bl::DryAtmosModel,
                                problem::AcousticWave,
                                state, aux, localgeo, t)
    coords = localgeo.coord
    # callable to set initial conditions
    FT = eltype(state)

    λ = @inbounds atan(coords[2], coords[1])
    φ =  @inbounds asin(coords[3] / norm(coords, 2))

    _planet_radius::FT = planet_radius(param_set)
    z =  norm(coords) - _planet_radius

    β = min(FT(1), problem.α * acos(cos(φ) * cos(λ)))
    f = (1 + cos(FT(π) * β)) / 2
    g = sin(problem.nv * FT(π) * z / problem.domain_height)
    Δp = problem.γ * f * g
    p = aux.ref_state.p + Δp

    _cv_d::FT = cv_d(param_set)
    _R_d::FT = R_d(param_set)
    T = problem.T_ref
    ρ = p / (_R_d * T)
    e_pot = aux.Φ
    e_int = _cv_d * T

    state.ρ = ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = ρ * (e_int + e_pot)
    nothing
end


function main()
    ClimateMachine.init(parse_clargs=true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 5
    numelem_horz = 10
    numelem_vert = 5

    #timeend = 5 * 60 * 60
    timeend = 33 * 60 * 60 # Full simulation
    outputtime = 60


    FT = Float64
    result = run(
        mpicomm,
        polynomialorder,
        numelem_horz,
        numelem_vert,
        timeend,
        outputtime,
        ArrayType,
        FT,
    )
end

function run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
)

    problem = AcousticWave{FT}()

    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + problem.domain_height),
        nelem = numelem_vert,
    )
    topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = cubedshellwarp,
    )

    T_profile = IsothermalProfile(param_set, problem.T_ref)

    model = DryAtmosModel{FT}(SphericalOrientation(),
                              problem,
                              ref_state = DryReferenceState(T_profile),
                             )

    esdg = ESDGModel(
        model,
        grid,
        volume_numerical_flux_first_order = EntropyConservative(),
        #surface_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = RusanovNumericalFlux(),
    )

    linearmodel = DryAtmosAcousticGravityLinearModel(model)
    lineardg = DGModel(
        linearmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        state_auxiliary = esdg.state_auxiliary,
    )

    # determine the time step
    element_size = (problem.domain_height / numelem_vert)
    acoustic_speed = soundspeed_air(param_set, FT(problem.T_ref))
    #dt_factor = 445
    dt_factor = 100
    #dt_factor = 100
    dt = dt_factor * element_size / acoustic_speed / polynomialorder^2
    #dx = min_node_distance(grid)
    #cfl = 1.0
    #dt = cfl * dx / acoustic_speed

    # Adjust the time step so we exactly hit 1 hour for VTK output
    dt = 60 * 60 / ceil(60 * 60 / dt)
    nsteps = ceil(Int, timeend / dt)

    Q = init_ode_state(esdg, FT(0))

    #odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)
    linearsolver = ManyColumnLU()
    odesolver = ARK2GiraldoKellyConstantinescu(
        esdg,
        lineardg,
        LinearBackwardEulerSolver(linearsolver; isadjustable = false),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = false,
    )

    #filterorder = 18
    #filter = ExponentialFilter(grid, 0, filterorder)
    #cbfilter = EveryXSimulationSteps(1) do
    #    Filters.apply!(Q, :, grid, filter, direction = VerticalDirection())
    #    nothing
    #end

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem_horz    = %d
                      numelem_vert    = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(odesolver) runtime energy
        end
    end
    callbacks = (cbinfo,)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_esdg_acousticwave" *
            "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
            "_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(
        Q,
        odesolver;
        numberofsteps = nsteps,
        adjustfinalstep = false,
        callbacks = callbacks,
    )

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    engf
end


function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "acousticwave",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...), eltype(Q))

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
