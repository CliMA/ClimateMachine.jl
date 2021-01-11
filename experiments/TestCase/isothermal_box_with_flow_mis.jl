using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods: DGModel, init_ode_state, remainder_DGModel
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Geometry
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles: IsothermalProfile
using ClimateMachine.Atmos
using ClimateMachine.TurbulenceClosures
using ClimateMachine.Orientations
using ClimateMachine.BalanceLaws: Prognostic, Auxiliary, vars_state

using ClimateMachine.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius, R_d, grav, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const output_vtk = true
const gravity = false

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 2
    numelem_horz = 1
    numelem_vert = 1

    timeend = 15 * 24 * 3600
    outputtime = timeend
    outputtime = 1

    FT = Float64

    discrete_splitting = true

    test_run(
        mpicomm,
        polynomialorder,
        numelem_horz,
        numelem_vert,
        timeend,
        outputtime,
        ArrayType,
        FT,
        discrete_splitting,
    )
end

function test_run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
    discrete_splitting,
)
    setup = ZonalFlowSetup{FT}()

    _planet_radius::FT = planet_radius(param_set)
    horz_range = range(FT(0), stop = _planet_radius/10, length = numelem_horz+1)
    vert_range = range(FT(0), stop = setup.domain_height/5, length = numelem_vert+1)
    brickrange = (horz_range, horz_range, vert_range)

    if gravity
      periodicity=(true, true, false)
    else
      periodicity=(true, true, true)
    end

    topology = StackedBrickTopology(mpicomm, brickrange, periodicity=periodicity)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    zero_ref_state_velocity = true
    if zero_ref_state_velocity
      ref_state = ZonalReferenceState(setup.T0, FT(0))
    else
      ref_state = ZonalReferenceState(setup.T0, setup.u0)
    end

    if gravity
      source = (Gravity(),)
      orientation = FlatOrientation()
    else
      source = ()
      orientation = NoOrientation()
    end

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = setup,
        orientation = orientation,
        ref_state = ref_state,
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        source = source,
    )

    # determine the time step
    acoustic_speed = soundspeed_air(model.param_set, FT(setup.T0))

    dx = min_node_distance(grid, HorizontalDirection())
    dz = min_node_distance(grid, VerticalDirection())
    aspect_ratio = dx / dz

    horz_sound_cfl = FT(60 // 100)
    #horz_sound_cfl = FT(30 // 100)
    dt = horz_sound_cfl * dx / acoustic_speed
    horz_adv_cfl = dt * setup.u0 / dx

    @show dt
    @show aspect_ratio
    @show horz_sound_cfl
    @show horz_adv_cfl

    nsteps = ceil(Int, timeend / dt)

    if gravity
      linearmodel = AtmosAcousticGravityLinearModel(model)
    else
      linearmodel = AtmosAcousticLinearModel(model)
    end

    dg = DGModel(
        model,
        grid,
#       RusanovNumericalFlux(),
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    fast_dg = DGModel(
        linearmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
#       direction = EveryDirection(),
        direction = VerticalDirection(),
        state_auxiliary = dg.state_auxiliary,
    )

    if discrete_splitting
        numerical_flux_first_order =
            (dg.numerical_flux_first_order, (dg.numerical_flux_first_order,))
    else
        numerical_flux_first_order = dg.numerical_flux_first_order
    end
    slow_dg = remainder_DGModel(
        dg,
        (fast_dg,);
        numerical_flux_first_order = numerical_flux_first_order,
    )

    fast_method = LSRK144NiegemannDiehlBusch
    nsubsteps = (50,)

    Q = init_ode_state(dg, FT(0))
    println("Q  ",Q.data[:, 4, :])
    ρ = Array(Q.data[:, 1, :])
        ρu = Array(Q.data[:, 2, :])
        ρv = Array(Q.data[:, 3, :])
        ρw = Array(Q.data[:, 4, :])
        ρe = Array(Q.data[:, 5, :])
        u = ρu ./ ρ
        v = ρv ./ ρ
        w = ρw ./ ρ
        @info "u = $(extrema(u))"
        @info "v = $(extrema(v))"
        @info "w = $(extrema(w))"
        @info "e = $(extrema(ρe))"

    odesolver = MIS2(
        slow_dg,
        fast_dg,
        fast_method,
        nsubsteps[1],
        Q;
        dt = dt,
        t0 = 0,
    )

    @test getsteps(odesolver) == 0

    filterorder = 18
    filter = ExponentialFilter(grid, 0, filterorder)
    cbfilter = EveryXSimulationSteps(100) do
        println(" Filter ")
        Filters.apply!(
            Q,
            :,
            #AtmosFilterPerturbations(model),
            grid,
            filter,
            #state_auxiliary = dg.state_auxiliary,
        )
        nothing
    end

    cbcheck = EveryXSimulationSteps(1) do
        println("Q  ",Q.data[:, 4, :])
        ρ = Array(Q.data[:, 1, :])
        ρu = Array(Q.data[:, 2, :])
        ρv = Array(Q.data[:, 3, :])
        ρw = Array(Q.data[:, 4, :])
        ρe = Array(Q.data[:, 5, :])
        u = ρu ./ ρ
        v = ρv ./ ρ
        w = ρw ./ ρ
        @info "u = $(extrema(u))"
        @info "v = $(extrema(v))"
        @info "w = $(extrema(w))"
        @info "e = $(extrema(ρe))"
        nothing
    end

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
    #cbinfo = EveryXSimulationSteps(1) do (s = false)
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
    callbacks = (cbinfo, cbfilter, cbcheck)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_zonalflow_box" *
            "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
            "_$(zero_ref_state_velocity)_MIS"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
        #cbvtk = EveryXSimulationSteps(1) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(odesolver))
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
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

    @test getsteps(odesolver) == nsteps

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    engf
end

Base.@kwdef struct ZonalFlowSetup{FT}
    domain_height::FT = 30e3
    T0::FT = 300
    u0::FT = 200
    v0::FT = 200
end

function (setup::ZonalFlowSetup)(problem, atmos, state, aux, localgeo, t)
    # callable to set initial conditions
    FT = eltype(state)

    (x, y, z) = localgeo.coord

    u0 = setup.u0
    v0 = setup.v0
    T0 = setup.T0

    ρ = aux.ref_state.ρ
    e_kin = u0 ^ 2 / 2 + v0 ^ 2 / 2 
    e_pot = gravitational_potential(atmos.orientation, aux)

    state.ρ = ρ
    state.ρu = ρ * SVector(u0, v0, 0)
    state.ρe = ρ * total_energy(atmos.param_set, e_kin, e_pot, T0)
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "zonalflow",
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

import ClimateMachine.Atmos: atmos_init_aux!, vars_state

struct ZonalReferenceState{FT} <: ReferenceState
    T0::FT
    u0::FT
end

vars_state(::ZonalReferenceState, ::Auxiliary, FT) =
  @vars(ρ::FT, ρu::SVector{3, FT}, ρe::FT, p::FT, T::FT)
function atmos_init_aux!(
    refstate::ZonalReferenceState,
    atmos::AtmosModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    (x, y, z) = geom.coord
    param_set = atmos.param_set
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    _grav::FT = grav(param_set)

    T0 = refstate.T0
    u0 = refstate.u0

    e_int = internal_energy(param_set, T0)
    e_pot = gravitational_potential(atmos.orientation, aux)
    e_kin = u0 ^ 2 / 2

    p = _MSLP * exp(-e_pot / (_R_d * T0))
    ρ = p / (_R_d * T0)

    aux.ref_state.ρ = ρ
    aux.ref_state.ρu = ρ * SVector(u0, 0, 0)
    aux.ref_state.ρe = ρ * (e_int + e_pot + e_kin)
    aux.ref_state.p = p
    aux.ref_state.T = T0
end

main()
