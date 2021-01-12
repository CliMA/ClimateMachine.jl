using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Geometry
using ClimateMachine.ODESolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics
using ClimateMachine.Atmos
using ClimateMachine.TurbulenceClosures
using ClimateMachine.Orientations
using ClimateMachine.BalanceLaws
using ClimateMachine.VariableTemplates: flattenednames

import ..BalanceLaws: source

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cv_d, cp_d, grav, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test
using Random, UnPack

const output_vtk = true

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    numelem_horz = 12
    numelem_vert = 12

    timeend = 15000
    outputtime = 200

    FT = Float64

    test_run(
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

function test_run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
)
    setup = PBLSetup{FT}()

    horz_range = range(FT(0),
                       stop = setup.domain_length,
                       length = numelem_horz+1)
    vert_range = range(FT(0),
                       stop = setup.domain_height,
                       length = numelem_vert+1)
    brickrange = (horz_range, horz_range, vert_range)
    topology = StackedBrickTopology(mpicomm,
                                    brickrange,
                                    periodicity=(true, true, false))

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = setup,
        orientation = FlatOrientation(),
        ref_state = NoReferenceState(),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        source = (Gravity(), HeatFlux()),
    )

    dg = DGModel(
        model,
        grid,
        #RoeNumericalFlux(),
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    # determine the time step
    acoustic_speed = soundspeed_air(model.param_set, FT(setup.θ0))
    dx = min_node_distance(grid)
    cfl = FT(1.0)
    dt = cfl * dx / acoustic_speed

    Q = init_ode_state(dg, FT(0); init_on_cpu=true)


    odesolver = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

    filterorder = 18
    filter = ExponentialFilter(grid, 0, filterorder)
    cbfilter = EveryXSimulationSteps(1) do
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
    
    cbcheck = EveryXSimulationSteps(1000) do
        ρ = Array(Q.data[:, 1, :])
        ρu = Array(Q.data[:, 2, :])
        ρv = Array(Q.data[:, 3, :])
        ρw = Array(Q.data[:, 4, :])
        u = ρu ./ ρ
        v = ρv ./ ρ
        w = ρw ./ ρ
        @info "u = $(extrema(u))"
        @info "v = $(extrema(v))"
        @info "w = $(extrema(w))"
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
    callbacks = (cbinfo, cbcheck, cbfilter)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_pbl" *
            "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
            "_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(
        Q,
        odesolver;
        timeend = timeend,
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

Base.@kwdef struct PBLSetup{FT}
    domain_length::FT = 3200
    domain_height::FT = 1500
    θ0::FT = 300
    zm::FT = 500
    st::FT = 1e-4 / grav(param_set)
end

function (setup::PBLSetup)(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)

    (x, y, z) = localgeo.coord
    
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)

    θ0 = setup.θ0
    zm = setup.zm
    st = setup.st
  
    α = _grav / (_cp_d * θ0)
    β = _cp_d / _R_d
    θ = θ0 * (z <= zm ? 1 : 1 + (z - zm) * st)
    if z <= zm
      p = _MSLP * (1 - α * z) ^ β
    else
      p = _MSLP * (1 - α * (zm + log(1 + st * (z - zm)) / st)) ^ β
    end
    T = θ * (p / _MSLP) ^ (_R_d / _cp_d)

    ρ = p / (_R_d * T)

    e_kin = FT(0)
    e_pot = aux.orientation.Φ

    rnd = rand() - FT(0.5)
    fac = rnd * max(FT(0), 1 - z / zm)
    δθ = FT(0.001) * fac
    δT = δθ * (p / _MSLP) ^ (_R_d / _cp_d)
    δw = FT(0.2) * fac
    δe = δw ^ 2 / 2 + _cv_d * δT

    state.ρ = ρ
    state.ρu = ρ * SVector(0, 0, δw)
    state.ρe = ρ * (total_energy(param_set, e_kin, e_pot, T) + δe)
end

struct HeatFlux{PV <: Energy} <: TendencyDef{Source, PV} end
HeatFlux() = HeatFlux{Energy}()
function source(s::HeatFlux{Energy}, m, args)
    @unpack state, aux = args
    @unpack ts = args.precomputed
    FT = eltype(aux)
    
    z = altitude(m, aux)
    _cv_d::FT = cv_d(m.param_set)

    ρ = state.ρ
    h0 = FT(1 / 100)
    hscale = FT(25)
    hflux = h0 * exp(-z / hscale) / hscale
    return _cv_d * ρ * hflux * exner(ts) 
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "pbl",
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
