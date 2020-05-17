using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies: BrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid
using ClimateMachine.DGmethods: DGModel, init_ode_state, LocalGeometry
using ClimateMachine.DGmethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder
using ClimateMachine.ODESolvers
using ClimateMachine.GeneralizedMinimalResidualSolver:
    GeneralizedMinimalResidual
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.MPIStateArrays: euclidean_distance
using ClimateMachine.MoistThermodynamics:
    air_density, total_energy, internal_energy, soundspeed_air
using ClimateMachine.Atmos:
    AtmosModel,
    AtmosAcousticLinearModel,
    RemainderModel,
    NoOrientation,
    NoReferenceState,
    ReferenceState,
    DryModel,
    NoPrecipitation,
    NoRadiation,
    ConstantViscosityWithDivergence,
    vars_state_conservative
using ClimateMachine.VariableTemplates: @vars, Vars, flattenednames
import ClimateMachine.Atmos: atmos_init_aux!, vars_state_auxiliary

using CLIMAParameters
using CLIMAParameters.Planet: kappa_d
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output_vtk = false

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    numlevels = integration_testing ? 4 : 1

    expected_error = Dict()

    expected_error[Float64, false, 1] = 2.3225467541870387e+01
    expected_error[Float64, false, 2] = 5.2663709730295070e+00
    expected_error[Float64, false, 3] = 1.2183770894070467e-01
    expected_error[Float64, false, 4] = 2.8660813871243937e-03

    expected_error[Float64, true, 1] = 2.3225467618783981e+01
    expected_error[Float64, true, 2] = 5.2663709730207771e+00
    expected_error[Float64, true, 3] = 1.2183770891083319e-01
    expected_error[Float64, true, 4] = 2.8660813810759854e-03

    @testset "$(@__FILE__)" begin
        for FT in (Float64,), dims in 2
            for split_explicit_implicit in (false, true)
                let
                    split = split_explicit_implicit ? "(Nonlinear, Linear)" :
                        "(Full, Linear)"
                    @info @sprintf """Configuration
                                      ArrayType = %s
                                      FT    = %s
                                      dims      = %d
                                      splitting = %s
                                      """ ArrayType "$FT" dims split
                end

                setup = IsentropicVortexSetup{FT}()
                errors = Vector{FT}(undef, numlevels)

                for level in 1:numlevels
                    numelems =
                        ntuple(dim -> dim == 3 ? 1 : 2^(level - 1) * 5, dims)
                    errors[level] = run(
                        mpicomm,
                        ArrayType,
                        polynomialorder,
                        numelems,
                        setup,
                        split_explicit_implicit,
                        FT,
                        dims,
                        level,
                    )

                    @test errors[level] ≈
                          expected_error[FT, split_explicit_implicit, level]
                end

                rates = @. log2(
                    first(errors[1:(numlevels - 1)]) /
                    first(errors[2:numlevels]),
                )
                numlevels > 1 && @info "Convergence rates\n" * join(
                    [
                        "rate for levels $l → $(l + 1) = $(rates[l])"
                        for l in 1:(numlevels - 1)
                    ],
                    "\n",
                )
            end
        end
    end
end

function run(
    mpicomm,
    ArrayType,
    polynomialorder,
    numelems,
    setup,
    split_explicit_implicit,
    FT,
    dims,
    level,
)
    brickrange = ntuple(dims) do dim
        range(
            -setup.domain_halflength;
            length = numelems[dim] + 1,
            stop = setup.domain_halflength,
        )
    end

    topology = BrickTopology(
        mpicomm,
        brickrange;
        periodicity = ntuple(_ -> true, dims),
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        orientation = NoOrientation(),
        ref_state = IsentropicVortexReferenceState{FT}(setup),
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = nothing,
        boundarycondition = (),
        init_state_conservative = isentropicvortex_initialcondition!,
    )

    linear_model = AtmosAcousticLinearModel(model)
    nonlinear_model = RemainderModel(model, (linear_model,))

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    dg_linear = DGModel(
        linear_model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        state_auxiliary = dg.state_auxiliary,
    )

    if split_explicit_implicit
        dg_nonlinear = DGModel(
            nonlinear_model,
            grid,
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient();
            state_auxiliary = dg.state_auxiliary,
        )
    end

    timeend = FT(2 * setup.domain_halflength / setup.translation_speed)

    # determine the time step
    elementsize = minimum(step.(brickrange))
    dt =
        elementsize / soundspeed_air(model.param_set, setup.T∞) /
        polynomialorder^2
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0))

    linearsolver = GeneralizedMinimalResidual(Q; M = 10, rtol = 1e-10)
    ode_solver = ARK2GiraldoKellyConstantinescu(
        split_explicit_implicit ? dg_nonlinear : dg,
        dg_linear,
        LinearBackwardEulerSolver(linearsolver; isadjustable = true),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = split_explicit_implicit,
        paperversion = true,
    )

    eng0 = norm(Q)
    dims == 2 && (numelems = (numelems..., 0))
    @info @sprintf """Starting refinement level %d
                      numelems  = (%d, %d, %d)
                      dt        = %.16e
                      norm(Q₀)  = %.16e
                      """ level numelems... dt eng0

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
                              """ gettime(ode_solver) runtime energy
        end
    end
    callbacks = (cbinfo,)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_isentropicvortex_imex" *
            "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)_level$(level)" *
            "_$(split_explicit_implicit)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

        # setup the output callback
        outputtime = timeend
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(ode_solver))
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, ode_solver; timeend = timeend, callbacks = callbacks)

    # final statistics
    Qe = init_ode_state(dg, timeend)
    engf = norm(Q)
    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished refinement level %d
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ level engf engf / eng0 engf - eng0 errf errf / engfe
    errf
end

Base.@kwdef struct IsentropicVortexSetup{FT}
    p∞::FT = 10^5
    T∞::FT = 300
    ρ∞::FT = air_density(param_set, FT(T∞), FT(p∞))
    translation_speed::FT = 150
    translation_angle::FT = pi / 4
    vortex_speed::FT = 50
    vortex_radius::FT = 1 // 200
    domain_halflength::FT = 1 // 20
end

struct IsentropicVortexReferenceState{FT} <: ReferenceState
    setup::IsentropicVortexSetup{FT}
end
vars_state_auxiliary(::IsentropicVortexReferenceState, FT) =
    @vars(ρ::FT, ρe::FT, p::FT, T::FT)
function atmos_init_aux!(
    m::IsentropicVortexReferenceState,
    atmos::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    setup = m.setup
    ρ∞ = setup.ρ∞
    p∞ = setup.p∞
    T∞ = setup.T∞

    aux.ref_state.ρ = ρ∞
    aux.ref_state.p = p∞
    aux.ref_state.T = T∞
    aux.ref_state.ρe = ρ∞ * internal_energy(atmos.param_set, T∞)
end

function isentropicvortex_initialcondition!(bl, state, aux, coords, t, args...)
    setup = bl.ref_state.setup
    FT = eltype(state)
    x = MVector(coords)

    ρ∞ = setup.ρ∞
    p∞ = setup.p∞
    T∞ = setup.T∞
    translation_speed = setup.translation_speed
    α = setup.translation_angle
    vortex_speed = setup.vortex_speed
    R = setup.vortex_radius
    L = setup.domain_halflength

    u∞ = SVector(translation_speed * cos(α), translation_speed * sin(α), 0)

    x .-= u∞ * t
    # make the function periodic
    x .-= floor.((x .+ L) / 2L) * 2L

    @inbounds begin
        r = sqrt(x[1]^2 + x[2]^2)
        δu_x = -vortex_speed * x[2] / R * exp(-(r / R)^2 / 2)
        δu_y = vortex_speed * x[1] / R * exp(-(r / R)^2 / 2)
    end
    u = u∞ .+ SVector(δu_x, δu_y, 0)

    _kappa_d::FT = kappa_d(param_set)
    T = T∞ * (1 - _kappa_d * vortex_speed^2 / 2 * ρ∞ / p∞ * exp(-(r / R)^2))
    # adiabatic/isentropic relation
    p = p∞ * (T / T∞)^(FT(1) / _kappa_d)
    ρ = air_density(bl.param_set, T, p)

    state.ρ = ρ
    state.ρu = ρ * u
    e_kin = u' * u / 2
    state.ρe = ρ * total_energy(bl.param_set, e_kin, FT(0), T)
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    Qe,
    model,
    testname = "isentropicvortex_imex",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state_conservative(model, eltype(Q)))
    exactnames = statenames .* "_exact"

    writevtk(filename, Q, dg, statenames, Qe, exactnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., exactnames...))

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
