using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies: BrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid
using ClimateMachine.DGmethods: DGModel, init_ode_state
using ClimateMachine.DGmethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxFirstOrder
using ClimateMachine.ODESolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.MPIStateArrays: euclidean_distance
using ClimateMachine.MoistThermodynamics:
    air_density, total_energy, soundspeed_air, PhaseDry_given_pT
using ClimateMachine.Atmos:
    AtmosModel,
    NoOrientation,
    NoReferenceState,
    DryModel,
    NoPrecipitation,
    NoRadiation,
    ConstantViscosityWithDivergence,
    vars_state_conservative
using ClimateMachine.VariableTemplates: flattenednames

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

    # just to make it shorter and aligning
    Rusanov = RusanovNumericalFlux
    Central = CentralNumericalFluxFirstOrder

    expected_error[Float64, 2, Rusanov, 1] = 1.1990999506538110e+01
    expected_error[Float64, 2, Rusanov, 2] = 2.0813000228865612e+00
    expected_error[Float64, 2, Rusanov, 3] = 6.3752572004789149e-02
    expected_error[Float64, 2, Rusanov, 4] = 2.0984975076420455e-03

    expected_error[Float64, 2, Central, 1] = 2.0840574601661153e+01
    expected_error[Float64, 2, Central, 2] = 2.9255455365299827e+00
    expected_error[Float64, 2, Central, 3] = 3.6935849488949657e-01
    expected_error[Float64, 2, Central, 4] = 8.3528804679907434e-03

    expected_error[Float64, 3, Rusanov, 1] = 3.7918869862613858e+00
    expected_error[Float64, 3, Rusanov, 2] = 6.5816485664822677e-01
    expected_error[Float64, 3, Rusanov, 3] = 2.0160333422867591e-02
    expected_error[Float64, 3, Rusanov, 4] = 6.6360317881818034e-04

    expected_error[Float64, 3, Central, 1] = 6.5903683487905749e+00
    expected_error[Float64, 3, Central, 2] = 9.2513872939749997e-01
    expected_error[Float64, 3, Central, 3] = 1.1680141169828175e-01
    expected_error[Float64, 3, Central, 4] = 2.6414127301659534e-03

    expected_error[Float32, 2, Rusanov, 1] = 1.1990781784057617e+01
    expected_error[Float32, 2, Rusanov, 2] = 2.0813269615173340e+00
    expected_error[Float32, 2, Rusanov, 3] = 6.7035309970378876e-02
    expected_error[Float32, 2, Rusanov, 4] = 5.3008597344160080e-02

    expected_error[Float32, 2, Central, 1] = 2.0840391159057617e+01
    expected_error[Float32, 2, Central, 2] = 2.9256355762481689e+00
    expected_error[Float32, 2, Central, 3] = 3.7092915177345276e-01
    expected_error[Float32, 2, Central, 4] = 1.1543693393468857e-01

    expected_error[Float32, 3, Rusanov, 1] = 3.7918186187744141e+00
    expected_error[Float32, 3, Rusanov, 2] = 6.5816193819046021e-01
    expected_error[Float32, 3, Rusanov, 3] = 2.0893247798085213e-02
    expected_error[Float32, 3, Rusanov, 4] = 1.1554701253771782e-02

    expected_error[Float32, 3, Central, 1] = 6.5903329849243164e+00
    expected_error[Float32, 3, Central, 2] = 9.2512512207031250e-01
    expected_error[Float32, 3, Central, 3] = 1.1707859486341476e-01
    expected_error[Float32, 3, Central, 4] = 2.1001411601901054e-02

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32), dims in (2, 3)
            for NumericalFlux in (RusanovNumericalFlux, Central)
                @info @sprintf """Configuration
                                  ArrayType     = %s
                                  FT        = %s
                                  NumericalFlux = %s
                                  dims          = %d
                                  """ ArrayType "$FT" "$NumericalFlux" dims

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
                        NumericalFlux,
                        setup,
                        FT,
                        dims,
                        level,
                    )

                    rtol = sqrt(eps(FT))
                    # increase rtol for comparing with GPU results using Float32
                    if FT === Float32 && ArrayType !== Array
                        rtol *= 10 # why does this factor have to be so big :(
                    end
                    @test isapprox(
                        errors[level],
                        expected_error[FT, dims, NumericalFlux, level];
                        rtol = rtol,
                    )
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
    NumericalFlux,
    setup,
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
        ref_state = NoReferenceState(),
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = nothing,
        boundarycondition = (),
        init_state_conservative = isentropicvortex_initialcondition!,
    )

    dg = DGModel(
        model,
        grid,
        NumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    timeend = FT(2 * setup.domain_halflength / 10 / setup.translation_speed)

    # determine the time step
    elementsize = minimum(step.(brickrange))
    dt =
        elementsize / soundspeed_air(model.param_set, setup.T∞) /
        polynomialorder^2
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0), setup)
    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

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
                              """ gettime(lsrk) runtime energy
        end
    end
    callbacks = (cbinfo,)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_isentropicvortex" *
            "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)_level$(level)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

        # setup the output callback
        outputtime = timeend
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(lsrk), setup)
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, lsrk; timeend = timeend, callbacks = callbacks)

    # final statistics
    Qe = init_ode_state(dg, timeend, setup)
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

function isentropicvortex_initialcondition!(bl, state, aux, coords, t, args...)
    setup = first(args)
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
    ts = PhaseDry_given_pT(bl.param_set, p, T)
    ρ = air_density(ts)

    e_pot = FT(0)
    state.ρ = ρ
    state.ρu = ρ * u
    e_kin = u' * u / 2
    state.ρe = ρ * total_energy(e_kin, e_pot, ts)
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    Qe,
    model,
    testname = "isentropicvortex",
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
