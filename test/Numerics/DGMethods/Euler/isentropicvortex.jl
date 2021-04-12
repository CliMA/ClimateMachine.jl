using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.Orientations
using ClimateMachine.SystemSolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.VTK

using CLIMAParameters
using CLIMAParameters.Planet: kappa_d
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

include("isentropicvortex_setup.jl")

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output_vtk = false

function main()
    ClimateMachine.init(parse_clargs = true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    numlevels = integration_testing ? 4 : 1

    expected_error = Dict()

    # just to make it shorter and aligning
    Rusanov = RusanovNumericalFlux()
    Central = CentralNumericalFluxFirstOrder()
    Roe = RoeNumericalFlux()
    HLLC = HLLCNumericalFlux()
    RoeMoist = RoeNumericalFluxMoist()
    RoeMoistLM = RoeNumericalFluxMoist(; LM = true)
    RoeMoistHH = RoeNumericalFluxMoist(; HH = true)
    RoeMoistLV = RoeNumericalFluxMoist(; LV = true)
    RoeMoistLVPP = RoeNumericalFluxMoist(; LVPP = true)

    expected_error[Float64, 2, Rusanov, 1] = 1.1990999506538110e+01
    expected_error[Float64, 2, Rusanov, 2] = 2.0813000228865612e+00
    expected_error[Float64, 2, Rusanov, 3] = 6.3752572004789149e-02
    expected_error[Float64, 2, Rusanov, 4] = 2.0984975076420455e-03

    expected_error[Float64, 2, Central, 1] = 2.0840574601661153e+01
    expected_error[Float64, 2, Central, 2] = 2.9255455365299827e+00
    expected_error[Float64, 2, Central, 3] = 3.6935849488949657e-01
    expected_error[Float64, 2, Central, 4] = 8.3528804679907434e-03

    expected_error[Float64, 2, Roe, 1] = 1.2891386634733328e+01
    expected_error[Float64, 2, Roe, 2] = 1.3895805145495934e+00
    expected_error[Float64, 2, Roe, 3] = 6.6174934435569849e-02
    expected_error[Float64, 2, Roe, 4] = 2.1917769287815940e-03

    expected_error[Float64, 2, RoeMoist, 1] = 1.2415957884123003e+01
    expected_error[Float64, 2, RoeMoist, 2] = 1.4188653323424882e+00
    expected_error[Float64, 2, RoeMoist, 3] = 6.7913325894248130e-02
    expected_error[Float64, 2, RoeMoist, 4] = 2.0377963111049128e-03

    expected_error[Float64, 2, RoeMoistLM, 1] = 1.2316906651180444e+01
    expected_error[Float64, 2, RoeMoistLM, 2] = 1.4359406523244560e+00
    expected_error[Float64, 2, RoeMoistLM, 3] = 6.8650238542101505e-02
    expected_error[Float64, 2, RoeMoistLM, 4] = 2.0000156591586842e-03

    expected_error[Float64, 2, RoeMoistHH, 1] = 1.2425625606467793e+01
    expected_error[Float64, 2, RoeMoistHH, 2] = 1.4029458093339020e+00
    expected_error[Float64, 2, RoeMoistHH, 3] = 6.8648208937091268e-02
    expected_error[Float64, 2, RoeMoistHH, 4] = 2.0711985861781648e-03

    expected_error[Float64, 2, RoeMoistLV, 1] = 1.2415957884123003e+01
    expected_error[Float64, 2, RoeMoistLV, 2] = 1.4188653323424882e+00
    expected_error[Float64, 2, RoeMoistLV, 3] = 6.7913325894248130e-02
    expected_error[Float64, 2, RoeMoistLV, 4] = 2.0377963111049128e-03

    expected_error[Float64, 2, RoeMoistLVPP, 1] = 1.2441813136310969e+01
    expected_error[Float64, 2, RoeMoistLVPP, 2] = 2.0219325767566727e+00
    expected_error[Float64, 2, RoeMoistLVPP, 3] = 6.7716921628626484e-02
    expected_error[Float64, 2, RoeMoistLVPP, 4] = 2.1051129944994005e-03

    expected_error[Float64, 2, HLLC, 1] = 1.2889756097329746e+01
    expected_error[Float64, 2, HLLC, 2] = 1.3895808565455936e+00
    expected_error[Float64, 2, HLLC, 3] = 6.6175116756217900e-02
    expected_error[Float64, 2, HLLC, 4] = 2.1917772135679118e-03

    expected_error[Float64, 3, Rusanov, 1] = 3.7918869862613858e+00
    expected_error[Float64, 3, Rusanov, 2] = 6.5816485664822677e-01
    expected_error[Float64, 3, Rusanov, 3] = 2.0160333422867591e-02
    expected_error[Float64, 3, Rusanov, 4] = 6.6360317881818034e-04

    expected_error[Float64, 3, Central, 1] = 6.5903683487905749e+00
    expected_error[Float64, 3, Central, 2] = 9.2513872939749997e-01
    expected_error[Float64, 3, Central, 3] = 1.1680141169828175e-01
    expected_error[Float64, 3, Central, 4] = 2.6414127301659534e-03

    expected_error[Float64, 3, Roe, 1] = 4.0766143963611068e+00
    expected_error[Float64, 3, Roe, 2] = 4.3942394181655547e-01
    expected_error[Float64, 3, Roe, 3] = 2.0926351682882375e-02
    expected_error[Float64, 3, Roe, 4] = 6.9310072176312712e-04

    expected_error[Float64, 3, RoeMoist, 1] = 3.9262706246552574e+00
    expected_error[Float64, 3, RoeMoist, 2] = 4.4868461432545598e-01
    expected_error[Float64, 3, RoeMoist, 3] = 2.1476079330305119e-02
    expected_error[Float64, 3, RoeMoist, 4] = 6.4440777504566171e-04

    expected_error[Float64, 3, RoeMoistLM, 1] = 3.8949478745407458e+00
    expected_error[Float64, 3, RoeMoistLM, 2] = 4.5408430461734528e-01
    expected_error[Float64, 3, RoeMoistLM, 3] = 2.1709111570716800e-02
    expected_error[Float64, 3, RoeMoistLM, 4] = 6.3246048386171073e-04

    expected_error[Float64, 3, RoeMoistHH, 1] = 3.9293278268948622e+00
    expected_error[Float64, 3, RoeMoistHH, 2] = 4.4365041912830866e-01
    expected_error[Float64, 3, RoeMoistHH, 3] = 2.1708469753267460e-02
    expected_error[Float64, 3, RoeMoistHH, 4] = 6.5497050185153694e-04

    expected_error[Float64, 3, RoeMoistLV, 1] = 3.9262706246552574e+00
    expected_error[Float64, 3, RoeMoistLV, 2] = 4.4868461432545598e-01
    expected_error[Float64, 3, RoeMoistLV, 3] = 2.1476079330305119e-02
    expected_error[Float64, 3, RoeMoistLV, 4] = 6.4440777504566171e-04

    expected_error[Float64, 3, RoeMoistLVPP, 1] = 3.9344467732944071e+00
    expected_error[Float64, 3, RoeMoistLVPP, 2] = 6.3939122178436703e-01
    expected_error[Float64, 3, RoeMoistLVPP, 3] = 2.1413970848172485e-02
    expected_error[Float64, 3, RoeMoistLVPP, 4] = 6.6569517942933988e-04

    expected_error[Float64, 3, HLLC, 1] = 4.0760987751605402e+00
    expected_error[Float64, 3, HLLC, 2] = 4.3942404996518236e-01
    expected_error[Float64, 3, HLLC, 3] = 2.0926409337758904e-02
    expected_error[Float64, 3, HLLC, 4] = 6.9310081182571569e-04

    expected_error[Float32, 2, Rusanov, 1] = 1.1990781784057617e+01
    expected_error[Float32, 2, Rusanov, 2] = 2.0813269615173340e+00
    expected_error[Float32, 2, Rusanov, 3] = 6.7035309970378876e-02
    expected_error[Float32, 2, Rusanov, 4] = 5.3008597344160080e-02

    expected_error[Float32, 2, Central, 1] = 2.0840391159057617e+01
    expected_error[Float32, 2, Central, 2] = 2.9256355762481689e+00
    expected_error[Float32, 2, Central, 3] = 3.7092915177345276e-01
    expected_error[Float32, 2, Central, 4] = 1.1543693393468857e-01

    expected_error[Float32, 2, Roe, 1] = 1.2891359329223633e+01
    expected_error[Float32, 2, Roe, 2] = 1.3895936012268066e+00
    expected_error[Float32, 2, Roe, 3] = 6.8037144839763641e-02
    expected_error[Float32, 2, Roe, 4] = 3.8893952965736389e-02

    expected_error[Float32, 2, RoeMoist, 1] = 1.2415886878967285e+01
    expected_error[Float32, 2, RoeMoist, 2] = 1.4188879728317261e+00
    expected_error[Float32, 2, RoeMoist, 3] = 6.9743692874908447e-02
    expected_error[Float32, 2, RoeMoist, 4] = 3.7607192993164063e-02

    expected_error[Float32, 2, RoeMoistLM, 1] = 1.2316809654235840e+01
    expected_error[Float32, 2, RoeMoistLM, 2] = 4.5408430461734528e+00
    expected_error[Float32, 2, RoeMoistLM, 3] = 7.0370830595493317e-02
    expected_error[Float32, 2, RoeMoistLM, 4] = 3.7792034447193146e-02

    expected_error[Float32, 2, RoeMoistHH, 1] = 1.2425449371337891e+01
    expected_error[Float32, 2, RoeMoistHH, 2] = 1.4030106067657471e+00
    expected_error[Float32, 2, RoeMoistHH, 3] = 7.0363849401473999e-02
    expected_error[Float32, 2, RoeMoistHH, 4] = 3.7904966622591019e-02

    expected_error[Float32, 2, RoeMoistLV, 1] = 1.2415886878967285e+01
    expected_error[Float32, 2, RoeMoistLV, 2] = 1.4188879728317261e+00
    expected_error[Float32, 2, RoeMoistLV, 3] = 6.9743692874908447e-02
    expected_error[Float32, 2, RoeMoistLV, 4] = 3.7607192993164063e-02

    expected_error[Float32, 2, RoeMoistLVPP, 1] = 1.2441481590270996e+01
    expected_error[Float32, 2, RoeMoistLVPP, 2] = 2.0217459201812744e+00
    expected_error[Float32, 2, RoeMoistLVPP, 3] = 7.0483185350894928e-02
    expected_error[Float32, 2, RoeMoistLVPP, 4] = 5.1601748913526535e-02

    expected_error[Float32, 2, HLLC, 1] = 1.2889801025390625e+01
    expected_error[Float32, 2, HLLC, 2] = 1.3895059823989868e+00
    expected_error[Float32, 2, HLLC, 3] = 6.8006515502929688e-02
    expected_error[Float32, 2, HLLC, 4] = 3.8637656718492508e-02

    expected_error[Float32, 3, Rusanov, 1] = 3.7918186187744141e+00
    expected_error[Float32, 3, Rusanov, 2] = 6.5816193819046021e-01
    expected_error[Float32, 3, Rusanov, 3] = 2.0893247798085213e-02
    expected_error[Float32, 3, Rusanov, 4] = 1.1554701253771782e-02

    expected_error[Float32, 3, Central, 1] = 6.5903329849243164e+00
    expected_error[Float32, 3, Central, 2] = 9.2512512207031250e-01
    expected_error[Float32, 3, Central, 3] = 1.1707859486341476e-01
    expected_error[Float32, 3, Central, 4] = 2.1001411601901054e-02

    expected_error[Float32, 3, Roe, 1] = 4.0765657424926758e+00
    expected_error[Float32, 3, Roe, 2] = 4.3941807746887207e-01
    expected_error[Float32, 3, Roe, 3] = 2.1365188062191010e-02
    expected_error[Float32, 3, Roe, 4] = 9.3323951587080956e-03

    expected_error[Float32, 3, RoeMoist, 1] = 3.9262301921844482e+00
    expected_error[Float32, 3, RoeMoist, 2] = 4.4864514470100403e-01
    expected_error[Float32, 3, RoeMoist, 3] = 2.1889146417379379e-02
    expected_error[Float32, 3, RoeMoist, 4] = 8.8266804814338684e-03

    expected_error[Float32, 3, RoeMoistLM, 1] = 3.8948786258697510e+00
    expected_error[Float32, 3, RoeMoistLM, 2] = 4.5405751466751099e-01
    expected_error[Float32, 3, RoeMoistLM, 3] = 2.2112159058451653e-02
    expected_error[Float32, 3, RoeMoistLM, 4] = 8.7371272966265678e-03

    expected_error[Float32, 3, RoeMoistHH, 1] = 3.9292929172515869e+00
    expected_error[Float32, 3, RoeMoistHH, 2] = 4.4363334774971008e-01
    expected_error[Float32, 3, RoeMoistHH, 3] = 2.2118536755442619e-02
    expected_error[Float32, 3, RoeMoistHH, 4] = 8.9262928813695908e-03

    expected_error[Float32, 3, RoeMoistLV, 1] = 3.9262151718139648e+00
    expected_error[Float32, 3, RoeMoistLV, 2] = 4.4865489006042480e-01
    expected_error[Float32, 3, RoeMoistLV, 3] = 2.1889505907893181e-02
    expected_error[Float32, 3, RoeMoistLV, 4] = 8.8385939598083496e-03

    expected_error[Float32, 3, RoeMoistLVPP, 1] = 3.9343423843383789e+00
    expected_error[Float32, 3, RoeMoistLVPP, 2] = 6.3935810327529907e-01
    expected_error[Float32, 3, RoeMoistLVPP, 3] = 2.1930629387497902e-02
    expected_error[Float32, 3, RoeMoistLVPP, 4] = 1.0632344521582127e-02

    expected_error[Float32, 3, HLLC, 1] = 4.0760631561279297e+00
    expected_error[Float32, 3, HLLC, 2] = 4.3940672278404236e-01
    expected_error[Float32, 3, HLLC, 3] = 2.1352596580982208e-02
    expected_error[Float32, 3, HLLC, 4] = 9.2315869405865669e-03

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32), dims in (2, 3)
            for NumericalFlux in (
                Rusanov,
                Central,
                Roe,
                HLLC,
                RoeMoist,
                RoeMoistLM,
                RoeMoistHH,
                RoeMoistLV,
                RoeMoistLVPP,
            )
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
                    errors[level] = test_run(
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

function test_run(
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

    grid = SpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    problem =
        AtmosProblem(boundaryconditions = (), init_state_prognostic = setup)
    if NumericalFlux isa RoeNumericalFluxMoist
        moisture = EquilMoist()
    else
        moisture = DryModel()
    end
    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = NoReferenceState(),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = moisture,
    )
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        physics;
        problem = problem,
        orientation = NoOrientation(),
        source = (),
    )

    dg = DGModel(
        model,
        grid,
        NumericalFlux,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    timeend = FT(2 * setup.domain_halflength / 10 / setup.translation_speed)

    # determine the time step
    elementsize = minimum(step.(brickrange))
    dt = elementsize / soundspeed_air(param_set, setup.T∞) / polynomialorder^2
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0))
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

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
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

        writepvtu(
            pvtuprefix,
            prefixes,
            (statenames..., exactnames...),
            eltype(Q),
        )

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
