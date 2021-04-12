using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Test
import ClimateMachine.VTK: writevtk, writepvtu
import ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using Dates

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{ns, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(
    ::Pseudo1D{ns, α, β},
    aux::Vars,
    geom::LocalGeometry,
) where {ns, α, β}
    # Direction of flow is ns[i] with magnitude α
    aux.advection.u = hcat(ntuple(i -> α * ns[i], Val(length(ns)))...)

    # Diffusion of strength β in the ns[i] direction
    aux.diffusion.D = hcat(ntuple(i -> β * ns[i] * ns[i]', Val(length(ns)))...)
end

function gaussian(x, t, α, β, μ, δ)
    exp(-(x - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end
function ∇gaussian(n, x, t, α, β, μ, δ)
    -2n * (x - μ - α * t) / (4 * β * (δ + t)) *
    exp(-(x - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end
function initial_condition!(
    ::Pseudo1D{ns, α, β, μ, δ},
    state,
    aux,
    localgeo,
    t,
) where {ns, α, β, μ, δ}
    ρ = ntuple(Val(length(ns))) do i
        ξn = dot(ns[i], localgeo.coord)
        gaussian(ξn, t, α, β, μ, δ)
    end
    state.ρ = length(ns) == 1 ? ρ[1] : ρ
end

inhomogeneous_data!(::Val{0}, P::Pseudo1D, x...) = initial_condition!(P, x...)

function inhomogeneous_data!(
    ::Val{1},
    ::Pseudo1D{ns, α, β, μ, δ},
    ∇state,
    aux,
    x,
    t,
) where {ns, α, β, μ, δ}
    ∇state.ρ = hcat(ntuple(Val(length(ns))) do i
        ξn = dot(ns[i], x)
        ∇gaussian(ns[i], ξn, t, α, β, μ, δ)
    end...)
end

function do_output(mpicomm, vtkdir, vtkstep, dgfvm, Q, Qe, model, testname)
    ## Name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    exactnames = statenames .* "_exact"

    writevtk(filename, Q, dgfvm, statenames, Qe, exactnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## Name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## Name of each of the ranks vtk files
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


function test_run(
    mpicomm,
    dim,
    fvmethod,
    polynomialorders,
    level,
    ArrayType,
    FT,
    vtkdir,
    direction,
)

    n_hd =
        dim == 2 ? SVector{3, FT}(1, 0, 0) :
        SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0)

    n_vd = dim == 2 ? SVector{3, FT}(0, 1, 0) : SVector{3, FT}(0, 0, 1)

    n_dg =
        dim == 2 ? SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0) :
        SVector{3, FT}(1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3))
    connectivity = dim == 2 ? :face : :full
    if direction isa EveryDirection
        ns = (n_hd, n_vd, n_dg)
    elseif direction isa HorizontalDirection
        ns = (n_hd,)
    elseif direction isa VerticalDirection
        ns = (n_vd,)
    end

    α = FT(1)
    β = FT(1 // 100)
    μ = FT(-1 // 2)
    δ = FT(1 // 10)

    # Grid/topology information
    base_num_elem = 4
    Ne = 2^(level - 1) * base_num_elem
    N = polynomialorders
    L = ntuple(j -> FT(j == dim ? 1 : N[1]) / 4, dim)
    brickrange = ntuple(j -> range(-L[j]; length = Ne + 1, stop = L[j]), dim)
    periodicity = ntuple(j -> false, dim)
    bc = ntuple(j -> (1, 2), dim)

    topl = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = periodicity,
        boundary = bc,
        connectivity = connectivity,
    )

    dt = (α / 4) * L[1] / (Ne * polynomialorders[1]^2)
    timeend = 1
    outputtime = timeend / 10
    @info "time step" dt

    @info @sprintf """Test parameters:
    FVM Reconstruction          = %s
    ArrayType                   = %s
    FloatType                   = %s
    Dimension                   = %s
    Direction                   = %s
    Horizontal polynomial order = %s
    Vertical polynomial order   = %s
      """ fvmethod ArrayType FT dim direction polynomialorders[1] polynomialorders[end]

    grid = SpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorders,
    )

    bcs = (InhomogeneousBC{0}(), InhomogeneousBC{1}())
    # Model being tested
    model = AdvectionDiffusion{dim}(
        Pseudo1D{ns, α, β, μ, δ}(),
        bcs,
        num_equations = length(ns),
    )

    # Main DG discretization
    dgfvm = DGFVModel(
        model,
        grid,
        fvmethod,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = direction,
    )

    # Initialize all relevant state arrays and create solvers
    Q = init_ode_state(dgfvm, FT(0))

    eng0 = norm(Q, dims = (1, 3))
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0[1]

    solver = LSRK54CarpenterKennedy(dgfvm, Q; dt = dt, t0 = 0)

    # Set up the information callback
    starttime = Ref(Dates.now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = Dates.now()
        else
            energy = norm(Q)
            @info @sprintf(
                """Update
                simtime = %.16e
                runtime = %s
                norm(Q) = %.16e""",
                gettime(solver),
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end
    callbacks = (cbinfo,)
    if ~isnothing(vtkdir)
        # Create vtk dir
        mkpath(vtkdir)

        vtkstep = 0
        # Output initial step
        do_output(
            mpicomm,
            vtkdir,
            vtkstep,
            dgfvm,
            Q,
            Q,
            model,
            "advection_diffusion",
        )

        # Setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dgfvm, gettime(solver))
            do_output(
                mpicomm,
                vtkdir,
                vtkstep,
                dgfvm,
                Q,
                Qe,
                model,
                "advection_diffusion",
            )
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, solver; timeend = timeend, callbacks = callbacks)

    # Reference solution
    engf = norm(Q, dims = (1, 3))
    Q_ref = init_ode_state(dgfvm, FT(timeend))

    engfe = norm(Q_ref, dims = (1, 3))
    errf = norm(Q_ref .- Q, dims = (1, 3))

    metrics = @. (engf, engf / eng0, engf - eng0, errf, errf / engfe)

    @info begin
        j = 1
        msg = "Finished\n"
        if direction isa HorizontalDirection || direction isa EveryDirection
            msg *= @sprintf """
            Horizontal field:
              norm(Q)                 = %.16e
              norm(Q) / norm(Q₀)      = %.16e
              norm(Q) - norm(Q₀)      = %.16e
              norm(Q - Qe)            = %.16e
              norm(Q - Qe) / norm(Qe) = %.16e
            """ ntuple(f -> metrics[f][j], 5)...
            j += 1
        end

        if direction isa VerticalDirection || direction isa EveryDirection
            msg *= @sprintf """
            Vertical field:
              norm(Q)                 = %.16e
              norm(Q) / norm(Q₀)      = %.16e
              norm(Q) - norm(Q₀)      = %.16e
              norm(Q - Qe)            = %.16e
              norm(Q - Qe) / norm(Qe) = %.16e
            """ ntuple(f -> metrics[f][j], 5)...
            j += 1
        end

        if direction isa EveryDirection
            msg *= @sprintf """
            Diagonal field:
              norm(Q)                 = %.16e
              norm(Q) / norm(Q₀)      = %.16e
              norm(Q) - norm(Q₀)      = %.16e
              norm(Q - Qe)            = %.16e
              norm(Q - Qe) / norm(Qe) = %.16e
            """ ntuple(f -> metrics[f][j], 5)...
        end
        msg
    end

    return Tuple(errf)
end

"""
    main()

Run this test problem
"""
function main()

    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    # Dictionary keys: dim, level, and FT
    expected_result = Dict()
    expected_result[2, 1, Float32, FVConstant()] =
        (2.3391991853713989e-02, 5.0738707184791565e-02, 4.1857458651065826e-02)
    expected_result[2, 2, Float32, FVConstant()] =
        (2.0331495907157660e-03, 3.3669386059045792e-02, 2.3904174566268921e-02)
    expected_result[2, 3, Float32, FVConstant()] =
        (2.6557327146292664e-05, 2.0002065226435661e-02, 1.2790882028639317e-02)

    expected_result[2, 1, Float32, FVLinear()] =
        (2.3391995579004288e-02, 4.9536284059286118e-02, 3.5911198705434799e-02)
    expected_result[2, 2, Float32, FVLinear()] =
        (2.0331430714577436e-03, 2.2621221840381622e-02, 1.5531544573605061e-02)
    expected_result[2, 3, Float32, FVLinear()] =
        (2.6568108296487480e-05, 7.5304862111806870e-03, 6.4526572823524475e-03)

    expected_result[3, 1, Float32, FVConstant()] =
        (8.7377587333321571e-03, 7.1755334734916687e-02, 4.3733172118663788e-02)
    expected_result[3, 2, Float32, FVConstant()] =
        (6.2517996411770582e-04, 4.7615684568881989e-02, 2.3986011743545532e-02)
    expected_result[3, 3, Float32, FVConstant()] =
        (3.5004395613214001e-05, 2.8287241235375404e-02, 1.2639496475458145e-02)

    expected_result[3, 1, Float32, FVLinear()] =
        (8.7377615272998810e-03, 7.0054858922958374e-02, 3.7571556866168976e-02)
    expected_result[3, 2, Float32, FVLinear()] =
        (6.2518107006326318e-04, 3.1991228461265564e-02, 1.6405479982495308e-02)
    expected_result[3, 3, Float32, FVLinear()] =
        (3.5004966775886714e-05, 1.0649790056049824e-02, 7.1499347686767578e-03)

    expected_result[2, 1, Float64, FVConstant()] =
        (2.3391871809628567e-02, 5.0738761087541523e-02, 4.1857480220018319e-02)
    expected_result[2, 2, Float64, FVConstant()] =
        (2.0332783913617892e-03, 3.3669399006499491e-02, 2.3904204301839819e-02)
    expected_result[2, 3, Float64, FVConstant()] =
        (2.6572168347086839e-05, 2.0002110124502395e-02, 1.2790993871268752e-02)
    expected_result[2, 4, Float64, FVConstant()] =
        (1.9890000550154039e-07, 1.0929144069594809e-02, 6.5110938897763263e-03)

    expected_result[2, 1, Float64, FVLinear()] =
        (2.3391871809628550e-02, 4.9536356061135857e-02, 3.5911213637569099e-02)
    expected_result[2, 2, Float64, FVLinear()] =
        (2.0332783913618278e-03, 2.2621252826152356e-02, 1.5531565558700888e-02)
    expected_result[2, 3, Float64, FVLinear()] =
        (2.6572168347111800e-05, 7.5305291439555161e-03, 6.4526740563561544e-03)
    expected_result[2, 4, Float64, FVLinear()] =
        (1.9890000549995218e-07, 2.5302226025389427e-03, 2.7792905025059711e-03)

    expected_result[3, 1, Float64, FVConstant()] =
        (8.7378337431297422e-03, 7.1755444068009461e-02, 4.3733196512722658e-02)
    expected_result[3, 2, Float64, FVConstant()] =
        (6.2510740807095622e-04, 4.7615720711942776e-02, 2.3986017606198881e-02)
    expected_result[3, 3, Float64, FVConstant()] =
        (3.4995405318038341e-05, 2.8287255414151377e-02, 1.2639742577376042e-02)
    expected_result[3, 4, Float64, FVConstant()] =
        (1.4362091045094841e-06, 1.5456143768350493e-02, 6.3677406803847331e-03)

    expected_result[3, 1, Float64, FVLinear()] =
        (8.7378337431297439e-03, 7.0054986572200995e-02, 3.7571599264936015e-02)
    expected_result[3, 2, Float64, FVLinear()] =
        (6.2510740807095286e-04, 3.1991282544615307e-02, 1.6405489038457288e-02)
    expected_result[3, 3, Float64, FVLinear()] =
        (3.4995405318035868e-05, 1.0649776447227678e-02, 7.1502921502446283e-03)
    expected_result[3, 4, Float64, FVLinear()] =
        (1.4362091045110612e-06, 3.5782751203335653e-03, 3.1186151510318319e-03)

    @testset "Variable degree DG: advection diffusion model" begin
        for FT in (Float32, Float64)
            for direction in
                (EveryDirection(), HorizontalDirection(), VerticalDirection())
                numlevels =
                    integration_testing ||
                    ClimateMachine.Settings.integration_testing ?
                    (FT == Float64 ? 4 : 3) : 1
                for dim in 2:3
                    for fvmethod in (FVConstant(), FVLinear(), FVLinear{3}())
                        polynomialorders = (4, 0)
                        result = Dict()
                        for level in 1:numlevels
                            vtkdir =
                                output ?
                                "vtk_advection" *
                                "_poly$(polynomialorders)" *
                                "_dim$(dim)_$(ArrayType)_$(FT)" *
                                "_fvmethod$(fvmethod)" *
                                "_level$(level)" :
                                nothing
                            result[level] = test_run(
                                mpicomm,
                                dim,
                                fvmethod,
                                polynomialorders,
                                level,
                                ArrayType,
                                FT,
                                vtkdir,
                                direction,
                            )
                            fv_key =
                                fvmethod isa FVLinear ? FVLinear() : fvmethod
                            if direction isa EveryDirection
                                @test all(
                                    result[level] .≈
                                    FT.(expected_result[
                                        dim,
                                        level,
                                        FT,
                                        fv_key,
                                    ]),
                                )
                            elseif direction isa HorizontalDirection
                                @test result[level][1] .≈ FT.(expected_result[
                                    dim,
                                    level,
                                    FT,
                                    fv_key,
                                ])[1]
                            elseif direction isa VerticalDirection
                                @test result[level][1] .≈ FT.(expected_result[
                                    dim,
                                    level,
                                    FT,
                                    fv_key,
                                ])[2]
                            end
                        end
                        @info begin
                            msg = ""
                            for l in 1:(numlevels - 1)
                                rate = @. log2(result[l]) - log2(result[l + 1])
                                msg *= @sprintf("\n  rates for level %d", l)
                                j = 1
                                if direction isa HorizontalDirection ||
                                   direction isa EveryDirection
                                    msg *=
                                        @sprintf(", Horizontal = %e", rate[j])
                                    j += 1
                                end
                                if direction isa VerticalDirection ||
                                   direction isa EveryDirection
                                    msg *= @sprintf(", Vertical = %e", rate[j])
                                    j += 1
                                end
                                if direction isa EveryDirection
                                    msg *= @sprintf(", Diagonal = %e", rate[j])
                                end
                                msg *= "\n"
                            end
                            msg
                        end
                    end
                end
            end
        end
    end
end

main()
