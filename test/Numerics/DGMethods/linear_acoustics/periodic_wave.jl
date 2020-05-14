using CLIMA
using CLIMA.ConfigTypes
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods: LocalGeometry
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.ODESolvers
using CLIMA.GeneralizedMinimalResidualSolver
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MPIStateArrays: MPIStateArray, euclidean_distance
using CLIMA.MoistThermodynamics
using CLIMA.Atmos
using CLIMA.Atmos: ReferenceState, vars_state_conservative
using CLIMA.VariableTemplates: @vars, Vars, flattenednames
import CLIMA.Atmos: atmos_init_aux!, vars_state_auxiliary

using CLIMAParameters
using CLIMAParameters.Planet: cp_d, cv_d, R_d, T_0
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
    CLIMA.init()
    ArrayType = CLIMA.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    numlevels = 4
    base_num_elem = 5

    @testset "$(@__FILE__)" begin
        for FT in (Float64,), dims in 2
                @info @sprintf """Configuration
                                  ArrayType = %s
                                  FT    = %s
                                  dims      = %d
                                  """ ArrayType "$FT" dims

                setup = WaveSetup{FT}()
                errors = Vector{FT}(undef, numlevels)

                for level in 1:numlevels
                    numelems =
                        ntuple(dim -> 2^(level - 1) * base_num_elem, dims)
                    errors[level] = run(
                        mpicomm,
                        ArrayType,
                        polynomialorder,
                        numelems,
                        setup,
                        FT,
                        dims,
                        level,
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

function run(
    mpicomm,
    ArrayType,
    polynomialorder,
    numelems,
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

    topology = StackedBrickTopology(
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
        ref_state = WaveReferenceState{FT}(setup),
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = nothing,
        boundarycondition = (),
        init_state_conservative = wave_initialcondition!,
    )

    linear_model = AtmosAcousticLinearModel(model)

    dg = DGModel(
        model,
        grid,
        CentralNumericalFluxFirstOrder(),
        #RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    dg_linear = DGModel(
        linear_model,
        grid,
        #RusanovNumericalFlux(),
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        state_auxiliary = dg.state_auxiliary,
    )

    #timeend = FT(0.0001)
    timeend = FT(0.001)

    # determine the time step
    dx = min_node_distance(grid)
    #cfl = FT(0.6) # 54
    #cfl = FT(2.6) # 144
    #cfl = FT(1.5) # 144
    #cfl = FT(7.5)
    #cfl = FT(2.6)
    cfl = FT(0.5)
    dt = cfl * dx / setup.c
    nsteps = ceil(Int, timeend / dt)
    dt = timeend / nsteps

    Q = init_ode_state(dg, FT(0))


    schur_model = AtmosAcousticLinearSchurComplement()
    schur_dg = SchurDGModel(schur_model, linear_model, grid, dg_linear.state_auxiliary)

    #linearsolver = GeneralizedMinimalResidual(Q; M = 60, rtol = 1e-6)
    linearsolver = GeneralizedMinimalResidual(schur_dg.schur_state; M = 60, rtol = 1e-6)

    #ode_solver = LSRK54CarpenterKennedy(dg_linear, Q; dt = dt, t0 = 0)
    #ode_solver = LSRK144NiegemannDiehlBusch(dg_linear, Q; dt = dt, t0 = 0)
    ode_solver = BackwardEulerODESolver(dg_linear, Q,
                                        LinearBackwardEulerSolver(linearsolver;
                                                                  isadjustable=true);
                                        dt = dt, t0 = 0, schur=schur_dg)

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
            "vtk_wave" *
            "_poly$(polynomialorder)_dims$(dims)_$(ArrayType)_$(FT)_level$(level)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model)

        # setup the output callback
        outputtime = timeend / 20
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(ode_solver))
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, ode_solver; timeend = timeend, callbacks = callbacks, adjustfinalstep = true)

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

Base.@kwdef struct WaveSetup{FT}
    γ::FT = cp_d(param_set) / cv_d(param_set)
    p∞::FT = 10^5
    ρ∞::FT = 1
    E∞::FT = (p∞ - R_d(param_set) * ρ∞ * T_0(param_set)) / (γ - 1) 
    T∞::FT = T_0(param_set) + E∞ / cv_d(param_set)
    c::FT = sqrt(γ * p∞ / ρ∞)
    domain_halflength::FT = 1 // 20
end

struct WaveReferenceState{FT} <: ReferenceState
    setup::WaveSetup{FT}
end
vars_state_auxiliary(::WaveReferenceState, FT) =
    @vars(ρ::FT, ρe::FT, p::FT, T::FT)
function atmos_init_aux!(
    m::WaveReferenceState,
    atmos::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    setup = m.setup

    aux.ref_state.ρ = setup.ρ∞
    aux.ref_state.p = setup.p∞
    aux.ref_state.ρe = setup.E∞
    aux.ref_state.T = setup.T∞
end

function wave_initialcondition!(bl, state, aux, coords, t, args...)
    setup = bl.ref_state.setup
    FT = eltype(state)
    x = MVector(coords)

    ρ∞ = setup.ρ∞
    p∞ = setup.p∞
    E∞ = setup.E∞
    γ = setup.γ
    L = setup.domain_halflength
    c = setup.c
    
    @inbounds ρu_x = sin(2π * (x[1] - c * t) / L)

    state.ρ = ρ∞  + ρu_x  / c
    state.ρu = SVector(ρu_x, 0, 0)
    state.ρe = E∞ + ρu_x * (E∞ + p∞) / (ρ∞ * c)
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    Qe,
    model,
    testname = "wave",
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
