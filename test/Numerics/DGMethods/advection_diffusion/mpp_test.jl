using Test
using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.BalanceLaws: update_auxiliary_state!
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude
using ClimateMachine.Orientations
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks: EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
import ClimateMachine.BalanceLaws: boundary_state!
using ClimateMachine.Mesh.Filters

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

#
# Solid Rotation Test
#
struct SolidRotation{nstate, dim} <: AdvectionDiffusionProblem end
number_multistates(::SolidRotation{nstate}) where {nstate} = nstate

function init_velocity_diffusion!(
    prob::SolidRotation{nstate, dim},
    aux::Vars,
    _...,
) where {nstate, dim}
    FT = eltype(aux)
    x, y, z = aux.coord
    r = SVector(x, y, z)
    ω = dim == 2 ? SVector(0, 0, 1) : SVector(1, 1, 1) / sqrt(FT(3))
    aux.u = cross(ω, r)
end

function initial_condition!(
    p::SolidRotation{_nstate, dim},
    state,
    aux,
    coord,
    t,
) where {_nstate, dim}
    FT = eltype(state)
    x, y, z = aux.coord
    unpack_val(::Val{k}) where {k} = k
    nstate = _nstate # XXX: Why is this needed for the GPU?
    @unroll_map(nstate) do kv
        k = unpack_val(kv)
        θ = (k - 1) * 2 * FT(π) / nstate
        x0, y0 = cos(θ) / 3, sin(θ) / 3
        z0 = dim == 2 ? 0 : cos(θ) / 3
        τ = hypot(x - x0, y - y0, z - z0)
        state.ρ[kv].val = exp(-(16τ) .^ 2)
    end
end
finaltime(p::SolidRotation) = 2π
u_scale(p::SolidRotation) = 1

#
# Swirl flow test
#
Base.@kwdef struct SwirlingFlow{FT, nstate, dim} <: AdvectionDiffusionProblem
    period::FT = 5
end
number_multistates(::SwirlingFlow{FT, nstate}) where {FT, nstate} = nstate

init_velocity_diffusion!(::SwirlingFlow, aux::Vars, geom::LocalGeometry) =
    nothing

cosbell(τ, q) = τ ≤ 1 ? ((1 + cospi(τ)) / 2)^q : zero(τ)

function initial_condition!(
    p::SwirlingFlow{_FT, _nstate, dim},
    state,
    aux,
    coord,
    t,
) where {_FT, _nstate, dim}
    FT = eltype(state)
    x, y, z = aux.coord
    unpack_val(::Val{k}) where {k} = k
    nstate = _nstate # XXX: Why is this needed for the GPU?
    @unroll_map(nstate) do kv
        k = unpack_val(kv)
        θ = (k - 1) * 2 * FT(π) / nstate + FT(π) / 4
        x0, y0 = sqrt(FT(2)) * cos(θ) / 2, sqrt(FT(2)) * sin(θ) / 2
        z0 = dim == 2 ? 0 : sqrt(FT(2)) * cos(θ) / 2
        τ = hypot(x - x0, y - y0, z - z0)
        state.ρ[kv].val = cosbell(2τ, 3)
    end
end

has_variable_coefficients(::SwirlingFlow) = true
function update_velocity_diffusion!(
    p::SwirlingFlow{FT, _nstate, dim},
    ::AdvectionDiffusion,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, _nstate, dim}
    x, y, z = aux.coord
    sx, cx = sinpi((x + 1) / 2), cospi((x + 1) / 2)
    sy, cy = sinpi((y + 1) / 2), cospi((y + 1) / 2)
    ct = cospi(t / p.period)


    u = 4 * sx^2 * sy * cy * ct
    v = -4 * sy^2 * sx * cx * ct
    w = FT(0)
    if dim == 3
        sz, cz = sinpi((z + 1) / 2), cospi((z + 1) / 2)

        u /= 2
        v /= 2
        w /= 2

        u += 2 * sx^2 * sz * cz * ct
        w += -2 * sz^2 * sx * cx * ct
    end
    aux.u = SVector(u, v, w)
end

finaltime(p::SwirlingFlow) = p.period

u_scale(p::SwirlingFlow) = sqrt(2) #TODO: Is this right?

function do_output(mpicomm, vtkdir, vtkstep, dg, state_prognostic, model)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/mpirank%04d_step%04d",
        vtkdir,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(
        model,
        Prognostic(),
        eltype(state_prognostic),
    ))

    writevtk(filename, state_prognostic, dg, statenames)

    ## generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/step%04d", vtkdir, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("mpirank%04d_step%04d", i - 1, vtkstep)
        end

        writepvtu(
            pvtuprefix,
            prefixes,
            (statenames...,),
            eltype(state_prognostic),
        )

        @info "Done writing VTK: $pvtuprefix"
    end
end

function run(
    mpicomm,
    ArrayType,
    topl,
    problem,
    explicit_method,
    cfl,
    N,
    timeend,
    FT,
    vtkdir,
    outputtime,
)
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    dx = min_node_distance(grid, HorizontalDirection())
    dt = FT(cfl * dx / u_scale(problem())) / 2
    dt = outputtime / ceil(Int64, outputtime / dt)

    model = AdvectionDiffusion{2, false, true}(problem())
    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    state_prognostic = init_ode_state(dg, FT(0))

    # These are the variables we will apply mpp to
    mpp_vars = (1, 3)
    num_mpp = length(mpp_vars)

    # Compute the mass and volume of the original problem

    ρ_mass = weightedsum(state_prognostic, mpp_vars)
    dg_vol = sum(grid.vgeo[:, Grids._M, grid.topology.realelems])
    dg_vol = MPI.Allreduce(dg_vol, (+), state_prognostic.mpicomm)

    # Initialize the MPP
    mppdata = DGMethods.mpp_initialize(dg, state_prognostic, mpp_vars)

    # Check that the results match the original version
    mpp_mass = weightedsum(mppdata.state)

    mpp_vol = sum(Array(mppdata.aux.vol))
    mpp_vol = MPI.Allreduce(mpp_vol, (+), state_prognostic.mpicomm)

    @test mpp_mass ≈ ρ_mass
    @test mpp_vol ≈ dg_vol

    odesolver =
        MPPSolver(mpp_vars, explicit_method, dg, state_prognostic; dt = dt)
    test_output(vtkstep) = do_output(
        state_prognostic.mpicomm,
        vtkdir,
        vtkstep,
        dg,
        state_prognostic,
        model,
    )

    output && mkpath(vtkdir)
    output && test_output(0)

    nsteps_out = ceil(Int, outputtime / dt)
    vtk_step = 0
    cb_output = EveryXSimulationSteps(nsteps_out) do
        vtk_step += 1
        output && test_output(vtk_step)
        nothing
    end
    cb_tmar = EveryXSimulationSteps(1) do
        Filters.apply!(state_prognostic, mpp_vars, dg.grid, TMARFilter())
        nothing
    end

    solve!(
        state_prognostic,
        odesolver;
        timeend = timeend,
        callbacks = (cb_tmar, cb_output),
    )

    mpp_mass = weightedsum(mppdata.state)
    @test ρ_mass ≈ mpp_mass
    for v in mpp_vars
        @test minimum(state_prognostic.data[:, v, :]) ≥ 0
    end

    state_prognostic_exact = init_ode_state(dg, FT(timeend))
    errf = euclidean_distance(state_prognostic, state_prognostic_exact)

    @info @sprintf """Finished
    L2 error   = %.16e
    """ errf

    return errf
end

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    base_num_elem = 6

    max_cfl = Dict(LSRK144NiegemannDiehlBusch => 5.0)

    expected = Dict()
    expected[SolidRotation{4, 2}, Float64, LSRK144NiegemannDiehlBusch] = [
        7.8082571458694766e-02
        4.7937947266619381e-02
        3.8490418523441879e-03
        5.7601322144544763e-05
    ]
    expected[SolidRotation{4, 3}, Float64, LSRK144NiegemannDiehlBusch] = [
        2.2886411087077153e-02
        1.5029461876191969e-02
        9.8974233334442633e-04
        1.6618079351940254e-05
    ]
    expected[SwirlingFlow{Float64, 4, 2}, Float64, LSRK144NiegemannDiehlBusch] =
        [
            3.4845780245634467e-01
            2.4488138040622157e-01
            1.0526862283423724e-01
            8.2630448523354928e-03
        ]
    expected[SwirlingFlow{Float64, 4, 3}, Float64, LSRK144NiegemannDiehlBusch] =
        [
            1.3101521857715073e-01
            8.0922061851209467e-02
            2.5513904210978679e-02
            2.7316678569712961e-03
        ]


    numlevels =
        integration_testing || ClimateMachine.Settings.integration_testing ? 4 :
        1
    @testset "$(@__FILE__)" begin
        for FT in (Float64,)
            for dim in (2, 3)
                for problem in (SolidRotation{4, dim}, SwirlingFlow{FT, 4, dim})
                    for explicit_method in (LSRK144NiegemannDiehlBusch,)
                        @testset "$(problem), $(explicit_method)" begin
                            cfl = max_cfl[explicit_method]
                            result = zeros(FT, numlevels)
                            for l in 1:numlevels
                                Ne = 2^(l - 1) * base_num_elem

                                brickrange = (
                                    range(FT(-1); length = Ne + 1,
                                        stop = 1),
                                    range(FT(-1); length = Ne + 1,
                                        stop = 1),
                                    range(FT(-1); length = Ne + 1,
                                        stop = 1),
                                )

                                topl = BrickTopology(
                                    mpicomm,
                                    brickrange[1:dim],
                                    boundary = ntuple(d -> (3, 3), dim),
                                )

                                timeend = finaltime(problem())
                                outputtime = timeend / 10

                                @info (ArrayType, FT)
                                vtkdir =
                                    "vtk_mpp_advection" *
                                    "_$problem" *
                                    "_dim$(dim)" *
                                    "_$explicit_method" *
                                    "_poly$(polynomialorder)" *
                                    "_$(ArrayType)_$(FT)" *
                                    "_level$(l)"

                                result[l] = run(
                                    mpicomm,
                                    ArrayType,
                                    topl,
                                    problem,
                                    explicit_method,
                                    cfl,
                                    polynomialorder,
                                    timeend,
                                    FT,
                                    vtkdir,
                                    outputtime,
                                )
                                @test result[l] ≈
                                      expected[problem, FT, explicit_method][l]
                                if l > 1
                                    @info begin
                                        rate =
                                            log2(result[l - 1]) -
                                            log2(result[l])
                                        @sprintf(
                                            "\n  rate for level %d = %f\n",
                                            l,
                                            rate
                                        )
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
