using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods: nodal_update_auxiliary_state!
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
import ClimateMachine.DGmethods: boundary_state!

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

# This is a setup similar to the one presented in:
# @article{WILLIAMSON1992211,
#   title = {A standard test set for numerical approximations to the
#            shallow water equations in spherical geometry},
#   journal = {Journal of Computational Physics},
#   volume = {102},
#   number = {1},
#   pages = {211 - 224},
#   year = {1992},
#   doi = {10.1016/S0021-9991(05)80016-6},
#   url = {https://doi.org/10.1016/S0021-9991(05)80016-6},
# }
struct SolidBodyRotation <: AdvectionDiffusionProblem end
function init_velocity_diffusion!(
    ::SolidBodyRotation,
    aux::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    λ = longitude(SphericalOrientation(), aux)
    φ = latitude(SphericalOrientation(), aux)
    r = norm(geom.coord)

    uλ = 2 * FT(π) * cos(φ) * r
    uφ = 0
    aux.u = SVector(
        -uλ * sin(λ) - uφ * cos(λ) * sin(φ),
        +uλ * cos(λ) - uφ * sin(λ) * sin(φ),
        +uφ * cos(φ),
    )
end
function initial_condition!(::SolidBodyRotation, state, aux, x, t)
    λ = longitude(SphericalOrientation(), aux)
    φ = latitude(SphericalOrientation(), aux)
    state.ρ = exp(-((3λ)^2 + (3φ)^2))
end
finaltime(::SolidBodyRotation) = 1
u_scale(::SolidBodyRotation) = 2π

# This is a setup similar to the one presented in:
# @Article{gmd-5-887-2012,
# AUTHOR = {Lauritzen, P. H. and Skamarock, W. C. and Prather, M. J.
#           and Taylor, M. A.},
# TITLE = {A standard test case suite for two-dimensional linear
#          transport on the sphere},
# JOURNAL = {Geoscientific Model Development},
# VOLUME = {5},
# YEAR = {2012},
# NUMBER = {3},
# PAGES = {887--901},
# URL = {https://www.geosci-model-dev.net/5/887/2012/},
# DOI = {10.5194/gmd-5-887-2012}
# }
struct ReversingDeformationalFlow <: AdvectionDiffusionProblem end
init_velocity_diffusion!(
    ::ReversingDeformationalFlow,
    aux::Vars,
    geom::LocalGeometry,
) = nothing
function initial_condition!(::ReversingDeformationalFlow, state, aux, coord, t)
    x, y, z = aux.coord
    r = norm(aux.coord)
    h_max = 0.95
    b = 5
    state.ρ = 0
    for (λ, φ) in ((5π / 6, 0), (7π / 6, 0))
        xi = r * cos(φ) * cos(λ)
        yi = r * cos(φ) * sin(λ)
        zi = r * sin(φ)
        state.ρ += h_max * exp(-b * ((x - xi)^2 + (y - yi)^2 + (z - zi)^2))
    end
end
has_variable_coefficients(::ReversingDeformationalFlow) = true
function update_velocity_diffusion!(
    ::ReversingDeformationalFlow,
    ::AdvectionDiffusion,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(aux)
    λ = longitude(SphericalOrientation(), aux)
    φ = latitude(SphericalOrientation(), aux)
    r = norm(aux.coord)
    T = FT(5)
    λp = λ - FT(2π) * t / T
    uλ =
        10 * r / T * sin(λp)^2 * sin(2φ) * cos(FT(π) * t / T) +
        FT(2π) * r / T * cos(φ)
    uφ = 10 * r / T * sin(2λp) * cos(φ) * cos(FT(π) * t / T)
    aux.u = SVector(
        -uλ * sin(λ) - uφ * cos(λ) * sin(φ),
        +uλ * cos(λ) - uφ * sin(λ) * sin(φ),
        +uφ * cos(φ),
    )
end
u_scale(::ReversingDeformationalFlow) = 2.9
finaltime(::ReversingDeformationalFlow) = 5

function advective_courant(
    m::AdvectionDiffusion,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    direction,
)
    return Δt * norm(aux.u) / Δx
end

function boundary_state!(
    ::RusanovNumericalFlux,
    ::AdvectionDiffusion,
    stateP::Vars,
    auxP::Vars,
    nM,
    stateM::Vars,
    auxM::Vars,
    bctype,
    t,
    _...,
)
    auxP.u = -auxM.u
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname)
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

    ## generate the pvtu file for these vtk files
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
        meshwarp = cubedshellwarp,
    )

    dx = min_node_distance(grid, HorizontalDirection())
    dt = FT(cfl * dx / u_scale(problem()))
    dt = outputtime / ceil(Int64, outputtime / dt)

    model = AdvectionDiffusion{3, false, true}(problem())
    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))

    odesolver = explicit_method(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
    problem   = %s
    method    = %s
    time step = %.16e
    norm(Q₀)  = %.16e""" problem explicit_method dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            @info @sprintf(
                """Update
                simtime = %.16e
                runtime = %s
                norm(Q) = %.16e""",
                gettime(odesolver),
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end
    cbcfl = EveryXSimulationSteps(10) do
        dt = ODESolvers.getdt(odesolver)
        cfl = DGmethods.courant(
            advective_courant,
            dg,
            model,
            Q,
            dt,
            HorizontalDirection(),
        )
        @info @sprintf(
            """Courant number
            simtime = %.16e
            courant = %.16e""",
            gettime(odesolver),
            cfl
        )
    end
    callbacks = (cbinfo,)
    if ~isnothing(vtkdir)
        # create vtk dir
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model, "advection_sphere")

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(odesolver))
            do_output(
                mpicomm,
                vtkdir,
                vtkstep,
                dg,
                Q,
                Qe,
                model,
                "advection_sphere",
            )
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, odesolver; timeend = timeend, callbacks = callbacks)

    # Print some end of the simulation information
    engf = norm(Q)
    Qe = init_ode_state(dg, FT(timeend))

    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    Δmass = abs(weightedsum(Q) - weightedsum(Qe)) / weightedsum(Qe)
    @info @sprintf """Finished
    Δmass                   = %.16e
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ Δmass engf engf / eng0 engf - eng0 errf errf / engfe
    return errf, Δmass
end

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    base_num_elem = 2

    max_cfl = Dict(
        LSRK144NiegemannDiehlBusch => 5.0,
        SSPRK33ShuOsher => 1.0,
        SSPRK34SpiteriRuuth => 1.5,
    )

    expected_result = Dict()

    expected_result[SolidBodyRotation, LSRK144NiegemannDiehlBusch, 1] =
        1.3199024557832748e-01
    expected_result[SolidBodyRotation, LSRK144NiegemannDiehlBusch, 2] =
        1.9868931633120656e-02
    expected_result[SolidBodyRotation, LSRK144NiegemannDiehlBusch, 3] =
        1.4052110916915061e-03
    expected_result[SolidBodyRotation, LSRK144NiegemannDiehlBusch, 4] =
        9.0193766298676310e-05

    expected_result[SolidBodyRotation, SSPRK33ShuOsher, 1] =
        1.1055145388897809e-01
    expected_result[SolidBodyRotation, SSPRK33ShuOsher, 2] =
        1.5510740467668628e-02
    expected_result[SolidBodyRotation, SSPRK33ShuOsher, 3] =
        1.8629481690361454e-03
    expected_result[SolidBodyRotation, SSPRK33ShuOsher, 4] =
        2.3567040048588889e-04

    expected_result[SolidBodyRotation, SSPRK34SpiteriRuuth, 1] =
        1.2641959922001456e-01
    expected_result[SolidBodyRotation, SSPRK34SpiteriRuuth, 2] =
        2.2780375948714751e-02
    expected_result[SolidBodyRotation, SSPRK34SpiteriRuuth, 3] =
        3.1274951764826459e-03
    expected_result[SolidBodyRotation, SSPRK34SpiteriRuuth, 4] =
        3.9734060514021565e-04

    expected_result[ReversingDeformationalFlow, LSRK144NiegemannDiehlBusch, 1] =
        5.5387951598735408e-01
    expected_result[ReversingDeformationalFlow, LSRK144NiegemannDiehlBusch, 2] =
        3.7610388138383732e-01
    expected_result[ReversingDeformationalFlow, LSRK144NiegemannDiehlBusch, 3] =
        1.7823508719111605e-01
    expected_result[ReversingDeformationalFlow, LSRK144NiegemannDiehlBusch, 4] =
        3.8639493470255713e-02

    expected_result[ReversingDeformationalFlow, SSPRK33ShuOsher, 1] =
        5.5353962032596349e-01
    expected_result[ReversingDeformationalFlow, SSPRK33ShuOsher, 2] =
        3.7645487928038762e-01
    expected_result[ReversingDeformationalFlow, SSPRK33ShuOsher, 3] =
        1.7823263736245307e-01
    expected_result[ReversingDeformationalFlow, SSPRK33ShuOsher, 4] =
        3.8605903366230925e-02

    expected_result[ReversingDeformationalFlow, SSPRK34SpiteriRuuth, 1] =
        5.5404045660832824e-01
    expected_result[ReversingDeformationalFlow, SSPRK34SpiteriRuuth, 2] =
        3.7788858038003154e-01
    expected_result[ReversingDeformationalFlow, SSPRK34SpiteriRuuth, 3] =
        1.8007113931230376e-01
    expected_result[ReversingDeformationalFlow, SSPRK34SpiteriRuuth, 4] =
        3.9941331660544775e-02

    numlevels =
        integration_testing || ClimateMachine.Settings.integration_testing ? 4 :
        1
    @testset "$(@__FILE__)" begin
        for FT in (Float64,)
            for problem in (SolidBodyRotation, ReversingDeformationalFlow)
                for explicit_method in (
                    LSRK144NiegemannDiehlBusch,
                    SSPRK33ShuOsher,
                    SSPRK34SpiteriRuuth,
                )
                    cfl = max_cfl[explicit_method]
                    result = zeros(FT, numlevels)
                    for l in 1:numlevels
                        numelems_horizontal = 2^(l - 1) * base_num_elem
                        numelems_vertical = 1

                        topl = StackedCubedSphereTopology(
                            mpicomm,
                            numelems_horizontal,
                            range(
                                FT(1),
                                stop = 2,
                                length = numelems_vertical + 1,
                            ),
                        )

                        timeend = finaltime(problem())
                        outputtime = timeend

                        @info (ArrayType, FT)
                        vtkdir = output ?
                            "vtk_advection_sphere" *
                        "_$problem" *
                        "_$explicit_method" *
                        "_poly$(polynomialorder)" *
                        "_$(ArrayType)_$(FT)" *
                        "_level$(l)" :
                            nothing

                        result[l], Δmass = run(
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
                              FT(expected_result[problem, explicit_method, l])
                        @test Δmass <= FT(5e-14)
                    end
                    @info begin
                        msg = ""
                        for l in 1:(numlevels - 1)
                            rate = log2(result[l]) - log2(result[l + 1])
                            msg *= @sprintf(
                                "\n  rate for level %d = %e\n",
                                l,
                                rate
                            )
                        end
                        msg
                    end
                end
            end
        end
    end
end
