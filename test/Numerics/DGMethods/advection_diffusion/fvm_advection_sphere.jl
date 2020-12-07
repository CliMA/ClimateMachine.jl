using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.BalanceLaws: update_auxiliary_state!
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods.FVReconstructions: FVConstant, FVLinear
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude
using ClimateMachine.Orientations
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
import ClimateMachine.BalanceLaws: boundary_state!
using CLIMAParameters.Planet: planet_radius

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")


# This is a setup similar to the one presented in [Williamson1992](@cite)
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
    aux.advection.u = SVector(
        -uλ * sin(λ) - uφ * cos(λ) * sin(φ),
        +uλ * cos(λ) - uφ * sin(λ) * sin(φ),
        +uφ * cos(φ),
    )
end
function initial_condition!(::SolidBodyRotation, state, aux, localgeo, t)
    λ = longitude(SphericalOrientation(), aux)
    φ = latitude(SphericalOrientation(), aux)
    state.ρ = exp(-((3λ)^2 + (3φ)^2))
end
finaltime(::SolidBodyRotation) = 1
u_scale(::SolidBodyRotation) = 2π

"""
This is the Divergent flow with smooth initial condition test case, the Case 4 in

@article{nair2010class,
  title={A class of deformational flow test cases for linear transport problems on the sphere},
  author={Nair, Ramachandran D and Lauritzen, Peter H},
  journal={Journal of Computational Physics},
  volume={229},
  number={23},
  pages={8868--8887},
  year={2010},
  publisher={Elsevier}
}

"""
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
        state.ρ +=
            h_max * exp(-b * ((x - xi)^2 + (y - yi)^2 + (z - zi)^2) / r^2)
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
    aux.advection.u = SVector(
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
    return Δt * norm(aux.advection.u) / Δx
end

function boundary_state!(
    ::RusanovNumericalFlux,
    bctype,
    ::AdvectionDiffusion,
    stateP::Vars,
    auxP::Vars,
    nM,
    stateM::Vars,
    auxM::Vars,
    t,
    _...,
)
    auxP.advection.u = -auxM.advection.u
end

function do_output(mpicomm, vtkdir, vtkstep, dgfvm, Q, Qe, model, testname)
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

    writevtk(filename, Q, dgfvm, statenames, Qe, exactnames)

    ## generate the pvtu file for these vtk files
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

function test_run(
    mpicomm,
    ArrayType,
    vert_range,
    topl,
    problem,
    explicit_method,
    fvmethod,
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
        polynomialorder = (N, 0),
        meshwarp = cubedshellwarp,
    )

    dx = min_node_distance(grid, HorizontalDirection())
    dt = FT(cfl * dx / (vert_range[2] * u_scale(problem())) / N)
    dt = outputtime / ceil(Int64, outputtime / dt)

    model = AdvectionDiffusion{3}(problem(), diffusion = false)
    dgfvm = DGFVModel(
        model,
        grid,
        fvmethod,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dgfvm, FT(0))

    odesolver = explicit_method(dgfvm, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
    problem           = %s
    ArrayType         = %s
    FV Reconstruction = %s
    method            = %s
    time step         = %.16e
    norm(Q₀)          = %.16e""" problem ArrayType fvmethod explicit_method dt eng0

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
        cfl = DGMethods.courant(
            advective_courant,
            dgfvm,
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
        do_output(
            mpicomm,
            vtkdir,
            vtkstep,
            dgfvm,
            Q,
            Q,
            model,
            "fvm_advection_sphere",
        )

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dgfvm, gettime(odesolver))
            do_output(
                mpicomm,
                vtkdir,
                vtkstep,
                dgfvm,
                Q,
                Qe,
                model,
                "fvm_advection_sphere",
            )
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, odesolver; timeend = timeend, callbacks = callbacks)

    # Print some end of the simulation information
    engf = norm(Q)
    Qe = init_ode_state(dgfvm, FT(timeend))

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
    base_num_elem = 3

    max_cfl = Dict(LSRK144NiegemannDiehlBusch => 5.0)

    expected_result = Dict()

    expected_result[SolidBodyRotation, 1, FVConstant] = 1.6249678501611961e+07
    expected_result[SolidBodyRotation, 2, FVConstant] = 7.2020207047738554e+05
    expected_result[SolidBodyRotation, 3, FVConstant] = 5.2452627634607365e+04
    expected_result[SolidBodyRotation, 4, FVConstant] = 3.1132401693922270e+03

    expected_result[SolidBodyRotation, 1, FVLinear] = 1.6650018586769039e+07
    expected_result[SolidBodyRotation, 2, FVLinear] = 7.2518593691652094e+05
    expected_result[SolidBodyRotation, 3, FVLinear] = 5.2473203187596591e+04
    expected_result[SolidBodyRotation, 4, FVLinear] = 3.1133127473480104e+03

    expected_result[ReversingDeformationalFlow, 1, FVConstant] =
        2.0097347028034222e+08
    expected_result[ReversingDeformationalFlow, 2, FVConstant] =
        7.0092390520693690e+07
    expected_result[ReversingDeformationalFlow, 3, FVConstant] =
        7.7527847763998993e+06
    expected_result[ReversingDeformationalFlow, 4, FVConstant] =
        1.4209138735343830e+05

    expected_result[ReversingDeformationalFlow, 1, FVLinear] =
        2.0156862801802599e+08
    expected_result[ReversingDeformationalFlow, 2, FVLinear] =
        7.0092714938572392e+07
    expected_result[ReversingDeformationalFlow, 3, FVLinear] =
        7.7527849036761308e+06
    expected_result[ReversingDeformationalFlow, 4, FVLinear] =
        1.4209138776027536e+05

    numlevels =
        integration_testing || ClimateMachine.Settings.integration_testing ? 4 :
        1
    explicit_method = LSRK144NiegemannDiehlBusch
    @testset "$(@__FILE__)" begin
        for FT in (Float64,)
            for problem in (SolidBodyRotation, ReversingDeformationalFlow)
                for fvmethod in (FVConstant, FVLinear)
                    cfl = max_cfl[explicit_method]
                    result = zeros(FT, numlevels)
                    for l in 1:numlevels
                        numelems_horizontal = 2^(l - 1) * base_num_elem
                        numelems_vertical = 3

                        _planet_radius = FT(planet_radius(param_set))
                        domain_height = FT(10e3)
                        vert_range =
                            (_planet_radius, _planet_radius + domain_height)

                        topl = StackedCubedSphereTopology(
                            mpicomm,
                            numelems_horizontal,
                            range(
                                vert_range[1],
                                stop = vert_range[2],
                                length = numelems_vertical + 1,
                            ),
                        )

                        timeend = finaltime(problem())
                        outputtime = timeend

                        @info (ArrayType, FT)
                        vtkdir =
                            output ?
                            "vtk_fvm_advection_sphere" *
                            "_$problem" *
                            "_$explicit_method" *
                            "_poly$(polynomialorder)" *
                            "_$(ArrayType)_$(FT)" *
                            "_level$(l)" :
                            nothing

                        result[l], Δmass = test_run(
                            mpicomm,
                            ArrayType,
                            vert_range,
                            topl,
                            problem,
                            explicit_method,
                            fvmethod(),
                            cfl,
                            polynomialorder,
                            timeend,
                            FT,
                            vtkdir,
                            outputtime,
                        )
                        @test result[l] ≈
                              FT(expected_result[problem, l, fvmethod])
                        @test Δmass <= FT(1e-12)
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
