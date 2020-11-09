using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.ODESolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.Mesh.Grids: min_node_distance

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

include("hyperdiffusion_model.jl")

struct ConstantHyperDiffusion{dim, dir, FT} <: HyperDiffusionProblem
    D::FT
    k::SVector{3, FT}
end

function nodal_init_state_auxiliary!(
    balance_law::HyperDiffusion,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    aux.D = balance_law.problem.D * SMatrix{3,3,Float64}(I)
end

"""
    initial condition is given by ρ0 = sin(kx+ly+mz)
    test: ∇^4_horz ρ0 = (k^2+l^2)^2 ρ0
"""

function initial_condition!(
    problem::ConstantHyperDiffusion{dim, dir},
    state,
    aux,
    x,
    t,
) where {dim, dir}
    @inbounds begin
        k = problem.k
        c = get_eigenvalue(k, dir, dim)
        state.ρ = sin(dot(k[SOneTo(dim)], x[SOneTo(dim)])) * exp(-c * problem.D * t)
    end
end

function run(
    mpicomm,
    ArrayType,
    dim,
    topl,
    N,
    FT,
    direction,
    τ,
    k
)

    grid = DiscontinuousSpectralElementGrid(
            topl,
            FloatType = FT,
            DeviceArray = ArrayType,
            polynomialorder = N,
        )
    dx = min_node_distance(grid, direction())

    # D = (dx/2)^4/2/τ
    D = 1

    model = HyperDiffusion{dim}(ConstantHyperDiffusion{dim, direction(), FT}(D, k))
    dg = DGModel(
            model,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            direction = direction(),
        )

    dg1 = dg
    Q0 = init_ode_state(dg, FT(0))
    Q_DGlsrk = init_ode_state(dg, FT(0))
    rhs_DGlsrk = init_ode_state(dg, FT(0))
    rhs_Qanal = init_ode_state(dg, FT(0))
    @info "Δ(horz)" dx

    # ϵ = 1e-3 

    # Model vs Model
    # collect rhs from DG
    rhs_DGsource = similar(Q0)
    dg(rhs_DGsource, Q0, nothing, 0)

    # Q_frhs = Q0 .+ ϵ*rhs_DGsource

    # timestepper for 1 step
    # Q_DGlsrk = Q0
    
    dt = dx^4 / 25 / sum(D)
    @info dt

    timeend = dt

    Q_anal = init_ode_state(dg1, timeend)

    lsrk = LSRK54CarpenterKennedy(dg1, Q_DGlsrk; dt = dt, t0 = 0)

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q_DGlsrk)
            @info @sprintf(
                """Update
                simtime = %.16e
                runtime = %s
                norm(Q) = %.16e""",
                gettime(lsrk),
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end
    callbacks = (cbinfo,)
    solve!(Q_DGlsrk, lsrk; timeend = dt, callbacks=callbacks)
    solve!(Q_DGlsrk, lsrk; timeend = timeend)
    @info "time-integrate DG vs anal -- Q: " norm(Q_anal-Q_DGlsrk)/norm(Q_anal)
    
    rhs_DGlsrk = Q0 - Q_DGlsrk
    # @info rhs_DGlsrk 
    rhs_Qanal = Q0 - Q_anal
    
    @info "time-integrate DG vs anal -- rhs: " norm(rhs_Qanal-rhs_DGlsrk)/norm(rhs_Qanal)

    # @info rhs_DGlsrk 
    # analytical vs analytical
    # ana rhs
    c = get_eigenvalue(k, direction(), dim)
    rhs_anal = -c*D*Q0 
    # rhs_analXdt = rhs_anal * dt
    # rhs_dgXdt = rhs_DGsource * dt 

    # ana ρ(t) + finite diff in time
    # rhs_FD = (init_ode_state(dg, FT(ϵ)) .- Q0) ./ ϵ 
    # @info "ANA vs FD" norm(rhs_anal .- rhs_FD.ρ)/norm(rhs_FD.ρ)

    # Model vs analytical
    # @info "DG vs FD" norm(rhs_DGsource.ρ .- rhs_FD.ρ)/norm(rhs_FD.ρ)
    # @info "DG vs ANA" norm(rhs_DGsource.ρ .- rhs_anal)/norm(rhs_anal)

    # do_output(mpicomm, "output", dg, rhs_DGsource, rhs_anal, model)
    # do_output(mpicomm, "output", dg, Q0, rhs_anal, model)
    # do_output(mpicomm, "output", dg, Q0, rhs_analXdt, model)
    # do_output(mpicomm, "output", dg, Q0, rhs_dgXdt, model)
    # do_output(mpicomm, "output", dg, Q_DGlsrk, Q_anal, model)
    do_output(mpicomm, "output", dg, rhs_DGlsrk, rhs_Qanal, model)

    # return norm(rhs_DGsource.ρ .- rhs_anal)/norm(rhs_anal)
    # return norm(Q_anal-Q_DGlsrk)/norm(Q_anal) 
    return norm(rhs_Qanal-rhs_DGlsrk)/norm(rhs_Qanal) 
end

get_eigenvalue(k, dir::HorizontalDirection, dim) = 
    sum(abs2, k[SOneTo(dim - 1)]) ^2

get_eigenvalue(k, dir::VerticalDirection, dim) = k[dim]^4
    
get_eigenvalue(k, dir::EveryDirection, dim) =
    sum(abs2, k[SOneTo(dim)]) ^2
    
function do_output(
    mpicomm,
    vtkdir,
    dg,
    rhs_DGsource,
    rhs_analytical,
    model,
)
    mkpath(vtkdir)

    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/compare_mpirank%04d",
        vtkdir,
        MPI.Comm_rank(mpicomm),
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(rhs_DGsource)))
    analytical_names = statenames .* "_analytical"

    writevtk(filename, rhs_DGsource, dg, statenames, rhs_analytical, analytical_names)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/compare", vtkdir)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("compare_mpirank%04d", i - 1)
        end

        writepvtu(
            pvtuprefix,
            prefixes,
            (statenames..., analytical_names...),
            eltype(rhs_DGsource),
        )

        @info "Done writing VTK: $pvtuprefix"
    end
end

using Test
let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    numlevels =
        integration_testing || ClimateMachine.Settings.integration_testing ? 3 :
        1

    direction = HorizontalDirection
    dim = 3

    @testset "$(@__FILE__)" begin
        for FT in (Float64, )# Float32,)
            for base_num_elem in (16, )# 8,16,) 
                for polynomialorder in (3, )#4,5,6,7,8,12)

                    for τ in (1, )#4,8,) # time scale for hyperdiffusion
                        xrange = range(FT(0); length = base_num_elem + 1, stop = FT(2pi))
                        brickrange = ntuple(j -> xrange, dim)
                        periodicity = ntuple(j -> true, dim)
                        topl = StackedBrickTopology(
                            mpicomm,
                            brickrange;
                            periodicity = periodicity,
                        )

                        @info "Array FT nhorz poly τ" (ArrayType, FT, base_num_elem, polynomialorder, τ)
                        result = run(mpicomm, ArrayType, dim, topl, 
                                    polynomialorder, FT, direction, τ*3600, SVector(2, 0, 0) )
                            
                        @test result < 5e-2
                    
                    end
                end
            end
        end
    end
end
nothing
