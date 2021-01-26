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
using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude

using ClimateMachine.Orientations
using CLIMAParameters
using CLIMAParameters.Planet

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 60e3

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

include("hyperdiffusion_model.jl")
include("SphericalHarmonics.jl")

struct ConstantHyperDiffusion{dim, dir, FT} <: HyperDiffusionProblem
    D::FT
    l::FT
    m::FT
end

function nodal_init_state_auxiliary!(
    balance_law::HyperDiffusion,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    aux.coord = geom.coord
    r = norm(aux.coord)
    l = balance_law.problem.l
    aux.c = get_c(l, r)
    aux.D = balance_law.problem.D * SMatrix{3,3,Float64}(I)
end

"""
    initial condition is given by ρ0 = Y{m,l}(θ, λ)
    test: ∇^4_horz ρ0 = l^2(l+1)^2/r^4 ρ0 where r=a+z
"""

function initial_condition!(
    problem::ConstantHyperDiffusion{dim, dir},
    state,
    aux,
    x,
    t,
) where {dim, dir}
    @inbounds begin
        FT = eltype(state)
        # import planet paraset
        _a::FT = planet_radius(param_set)

        φ = latitude(SphericalOrientation(), aux)
        λ = longitude(SphericalOrientation(), aux)
        r = norm(aux.coord)
        z = r - _a

        l = Int64(problem.l)
        m = Int64(problem.m)

        c = get_c(l, r)
        # state.ρ = calc_Ylm(φ, λ, l, m) * exp(-problem.D*c*t)
        # state.ρ = calc_Ylm(φ, λ, l, m) * exp(-problem.D*c*t) * exp(-z/30.0e3)
        state.ρ = cos(z/30.0e3)
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
    l,
    m,
)

    grid = DiscontinuousSpectralElementGrid(
            topl,
            FloatType = FT,
            DeviceArray = ArrayType,
            polynomialorder = N,
            meshwarp = ClimateMachine.Mesh.Topologies.cubedshellwarp,
        )
    dx = min_node_distance(grid, HorizontalDirection())
    dz = min_node_distance(grid, VerticalDirection())

    D = (dx/2)^4/2/τ

    model = HyperDiffusion{dim}(ConstantHyperDiffusion{dim, direction(), FT}(D, l, m))
    dg = DGModel(
            model,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            direction = direction(),
        )

    Q0 = init_ode_state(dg, FT(0))
    @info "Δ(horz) Δ(vert)" (dx, dz)

    ϵ = 1e-3

    rhs_DGsource = similar(Q0)
    dg(rhs_DGsource, Q0, nothing, 0)

    # analycal vs analycal
    # analytical solution for Y_{l,m}
    rhs_anal = .- dg.state_auxiliary.c * D .* Q0

    # timestepper for 1 step
    Q_DGlsrk = Q0
    dg1 = dg
    
    dt = dx^4 / 25 / sum(D)
    @info dt

    Q_anal = init_ode_state(dg1, dt)

    lsrk = LSRK54CarpenterKennedy(dg1, Q_DGlsrk; dt = dt, t0 = 0)
    solve!(Q_DGlsrk, lsrk; timeend = dt)
    @info "DG stepper vs rhs: " norm(Q_anal-Q_DGlsrk)/norm(Q_anal) 

    # ana ρ(t) + finite diff in time
    # rhs_FD = (init_ode_state(dg, FT(ϵ)) .- Q0) ./ ϵ

    # @show "ANA" norm(rhs_anal)
    # @show "FD" norm(rhs_FD)
    # @show "DG" norm(rhs_DGsource)
    # @show "ANA vs FD" norm(rhs_anal .- rhs_FD)/norm(rhs_anal)
    # @show "ANA vs DG" norm(rhs_anal .- rhs_DGsource) / norm(rhs_anal)
    # @show "FD vs DG" norm(rhs_FD .- rhs_DGsource) / norm(rhs_FD)

    do_output(mpicomm, "output", dg, rhs_DGsource, rhs_anal, model)
    # do_output(mpicomm, "output", dg, Q0, rhs_anal, model)
    # do_output(mpicomm, "output", dg, Q_DGlsrk, Q_anal, model)

    return norm(rhs_anal .- rhs_DGsource) / norm(rhs_anal)
    # return norm(Q_anal-Q_DGlsrk)/norm(Q_anal)
end

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

get_c(l,r) = l^2*(l+1)^2/r^4

calc_Plm(φ, l, m) = Compute_Legendre!(m, l, sin(φ), length(φ))

function calc_Ylm(φ, λ, l, m)
    qnm = calc_Plm(φ, l, m)[1][m+1,l+1,:][1]
    return real(qnm .* exp(im*m*λ))
end

# function altitude(orientation::Orientation, param_set::APS, aux::Vars)
#     FT = eltype(aux)
#     return gravitational_potential(orientation, aux) / FT(grav(param_set))
# end

# gravitational_potential(::Orientation, aux::Vars) = aux.orientation.Φ

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

    _a = planet_radius(param_set)
    @info _a

    # height = _a * 0.01
    height = 30.0e3

    @testset "$(@__FILE__)" begin
        for FT in (Float64, )# Float32,)
		    for base_num_elem in (8, )# 12, 15,)
                # for polynomialorder in (6, )#(3,4,5,6,)#4,5,6,)
		        for (polynomialorder, vert_num_elem) in ((5,8), )#(4,5), (5,3), (6,2), )

                    for τ in (1,)#4,8,) # time scale for hyperdiffusion

                        topl = StackedCubedSphereTopology(
                            mpicomm,
                            base_num_elem,
                            grid1d(_a, _a + height, nelem = vert_num_elem)
                        )

                        @info "Array FT nhorz nvert poly τ" (ArrayType, FT, base_num_elem, vert_num_elem, polynomialorder, τ)
                        result = run(mpicomm, ArrayType, dim, topl,
                                    polynomialorder, FT, direction, τ*3600, 7, 4 )

                        @test result < 5e-2

                    end
                end
            end
        end
    end
end
nothing
