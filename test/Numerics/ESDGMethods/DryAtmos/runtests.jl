using Test
using ClimateMachine
import ClimateMachine.BalanceLaws
import ClimateMachine.BalanceLaws: boundary_state!
using ClimateMachine.DGMethods: ESDGModel, init_ode_state
using ClimateMachine.DGMethods.NumericalFluxes:
    numerical_volume_flux_first_order!
using ClimateMachine.Mesh.Topologies: BrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid
using ClimateMachine.VariableTemplates: varsindex
using StaticArrays: MArray, @SVector
using KernelAbstractions: wait
using Random
using DoubleFloats
using MPI
using ClimateMachine.MPIStateArrays

Random.seed!(7)

include("DryAtmos.jl")

boundary_state!(::Nothing, _...) = nothing

struct TestProblem <: AbstractDryAtmosProblem end

# Random initialization function
function init_state_prognostic!(
    ::DryAtmosModel,
    ::TestProblem,
    state_prognostic,
    state_auxiliary,
    _...,
) where {dim}
    FT = eltype(state_prognostic)
    ρ = state_prognostic.ρ = rand(FT) + 1
    ρu = state_prognostic.ρu = 2 * (@SVector rand(FT, 3)) .- 1
    p = rand(FT) + 1
    Φ = state_auxiliary.Φ
    state_prognostic.ρe = totalenergy(ρ, ρu, p, Φ)
end

function check_operators(FT, dim, mpicomm, N, ArrayType)
    # Create a warped mesh so the metrics are not constant
    Ne = (8, 9, 10)
    brickrange = (
        range(FT(-1); length = Ne[1] + 1, stop = 1),
        range(FT(-1); length = Ne[2] + 1, stop = 1),
        range(FT(-1); length = Ne[3] + 1, stop = 1),
    )
    topl = BrickTopology(
        mpicomm,
        ntuple(k -> brickrange[k], dim);
        periodicity = ntuple(k -> true, dim),
    )
    warpfun =
        (x1, x2, x3) -> begin
            α = (4 / π) * (1 - x1^2) * (1 - x2^2) * (1 - x3^2)
            # Rotate by α with x1 and x2
            x1, x2 = cos(α) * x1 - sin(α) * x2, sin(α) * x1 + cos(α) * x2
            # Rotate by α with x1 and x3
            if dim == 3
                x1, x3 = cos(α) * x1 - sin(α) * x3, sin(α) * x1 + cos(α) * x3
            end
            return (x1, x2, x3)
        end
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = warpfun,
    )

    # Orientation does not matter since we will be setting the geopotential to a
    # random field
    model = DryAtmosModel{dim}(FlatOrientation(), TestProblem())

    ##################################################################
    # check that the volume terms lead to only surface contributions #
    ##################################################################
    # Create the ES model
    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = nothing,
    )

    # Make the Geopotential random
    esdg.state_auxiliary .= ArrayType(2rand(FT, size(esdg.state_auxiliary)))
    start_exchange = MPIStateArrays.begin_ghost_exchange!(esdg.state_auxiliary)
    end_exchange = MPIStateArrays.end_ghost_exchange!(
        esdg.state_auxiliary,
        dependencies = start_exchange,
    )
    wait(end_exchange)

    # Create a random state
    state_prognostic = init_ode_state(esdg; init_on_cpu = true)

    # Storage for the tendency
    volume_tendency = similar(state_prognostic)

    # Compute the tendency function
    esdg(volume_tendency, state_prognostic, nothing, 0)

    # Check that the volume terms only lead to surface integrals of
    #    ∑_{j} n_j ψ_j
    # where Ψ_j = β^T f_j - ζ_j = ρu_j
    Np, K = (N + 1)^dim, length(esdg.grid.topology.realelems)

    # Get the mass matrix on the host
    _M = ClimateMachine.Grids._M
    M = Array(grid.vgeo[:, _M:_M, 1:K])

    # Get the state, tendency, and aux on the host
    Q = Array(state_prognostic.data[:, :, 1:K])
    dQ = Array(volume_tendency.data[:, :, 1:K])
    A = Array(esdg.state_auxiliary.data[:, :, 1:K])

    # Compute the entropy variables
    β = similar(Q, Np, number_states(model, Entropy()), K)
    @views for e in 1:K
        for i in 1:Np
            state_to_entropy_variables!(
                model,
                β[i, :, e],
                Q[i, :, e],
                A[i, :, e],
            )
        end
    end

    # Get the unit normals and surface mass matrix
    sgeo = Array(grid.sgeo)
    n1 = sgeo[ClimateMachine.Grids._n1, :, :, 1:K]
    n2 = sgeo[ClimateMachine.Grids._n2, :, :, 1:K]
    n3 = sgeo[ClimateMachine.Grids._n3, :, :, 1:K]
    sM = sgeo[ClimateMachine.Grids._sM, :, :, 1:K]

    # Get the Ψs
    fmask = Array(grid.vmap⁻[:, :, 1])
    _ρu = varsindex(vars_state(model, Prognostic(), FT), :ρu)
    Ψ1 = Q[fmask, _ρu[1], 1:K]
    Ψ2 = Q[fmask, _ρu[2], 1:K]
    Ψ3 = Q[fmask, _ρu[3], 1:K]

    # Compute the surface integral:
    #    ∫_Ωf ∑_j n_j * Ψ_j
    surface = sum(sM .* (n1 .* Ψ1 + n2 .* Ψ2 + n3 .* Ψ3), dims = (1, 2))[:]

    # Compute the volume integral:
    #   -∫_Ω ∑_j β^T (dq/dt)
    # (tendency is -dq / dt)
    num_state = number_states(model, Prognostic())
    volume = sum(β[:, 1:num_state, :] .* M .* dQ, dims = (1, 2))[:]

    @test all(isapprox.(surface, volume; atol = 10eps(FT), rtol = sqrt(eps(FT))))

    ###########################################
    # check that the volume and surface match #
    ###########################################
    esdg = ESDGModel(
        model,
        grid;
        state_auxiliary = esdg.state_auxiliary,
        volume_numerical_flux_first_order = nothing,
        surface_numerical_flux_first_order = EntropyConservative(),
    )

    surface_tendency = similar(state_prognostic)

    # Compute the tendency function
    esdg(surface_tendency, state_prognostic, nothing, 0)

    # Surface integral should be equal and opposite to the volume integral
    dQ = Array(surface_tendency.data[:, :, 1:K])
    volume_integral = MPI.Allreduce(sum(volume), +, mpicomm)
    surface_integral =
        MPI.Allreduce(sum(β[:, 1:num_state, :] .* M .* dQ), +, mpicomm)
    @test volume_integral ≈ -surface_integral

    ########################################################
    # check that the full tendency is entropy conservative #
    ########################################################
    esdg = ESDGModel(
        model,
        grid;
        state_auxiliary = esdg.state_auxiliary,
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = EntropyConservative(),
    )

    tendency = similar(state_prognostic)

    # Compute the tendency function
    esdg(tendency, state_prognostic, nothing, 0)

    # Check for entropy conservation
    dQ = Array(tendency.data[:, :, 1:K])
    integral = MPI.Allreduce(sum(β[:, 1:num_state, :] .* M .* dQ), +, mpicomm)
    @test isapprox(integral, 0, atol = sqrt(eps(sum(volume))))
end

let
    model = DryAtmosModel{3}(FlatOrientation(), TestProblem())
    num_state = number_states(model, Prognostic())
    num_aux = number_states(model, Auxiliary())
    num_entropy = number_states(model, Entropy())

    @testset "state to entropy variable transforms" begin
        for FT in (Float32, Float64, Double64, BigFloat)
            state_in =
                [1, 2, 2, 2, 1] .* rand(FT, num_state) + [3, -1, -1, -1, 100]
            aux_in = rand(FT, num_aux)
            state_out = similar(state_in)
            aux_out = similar(aux_in)
            entropy = similar(state_in, num_entropy)

            state_to_entropy_variables!(model, entropy, state_in, aux_in)
            entropy_variables_to_state!(model, state_out, aux_out, entropy)

            @test all(state_in .≈ state_out)
            #@test all(aux_in .≈ aux_out)
        end
    end

    @testset "test numerical flux for Tadmor shuffle" begin
        # Vars doesn't work with BigFloat so we will use Double64
        for FT in (Float32, Float64, Double64)
            # Create some random states
            state_1 =
                [1, 2, 2, 2, 1] .* rand(FT, num_state) + [3, -1, -1, -1, 100]
            aux_1 = 0 * rand(FT, num_aux)

            state_2 =
                [1, 2, 2, 2, 1] .* rand(FT, num_state) + [3, -1, -1, -1, 100]
            aux_2 = 0 * rand(FT, num_aux)

            # Get the entropy variables for the two states
            entropy_1 = similar(state_1, num_entropy)
            state_to_entropy_variables!(model, entropy_1, state_1, aux_1)

            entropy_2 = similar(state_1, num_entropy)
            state_to_entropy_variables!(model, entropy_2, state_2, aux_2)

            # Get the values of Ψ_j = β^T f_j - ζ_j = ρu_j where β is the
            # entropy variables, f_j is the conservative flux, and ζ_j is the
            # entropy flux. For conservation laws this is the entropy potential.
            Ψ_1 = Vars{vars_state(model, Prognostic(), FT)}(state_1).ρu
            Ψ_2 = Vars{vars_state(model, Prognostic(), FT)}(state_2).ρu

            # Evaluate the flux with both orders of the two states
            H_12 = fill!(MArray{Tuple{3, num_state}, FT}(undef), -zero(FT))
            numerical_volume_flux_first_order!(
                EntropyConservative(),
                model,
                H_12,
                state_1,
                aux_1,
                state_2,
                aux_2,
            )

            H_21 = fill!(MArray{Tuple{3, num_state}, FT}(undef), -zero(FT))
            numerical_volume_flux_first_order!(
                EntropyConservative(),
                model,
                H_21,
                state_2,
                aux_2,
                state_1,
                aux_1,
            )

            # Check that we satisfy the Tadmor shuffle
            @test all(
                H_12 * entropy_1[1:num_state] - H_21 * entropy_2[1:num_state] .≈
                Ψ_1 - Ψ_2,
            )
        end
    end

    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD
    polynomialorder = 4
    # XXX: Unfortunately `Double64` doesn't work completely on the GPU
    test_types =
        ArrayType <: Array ? (Float32, Float64, Double64) : (Float32, Float64)
    for FT in test_types
        for dim in 2:3
            @testset "check ESDGMethods relations for dim = $dim and FT = $FT" begin
                check_operators(FT, dim, mpicomm, polynomialorder, ArrayType)
            end
        end
    end
end
