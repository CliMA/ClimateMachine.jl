using MPI
using StaticArrays
using ClimateMachine
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.MPIStateArrays
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using Printf
using LinearAlgebra
using Logging

using ClimateMachine.BalanceLaws:
    BalanceLaw,
    Prognostic,
    Auxiliary,
    GradientFlux,
    UpwardIntegrals,
    DownwardIntegrals

import ClimateMachine.BalanceLaws:
    vars_state,
    integral_load_auxiliary_state!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    update_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    boundary_conditions,
    boundary_state!,
    compute_gradient_argument!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import ClimateMachine.DGMethods: init_ode_state
using ClimateMachine.Mesh.Geometry: LocalGeometry

struct IntegralTestSphereModel{T} <: BalanceLaw
    Rinner::T
    Router::T
end

function update_auxiliary_state!(
    dg::DGModel,
    m::IntegralTestSphereModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    indefinite_stack_integral!(dg, m, Q, dg.state_auxiliary, t, elems)
    reverse_indefinite_stack_integral!(dg, m, Q, dg.state_auxiliary, t, elems)

    return true
end

vars_state(::IntegralTestSphereModel, ::UpwardIntegrals, T) = @vars(v::T, r::T)
vars_state(::IntegralTestSphereModel, ::DownwardIntegrals, T) =
    @vars(v::T, r::T)
vars_state(m::IntegralTestSphereModel, ::Auxiliary, T) = @vars(
    int::vars_state(m, UpwardIntegrals(), T),
    rev_int::vars_state(m, DownwardIntegrals(), T),
    r::T,
    v::T
)

vars_state(::IntegralTestSphereModel, ::Prognostic, T) = @vars()
vars_state(::IntegralTestSphereModel, ::GradientFlux, T) = @vars()

flux_first_order!(::IntegralTestSphereModel, _...) = nothing
flux_second_order!(::IntegralTestSphereModel, _...) = nothing
source!(::IntegralTestSphereModel, _...) = nothing
boundary_conditions(::IntegralTestSphereModel) = (nothing,)
boundary_state!(_, ::Nothing, ::IntegralTestSphereModel, _...) = nothing
init_state_prognostic!(::IntegralTestSphereModel, _...) = nothing
wavespeed(::IntegralTestSphereModel, _...) = 1

function nodal_init_state_auxiliary!(
    m::IntegralTestSphereModel,
    aux::Vars,
    tmp::Vars,
    g::LocalGeometry,
)
    x, y, z = g.coord
    aux.r = hypot(x, y, z)
    θ = atan(y, x)
    ϕ = asin(z / aux.r)
    # Exact integral
    aux.v = 1 + cos(ϕ)^2 * sin(θ)^2 + sin(ϕ)^2
    aux.int.v = exp(-aux.v * aux.r^2) - exp(-aux.v * m.Rinner^2)
    aux.int.r = aux.r - m.Rinner
    aux.rev_int.v = exp(-aux.v * m.Router^2) - exp(-aux.v * aux.r^2)
    aux.rev_int.r = m.Router - aux.r
end

@inline function integral_load_auxiliary_state!(
    m::IntegralTestSphereModel,
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    integrand.v = -2 * aux.r * aux.v * exp(-aux.v * aux.r^2)
    integrand.r = 1
end

@inline function integral_set_auxiliary_state!(
    m::IntegralTestSphereModel,
    aux::Vars,
    integral::Vars,
)
    aux.int.v = integral.v
    aux.int.r = integral.r
end

@inline function reverse_integral_load_auxiliary_state!(
    m::IntegralTestSphereModel,
    integral::Vars,
    state::Vars,
    aux::Vars,
)
    integral.v = aux.int.v
    integral.r = aux.int.r
end

@inline function reverse_integral_set_auxiliary_state!(
    m::IntegralTestSphereModel,
    aux::Vars,
    integral::Vars,
)
    aux.rev_int.v = integral.v
    aux.rev_int.r = integral.r
end

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

using Test
function test_run(mpicomm, topl, ArrayType, N, FT, Rinner, Router)
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = Topologies.cubedshellwarp,
    )
    dg = DGModel(
        IntegralTestSphereModel(Rinner, Router),
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))
    dQdt = similar(Q)

    exact_aux = copy(dg.state_auxiliary)
    dg(dQdt, Q, nothing, 0.0)
    (int_r_ind, rev_int_r_ind, aux_center_exact_ind) = varsindices(
        vars_state(dg.balance_law, Auxiliary(), FT),
        ("int.r", "rev_int.r", "r"),
    )

    # We should be exact for the integral of ∫_{R_{inner}}^{r} 1
    if N > 0
        @test exact_aux[:, int_r_ind, :] ≈ dg.state_auxiliary[:, int_r_ind, :]
        @test exact_aux[:, rev_int_r_ind, :] ≈
              dg.state_auxiliary[:, rev_int_r_ind, :]
    else
        # For N = 0 we only compare the first integral which is the integral of
        # a vertical constant function; N = 0 can also integrate linears exactly
        # since we use the midpoint rule to compute the face value, but the
        # averaging procedure we use below does not work in this case
        nvertelem = topl.stacksize
        nhorzelem = div(length(topl.elems), nvertelem)
        naux = size(exact_aux, 2)
        ndof = size(exact_aux, 1)

        # Reshape the data array to be (dofs in 1D, dofs in 1D, dofs in 1D,
        # naux, vertical elm, horizontal elm)
        aux =
            reshape(dg.state_auxiliary.data, (ndof, naux, nvertelem, nhorzelem))

        # Store the computed face values
        A_faces = aux[:, int_r_ind, :, :]
        # Store the exact center values
        A_center_exact = aux[:, aux_center_exact_ind, :, :] .- Rinner

        # With N = 0, the integral will return the values in the faces. Namely,
        # verical element index `eV` will be the value of the integral on the
        # face ABOVE element `eV`. Namely,
        #    A_faces[n, eV, eH]
        # will be degree of freedom `n`, in horizontal element stack `eH`, and
        # face `eV + 1/2`.
        #
        # The exact values stored in `A_center_exact` are actually at the cell
        # centers because these are computed using the `init_state_auxiliary!`
        # which evaluates using the cell centers.
        #
        # This mismatch means we need to convert from faces to cell centers for
        # comparison, and we do this using averaging to go from faces to cell
        # centers.

        # Storage for the averaging
        A_center = similar(A_faces)

        # Bottom cell value is average of 0 and top face of cell
        A_center[:, 1, :] .= A_faces[:, 1, :] / 2

        # Remaining cells are average of the two faces
        A_center[:, 2:end, :] .=
            (A_faces[:, 1:(end - 1), :] + A_faces[:, 2:end, :]) / 2

        # Compare the exact and computed
        @test A_center ≈ A_center_exact

        # We do the same things for the reverse integral, the only difference is
        # now the values
        #    RA_faces[n, eV, eH]
        # will be degree of freedom `n`, in horizontal element stack `eH`, and

        # We do the same things for the reverse integral, the only difference is
        # now the values
        #    RA_faces[n, eV, eH]
        # will be degree of freedom `n`, in horizontal element stack `eH`, and
        # face `eV - 1/2` (e.g., the face below element `eV`

        # Store the computed face values
        RA_faces = aux[:, rev_int_r_ind, :, :]
        # Store the exact center values
        RA_center_exact = aux[:, aux_center_exact_ind, :, :]

        # Storage for the averaging
        RA_center = similar(RA_faces)

        # Top cell value is average of 0 and top face of cell
        RA_center[:, end, :] .= Router .- RA_faces[:, end, :] / 2

        # Remaining cells are average of the two faces
        RA_center[:, 1:(end - 1), :] .=
            (RA_faces[:, 1:(end - 1), :] + RA_faces[:, 2:end, :]) / 2

        # Compare the exact and computed
        @test RA_center ≈ RA_center_exact

        # All the `JcV` (line integral metrics) values should be `Δ / 2`
        Δ = (Router - Rinner) / nvertelem
        @test all(Δ .≈ 2grid.vgeo[:, Grids._JcV, :])
    end

    euclidean_distance(exact_aux, dg.state_auxiliary)
end

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    base_Nhorz = 4
    base_Nvert = 2
    Rinner = 1 // 2
    Router = 1

    expected_result = Dict()
    expected_result[1] = [
        1.5934735012225074e-02
        4.0030667455285352e-03
        1.0020652111566574e-03
        2.5059856392475887e-04
    ]
    expected_result[4] = [
        4.662884229467401e-7,
        7.218989778540723e-9,
        1.1258613174916711e-10,
        1.7587739986848968e-12,
    ]
    lvls = integration_testing ? length(expected_result[4]) : 1

    for N in (0, 1, 4)
        for FT in (Float64,)
            err = zeros(FT, lvls)
            for l in 1:lvls
                @info (ArrayType, FT, "sphere", N, l)
                Nhorz = 2^(l - 1) * base_Nhorz
                Nvert = 2^(l - 1) * base_Nvert
                Rrange = grid1d(FT(Rinner), FT(Router); nelem = Nvert)
                topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
                err[l] = test_run(
                    mpicomm,
                    topl,
                    ArrayType,
                    N,
                    FT,
                    FT(Rinner),
                    FT(Router),
                )
                if N != 0
                    @test expected_result[N][l] ≈ err[l] rtol = 1e-3 atol =
                        eps(FT)
                end
            end
            if integration_testing && N != 0
                @info begin
                    msg = "polynomialorder order $N"
                    for l in 1:(lvls - 1)
                        rate = log2(err[l]) - log2(err[l + 1])
                        msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
                    end
                    msg
                end
            end
        end
    end
end

nothing
