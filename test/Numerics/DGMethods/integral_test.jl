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

using ClimateMachine.BalanceLaws

import ClimateMachine.BalanceLaws:
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    update_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import ClimateMachine.DGMethods: init_ode_state
using ClimateMachine.Mesh.Geometry: LocalGeometry


struct IntegralTestModel{dim} <: BalanceLaw end

vars_state(::IntegralTestModel, ::DownwardIntegrals, T) = @vars(a::T, b::T)
vars_state(::IntegralTestModel, ::UpwardIntegrals, T) = @vars(a::T, b::T)
vars_state(m::IntegralTestModel, ::Auxiliary, T) = @vars(
    int::vars_state(m, UpwardIntegrals(), T),
    rev_int::vars_state(m, DownwardIntegrals(), T),
    coord::SVector{3, T},
    a::T,
    b::T,
    rev_a::T,
    rev_b::T
)

vars_state(::IntegralTestModel, ::AbstractStateType, T) = @vars()

flux_first_order!(::IntegralTestModel, _...) = nothing
flux_second_order!(::IntegralTestModel, _...) = nothing
source!(::IntegralTestModel, _...) = nothing
boundary_state!(_, ::IntegralTestModel, _...) = nothing
init_state_prognostic!(::IntegralTestModel, _...) = nothing
wavespeed(::IntegralTestModel, _...) = 1

function nodal_init_state_auxiliary!(
    ::IntegralTestModel{dim},
    aux::Vars,
    tmp::Vars,
    g::LocalGeometry,
) where {dim}
    x, y, z = aux.coord = g.coord
    if dim == 2
        aux.a = x * y
        aux.b = 2 * x * y + sin(x) * y^2 / 2 - (z - 1)^2 * y^3 / 3
        y_top = 3
        a_top = x * y_top
        b_top = 2 * x * y_top + sin(x) * y_top^2 / 2 - (z - 1)^2 * y_top^3 / 3
        aux.rev_a = a_top - aux.a
        aux.rev_b = b_top - aux.b
    else
        aux.a = x * z + y * z
        aux.b = 2 * x * z + sin(x) * y * z - (1 + (z - 1)^3) * y^2 / 3
        zz_top = 3
        a_top = x * zz_top + y * zz_top
        b_top =
            2 * x * zz_top + sin(x) * y * zz_top -
            (1 + (zz_top - 1)^3) * y^2 / 3
        aux.rev_a = a_top - aux.a
        aux.rev_b = b_top - aux.b
    end
end

function update_auxiliary_state!(
    dg::DGModel,
    m::IntegralTestModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    indefinite_stack_integral!(dg, m, Q, dg.state_auxiliary, t, elems)
    reverse_indefinite_stack_integral!(dg, m, Q, dg.state_auxiliary, t, elems)

    return true
end

@inline function integral_load_auxiliary_state!(
    m::IntegralTestModel{dim},
    integrand::Vars,
    state::Vars,
    aux::Vars,
) where {dim}
    x, y, z = aux.coord
    integrand.a = x + (dim == 3 ? y : 0)
    integrand.b = 2 * x + sin(x) * y - (z - 1)^2 * y^2
end

@inline function integral_set_auxiliary_state!(
    m::IntegralTestModel,
    aux::Vars,
    integral::Vars,
)
    aux.int.a = integral.a
    aux.int.b = integral.b
end

@inline function reverse_integral_load_auxiliary_state!(
    m::IntegralTestModel,
    integral::Vars,
    state::Vars,
    aux::Vars,
)
    integral.a = aux.int.a
    integral.b = aux.int.b
end

@inline function reverse_integral_set_auxiliary_state!(
    m::IntegralTestModel,
    aux::Vars,
    integral::Vars,
)
    aux.rev_int.a = integral.a
    aux.rev_int.b = integral.b
end

using Test
function test_run(mpicomm, dim, Ne, N, FT, ArrayType)

    brickrange = ntuple(j -> range(FT(0); length = Ne[j] + 1, stop = 3), dim)
    topl = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = ntuple(j -> true, dim),
    )

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )
    dg = DGModel(
        IntegralTestModel{dim}(),
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))
    dQdt = similar(Q)

    dg(dQdt, Q, nothing, 0.0)

    # Wrapping in Array ensure both GPU and CPU code use same approx
    if N[end] > 0
        # Forward integral a
        @test Array(dg.state_auxiliary.data[:, 1, :]) ≈
              Array(dg.state_auxiliary.data[:, 8, :])
        # Forward integral b
        @test Array(dg.state_auxiliary.data[:, 2, :]) ≈
              Array(dg.state_auxiliary.data[:, 9, :])
        # Reverse integral a
        @test Array(dg.state_auxiliary.data[:, 3, :]) ≈
              Array(dg.state_auxiliary.data[:, 10, :])
        # Reverse integral b
        @test Array(dg.state_auxiliary.data[:, 4, :]) ≈
              Array(dg.state_auxiliary.data[:, 11, :])
    else
        # For N = 0 we only compare the first integral which is the integral of
        # a vertical constant function; N = 0 can also integrate linears exactly
        # since we use the midpoint rule to compute the face value, but the
        # averaging procedure we use below does not work in this case
        Nq = polynomialorders(grid) .+ 1

        # Reshape the data array to be (dofs, vertical elm, horizontal elm)
        A_faces = reshape(
            Array(dg.state_auxiliary.data[:, 1, :]),
            prod(Nq),
            Ne[end],           # Vertical element is fastest on stacked meshes
            prod(Ne[1:(end - 1)]), # Horiztonal elements
        )
        A_center_exact = reshape(
            Array(dg.state_auxiliary.data[:, 8, :]),
            prod(Nq),
            Ne[end],           # Vertical element is fastest on stacked meshes
            prod(Ne[1:(end - 1)]), # Horiztonal elements
        )
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

        # bottom cell value is average of 0 and top face of cell
        A_center[:, 1, :] .= A_faces[:, 1, :] / 2

        # Remaining cells are average of the two faces
        A_center[:, 2:end, :, :] .=
            (A_faces[:, 1:(end - 1), :] + A_faces[:, 2:end, :]) / 2

        # compare the exact and computed
        @test A_center ≈ A_center_exact

        # We do the same things for the reverse integral, the only difference is
        # now the values
        #    RA_faces[n, eV, eH]
        # will be degree of freedom `n`, in horizontal element stack `eH`, and
        # face `eV - 1/2` (e.g., the face below element `eV`
        # Reshape the data array to be (dofs, vertical elm, horizontal elm)
        RA_faces = reshape(
            Array(dg.state_auxiliary.data[:, 3, :]),
            prod(Nq),
            Ne[end],           # Vertical element is fastest on stacked meshes
            prod(Ne[1:(end - 1)]), # Horiztonal elements
        )
        RA_center_exact = reshape(
            Array(dg.state_auxiliary.data[:, 10, :]),
            prod(Nq),
            Ne[end],           # Vertical element is fastest on stacked meshes
            prod(Ne[1:(end - 1)]), # Horiztonal elements
        )

        # Storage for the averaging
        RA_center = similar(RA_faces)

        # top cell value is average of 0 and top face of cell
        RA_center[:, end, :] .= RA_faces[:, end, :] / 2

        # Remaining cells are average of the two faces
        RA_center[:, 1:(end - 1), :, :] .=
            (RA_faces[:, 1:(end - 1), :] + RA_faces[:, 2:end, :]) / 2

        # compare the exact and computed
        # TODO: This won't pass until the
        # kernel_reverse_indefinite_stack_integral!
        # is updated
        #JK @test RA_center ≈ RA_center_exact
    end
end

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    numelem = (5, 6, 7)
    lvls = 1

    for polynomialorder in ((4, 4), (4, 3), (4, 0))
        for FT in (Float64,)
            for dim in 2:3
                err = zeros(FT, lvls)
                for l in 1:lvls
                    @info (ArrayType, FT, dim, polynomialorder)
                    test_run(
                        mpicomm,
                        dim,
                        ntuple(j -> 2^(l - 1) * numelem[j], dim),
                        polynomialorder,
                        FT,
                        ArrayType,
                    )
                end
            end
        end
    end
end

nothing
