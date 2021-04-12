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
    grid = SpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = Topologies.equiangular_cubed_sphere_warp,
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
    (int_r_ind, rev_int_r_ind) = varsindices(
        vars_state(dg.balance_law, Auxiliary(), FT),
        ("int.r", "rev_int.r"),
    )

    # Since N = 0 integrals live at the faces we need to average values to the
    # faces for comparison
    if N == 0

        nvertelem = topl.stacksize
        nhorzelem = div(length(topl.elems), nvertelem)
        naux = size(exact_aux, 2)
        ndof = size(exact_aux, 1)

        # Reshape the data array to be (dofs, naux, vertical elm, horizontal elm)
        aux =
            reshape(dg.state_auxiliary.data, (ndof, naux, nvertelem, nhorzelem))

        # average forward integrals to cell centers
        for ind in varsindices(
            vars_state(dg.balance_law, Auxiliary(), FT),
            ("int.r", "int.v"),
        )

            # Store the computed face values
            A_faces = aux[:, ind, :, :]

            # Bottom cell value is average of 0 and top face of cell
            aux[:, ind, 1, :] .= A_faces[:, 1, :] / 2

            # Remaining cells are average of the two faces
            aux[:, ind, 2:end, :] .=
                (A_faces[:, 1:(end - 1), :] + A_faces[:, 2:end, :]) / 2
        end

        # average reverse integrals to cell centers
        for ind in varsindices(
            vars_state(dg.balance_law, Auxiliary(), FT),
            ("rev_int.r", "rev_int.v"),
        )

            # Store the computed face values
            RA_faces = aux[:, ind, :, :]

            # Bottom cell value is average of 0 and top face of cell
            aux[:, ind, end, :] .= RA_faces[:, end, :] / 2

            # Remaining cells are average of the two faces
            aux[:, ind, 1:(end - 1), :] .=
                (RA_faces[:, 1:(end - 1), :] + RA_faces[:, 2:end, :]) / 2
        end
    end
    # We should be exact for the integral of ∫_{R_{inner}}^{r} 1
    @test exact_aux[:, int_r_ind, :] ≈ dg.state_auxiliary[:, int_r_ind, :]
    @test exact_aux[:, rev_int_r_ind, :] ≈
          dg.state_auxiliary[:, rev_int_r_ind, :]
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
    expected_result[0] = [
        2.2259657670562167e-02
        5.6063943176909315e-03
        1.4042479532664005e-03
        3.5122834187695408e-04
    ]
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
                @test expected_result[N][l] ≈ err[l] rtol = 1e-3 atol = eps(FT)
            end
            if lvls > 1
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
