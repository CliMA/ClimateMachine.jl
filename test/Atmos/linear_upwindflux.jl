using Test

using StaticArrays
using LinearAlgebra

using ClimateMachine.VariableTemplates: @vars, Grad, Vars

using ClimateMachine.Atmos:
    AtmosAcousticLinearModel,
    AtmosAcousticGravityLinearModel,
    AtmosModel,
    AtmosGCMConfigType,
    linearized_pressure,
    gravitational_potential

using ClimateMachine.DGmethods:
    flux_first_order!,
    vars_state_conservative,
    vars_state_auxiliary,
    number_state_conservative,
    number_state_auxiliary,
    init_state_auxiliary!

using ClimateMachine.DGmethods.NumericalFluxes:
    UpwindNumericalFlux, numerical_flux_first_order!

using ClimateMachine.Mesh.Geometry: LocalGeometry

using CLIMAParameters: AbstractEarthParameterSet
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

@testset "Linear Atmos Upwind Flux" begin
    FT = Float64

    # Create a minimal atmos model type
    atmos_model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_conservative = (x...) -> nothing,
    )

    num_state_conservative = 5
    num_state_auxiliary = number_state_auxiliary(atmos_model, FT)

    state = zeros(FT, num_state_conservative)
    flux = MArray{Tuple{3, num_state_conservative}, FT}(undef)
    auxiliary = Vars{vars_state_auxiliary(atmos_model, FT)}(MArray{
        Tuple{3, num_state_auxiliary},
        FT,
    }(
        undef,
    ))

    coord = @SVector FT[0, 0, 1e5]
    invJ = @SMatrix FT[
        1 0 0
        0 1 0
        0 0 1
    ]
    pnt = LocalGeometry{FT, Val{1}}(Val(1), coord, invJ)
    init_state_auxiliary!(atmos_model, auxiliary, pnt)
    normal = FT.([1 2 3])
    normal /= norm(normal)
    normal = SArray{Tuple{3}, FT}(normal...)

    fluxᵀn = MArray{Tuple{num_state_conservative}, FT}(undef)

    state⁺ = rand(FT, num_state_conservative)
    state⁻ = rand(FT, num_state_conservative)

    A = zeros(FT, num_state_conservative, num_state_conservative)
    for linear_model in (
        AtmosAcousticLinearModel(atmos_model),
        AtmosAcousticGravityLinearModel(atmos_model),
    )
        # Build the matrix by multiplying by the identity
        for k in 1:num_state_conservative
            flux .= 0
            state[k] = 1
            flux_first_order!(
                linear_model,
                Grad{vars_state_conservative(linear_model, FT)}(flux),
                Vars{vars_state_conservative(linear_model, FT)}(state),
                auxiliary,
                FT(0),
            )
            state[k] = 0
            A[:, k] = flux' * normal
        end

        # Compute the numerical flux using the upwind function
        numerical_flux_first_order!(
            UpwindNumericalFlux(),
            linear_model,
            Vars{vars_state_conservative(linear_model, FT)}(fluxᵀn),
            normal,
            Vars{vars_state_conservative(linear_model, FT)}(state⁻),
            auxiliary,
            Vars{vars_state_conservative(linear_model, FT)}(state⁺),
            auxiliary,
            FT(0),
        )

        # Compute the eigenvalues of the system and check that they match the
        # upwind flux
        (Λ, W) = eigen(A)
        Λ = real.(Λ)
        Λ⁺ = (Λ + abs.(Λ)) / 2
        Λ⁻ = (Λ - abs.(Λ)) / 2
        Ω⁺ = Λ⁻ .* (W \ state⁺)
        Ω⁻ = Λ⁺ .* (W \ state⁻)
        @test all(fluxᵀn .≈ W * (Ω⁻ + Ω⁺))
    end
end
