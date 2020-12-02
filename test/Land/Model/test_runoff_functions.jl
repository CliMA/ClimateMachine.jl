# Test functions used in runoff modeling.
using MPI
using OrderedCollections
using StaticArrays
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land.Runoff
using ClimateMachine.VariableTemplates


@testset "Runoff testing" begin
    F = Float32
    runoff_model = NoRunoff()
    precip_model = DrivenConstantPrecip{F}((t) -> (2 * t))
    struct RunoffTestModel end

    function state(m::RunoffTestModel, T)
        @vars begin
            θ_i::T
            ϑ_l::T
        end
    end

    model = RunoffTestModel()

    st = state(model, F)
    v = Vars{st}(zeros(MVector{varsize(st), F}))

    flux_bc =
        compute_surface_flux.(
            Ref(runoff_model),
            Ref(precip_model),
            Ref(v),
            [1, 2, 3, 4],
        )
    @test flux_bc ≈ F.([-2, -4, -6, -8])
    @test eltype(flux_bc) == F
end
