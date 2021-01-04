# Test functions used in runoff modeling.
using MPI
using OrderedCollections
using StaticArrays
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
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

    soil_param_functions =
        SoilParamFunctions{F}(porosity = 0.75, Ksat = 1e-7, S_s = 1e-3)
    ϑ_l0 = (aux) -> eltype(aux)(0.2)

    soil_water_model = SoilWaterModel(F; initialϑ_l = ϑ_l0)
    soil_heat_model = PrescribedTemperatureModel()

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)

    
    flux_bc =
        compute_surface_flux.(
            Ref(m_soil),
            Ref(runoff_model),
            Ref(precip_model),
            Ref(v),
            Ref(v),
            [1, 2, 3, 4],
        )
    @test flux_bc ≈ F.([2, 4, 6, 8])
    @test eltype(flux_bc) == F
end
