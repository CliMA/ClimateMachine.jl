module TestThermodynamics

using Test
using KernelAbstractions
using CUDAKernels
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using UnPack
using CUDA
using Random
using RootSolvers
const TD = Thermodynamics

using LinearAlgebra
using CLIMAParameters
using CLIMAParameters.Planet

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
ClimateMachine.init()
ArrayType = ClimateMachine.array_type()

@show ArrayType

device(::Type{T}) where {T <: Array} = CPU()
device(::Type{T}) where {T <: CuArray} = CUDADevice()

include("profiles.jl")

@kernel function test_thermo_kernel!(
    param_set,
    dst::AbstractArray{FT},
    e_int,
    ρ,
    p,
    q_tot,
) where {FT}
    i = @index(Group, Linear)
    @inbounds begin

        ts = PhaseEquil(param_set, FT(e_int[i]), FT(ρ[i]), FT(q_tot[i]))
        dst[1, i] = air_temperature(ts)

        ts_ρpq = PhaseEquil_ρpq(
            param_set,
            FT(ρ[i]),
            FT(p[i]),
            FT(q_tot[i]),
            true,
            100,
            RegulaFalsiMethod,
        )
        dst[2, i] = air_temperature(ts_ρpq)
    end
end


@testset "Thermodynamics - kernels" begin
    FT = Float32
    dev = device(ArrayType)
    profiles = PhaseEquilProfiles(param_set, Array)
    slice = Colon()
    profiles = convert_profile_set(profiles, ArrayType, slice)

    n_profiles = length(profiles.z)
    n_vars = length(propertynames(profiles))
    d_dst = ArrayType(Array{FT}(undef, 2, n_profiles))
    fill!(d_dst, 0)

    @unpack e_int, ρ, p, q_tot = profiles

    work_groups = (1,)
    ndrange = (n_profiles,)
    kernel! = test_thermo_kernel!(dev, work_groups)
    event = kernel!(param_set, d_dst, e_int, ρ, p, q_tot, ndrange = ndrange)
    wait(dev, event)

    ts_correct = PhaseEquil.(param_set, Array(e_int), Array(ρ), Array(q_tot))
    @test all(Array(d_dst)[1, :] .≈ air_temperature.(ts_correct))

    ts_correct =
        PhaseEquil_ρpq.(
            param_set,
            Array(ρ),
            Array(p),
            Array(q_tot),
            true,
            100,
            RegulaFalsiMethod,
        )
    @test all(Array(d_dst)[2, :] .≈ air_temperature.(ts_correct))

end

end
