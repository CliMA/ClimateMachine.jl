using Test
using StaticArrays
using ClimateMachine.VariableTemplates
using CUDA
using KernelAbstractions
using CUDAKernels
using KernelAbstractions.Extras: @unroll

include("complex_models.jl")

run_gpu = CUDA.has_cuda_gpu()
if run_gpu
    CUDA.allowscalar(false)
else
    CUDA.allowscalar(true)
end

get_device() = run_gpu ? CPU() : CUDADevice()
device_array(a, ::CUDADevice) = CuArray(a)
device_array(a, ::CPU) = Array(a)
device_rand(::CUDADevice, args...) = CUDA.rand(args...)
device_rand(::CPU, args...) = rand(args...)
number_states(m) = varsize(state(m, Int))

@kernel function mem_copy_kernel!(
    m::AbstractModel,
    dst::AbstractArray{FT, N},
    src::AbstractArray{FT, N},
) where {FT, N}
    @uniform begin
        ns = number_states(m)
        vs = state(m, FT)
        local_src = MArray{Tuple{ns}, FT}(undef)
        local_dst = MArray{Tuple{ns}, FT}(undef)
    end
    i = @index(Group, Linear)
    @inbounds begin
        @unroll for s in 1:ns
            local_src[s] = src[s, i]
        end
        mem_copy!(m, Vars{vs}(local_dst), Vars{vs}(local_src))
        @unroll for s in 1:ns
            dst[s, i] = local_dst[s]
        end
    end
end

function mem_copy!(m::ScalarModel, dst::Vars, src::Vars)
    dst.x = src.x
end

function mem_copy!(m::NTupleContainingModel{N}, dst::Vars, src::Vars) where {N}
    dst.vector_model.x = src.vector_model.x
    dst.scalar_model.x = src.scalar_model.x

    up = vuntuple(i -> src.ntuple_model[i].scalar_model.x, N)
    up_v = vuntuple(i -> src.ntuple_model[i].vector_model.x, N)
    up_sv = SVector(up...)

    @unroll_map(N) do i
        dst.ntuple_model[i].scalar_model.x = up[i]    # index into tuple
        dst.ntuple_model[i].scalar_model.x = up_sv[i] # index into SArray
        dst.ntuple_model[i].vector_model.x = up_v[i]
    end
end

@testset "ScalarModel" begin
    FT = Float32
    device = get_device()
    n_elem = 10
    m = ScalarModel()
    ns = number_states(m)
    a_src = Array{FT}(undef, ns, n_elem)
    a_dst = Array{FT}(undef, ns, n_elem)
    d_src = device_array(a_src, device)
    d_dst = device_array(a_dst, device)
    d_src .= device_rand(device, FT, ns, n_elem)
    fill!(d_dst, 0)

    work_groups = (1,)
    ndrange = (n_elem,)
    kernel! = mem_copy_kernel!(device, work_groups)
    event = kernel!(m, d_dst, d_src, ndrange = ndrange)
    wait(device, event)

    a_src = Array(d_src)
    a_dst = Array(d_dst)
    @test a_src == a_dst
end

@testset "NTupleContainingModel" begin
    FT = Float32
    device = get_device()
    n_elem = 10
    Nv = 4
    N = 3
    m = NTupleContainingModel(N, Nv)
    ns = number_states(m)
    a_src = Array{FT}(undef, ns, n_elem)
    a_dst = Array{FT}(undef, ns, n_elem)
    d_src = device_array(a_src, device)
    d_dst = device_array(a_dst, device)
    d_src .= device_rand(device, FT, ns, n_elem)
    fill!(d_dst, 0)

    work_groups = (1,)
    ndrange = (n_elem,)
    kernel! = mem_copy_kernel!(device, work_groups)
    event = kernel!(m, d_dst, d_src, ndrange = ndrange)
    wait(device, event)

    a_src = Array(d_src)
    a_dst = Array(d_dst)
    @test a_src == a_dst
end
