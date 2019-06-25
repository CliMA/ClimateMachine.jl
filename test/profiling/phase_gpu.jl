using GPUifyLoops, CuArrays, CUDAnative, LinearAlgebra
using CLIMA
using CLIMA.MoistThermodynamics

function kernel!(B, A)
  @inbounds @loop for i in (1:size(A,1);
                            (blockIdx().x-1) * blockDim().x + threadIdx().x)

    q_tot = A[i, 1]
    ρ     = A[i, 2]
    e_int = A[i, 3]

    TS = PhaseEquil(e_int, q_tot, ρ)
    part = PhasePartition(TS)

    B[i, 1] = part.ice
    B[i, 2] = part.liq
    B[i, 3] = part.tot
  end
  nothing
end

T = Float32
L = 10^6

q_tot = T(0.05) .* (1 .+ rand(T, L))
ρ     = T(0.75) .* (1 .+ rand(T, L))
e_int = T(7100) .* (1 .+ rand(T, L))

a = [q_tot ρ e_int]
b = similar(a)
kernel!(b, a)
@show Base.summarysize(a)/10^9

@eval function kernel!(A::CuArray, B::CuArray)
  threads = 512
  blocks = ceil(Int, size(A,1)/threads)
  @launch(CUDA(), threads=threads, blocks=blocks, kernel!(A, B))
end

ca = CuArray(a)
cb = similar(ca)

kernel!(cb, ca)

b′ = Array(cb)
@info("Final", b′ ≈ b, norm(b′-b, Inf))
