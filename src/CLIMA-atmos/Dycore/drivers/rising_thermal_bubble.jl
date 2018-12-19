using CLIMAAtmosDycore
using Canary
using MPI

const HAVE_CUDA = try
  using CUDAdrv
  using CUDAnative
  true
catch
  false
end

macro hascuda(ex)
  return HAVE_CUDA ? :($(esc(ex))) : :()
end

const _γ = 14  // 10
const _p0 = 100000
const _R_gas = 28717 // 100
const _c_p = 100467 // 100
const _c_v = 7175 // 10
const _gravity = 10

# {{{ main
function main()
  DFloat = Float64

  # MPI.Init()
  MPI.Initialized() ||MPI.Init()
  MPI.finalize_atexit()

  mpicomm = MPI.COMM_WORLD
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # FIXME: query via hostname
  @hascuda device!(mpirank % length(devices()))

  #Initial Conditions
  function ic(dim, x...)
    # FIXME: Type generic?
    DFloat = eltype(x)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    u0 = 0
    r = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
    rc = 250.0
    θ_ref=300.0
    θ_c=0.5
    Δθ=0.0
    if r <= rc
      Δθ = 0.5 * θ_c * (1.0 + cos(π * r/rc))
    end
    θ_k=θ_ref + Δθ
    π_k=1.0 - gravity/(c_p*θ_k)*x[dim]
    c=c_v/R_gas
    ρ_k=p0/(R_gas*θ_k)*(π_k)^c
    ρ = ρ_k
    U = ρ*(u0)
    V = ρ*(0.0)
    W = ρ*(0.0)
    E = ρ*θ_k
    ρ, U, V, W, E
  end

  tend = DFloat(0.1)
  Ne = 10
  N  = 4

  mesh2D = brickmesh((range(DFloat(0); length=Ne+1, stop=1000),
                      range(DFloat(0); length=Ne+1, stop=1000)),
                     (true, false), part=mpirank+1, numparts=mpisize)

  mpirank == 0 && println("Running 2d (CPU)...")
  atmosdycore(Val(2), Val(N), mpicomm, (x...)->ic(2, x...), mesh2D, tend;
              ArrType=Array, tout = 10)
  mpirank == 0 && println()

  @hascuda begin
    mpirank == 0 && println("Running 2d (GPU)...")
    atmosdycore(Val(2), Val(N), mpicomm, (x...)->ic(2, x...), mesh2D, tend;
                ArrType=CuArray, tout = 10)
    mpirank == 0 && println()
  end

  mesh3D = brickmesh((range(DFloat(0); length=Ne+1, stop=1000),
                      range(DFloat(0); length=Ne+1, stop=1000),
                      range(DFloat(0); length=Ne+1, stop=1000)),
                   (true, true, false),
                   part=mpirank+1, numparts=mpisize)

  mpirank == 0 && println("Running 3d (CPU)...")
  atmosdycore(Val(3), Val(N), mpicomm, (x...)->ic(3, x...), mesh3D, tend;
              ArrType=Array, tout = 10)
  mpirank == 0 && println()

  @hascuda begin
    mpirank == 0 && println("Running 3d (GPU)...")
    atmosdycore(Val(3), Val(N), mpicomm, (x...)->ic(3, x...), mesh3D, tend;
                ArrType=CuArray, tout = 10)
  end

  # MPI.Finalize()
  nothing
end
# }}}

main()
