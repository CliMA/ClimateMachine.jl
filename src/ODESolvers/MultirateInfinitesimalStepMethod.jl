module MultirateInfinitesimalStepMethod
using ..ODESolvers
using ..MPIStateArrays: device, realview

using StaticArrays

using GPUifyLoops
include("MultirateInfinitesimalStepMethod_kernels.jl")


export MIS2, MIS3C, MIS4, MIS4a

ODEs = ODESolvers

"""
    TimeScaledRHS(α, β, rhs!)

When evaluate at time `t`, evaluates `rhs!` at time `α + βt`.
"""
mutable struct TimeScaledRHS{RT}
  α::RT
  β::RT
  rhs!
end

function (o::TimeScaledRHS)(dQ, Q, params, tau; increment)  
  o.rhs!(dQ, Q, params, o.α + o.β*tau; increment=increment)
end

using GPUifyLoops

"""
MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod,
                           α, β, γ,
                           Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

This is a time stepping object for explicitly time stepping the partitioned differential
equation given by right-hand-side functions `f_fast` and `f_slow` with the state `Q`, i.e.,

```math
  \\dot{Q} = f_fast(Q, t) + f_slow(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a multirate infinitesimal step Runge-Kutta scheme
based on the provided `α`, `β` and `γ` tableaux and `fastmethod` for solving
the fast modes.

  - [`LowStorageRungeKuttaMethod`](@ref)

### References
    @article{KnothWensch2014,
      title={Generalized split-explicit Runge--Kutta methods for the compressible Euler equations},
      author={Knoth, Oswald and Wensch, Joerg},
      journal={Monthly Weather Review},
      volume={142},
      number={5},
      pages={2067--2081},
      year={2014}
    }
"""
struct MultirateInfinitesimalStep{T, RT, AT, Nstages, Nstagesm1, Nstagesm2, Nstages_sq} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT, 1}
  "time"
  t::Array{RT, 1}
  "storage for y_n"
  yn::AT
  "Storage for ``Y_nj - y_n``"
  ΔYnj::NTuple{Nstagesm2, AT}
  "Storage for ``f(Y_nj)``"
  fYnj::NTuple{Nstagesm1, AT}
  "Storage for offset"
  offset::AT
  "slow rhs function"
  slowrhs!
  "RHS for fast solver"
  fastrhs!::TimeScaledRHS{RT}
  "fast rhs method"
  fastmethod
  "number of substeps per stage"
  nsubsteps::Int
  α::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq} 
  β::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  γ::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  d::SArray{NTuple{1, Nstages}, RT, 1, Nstages} 
  c::SArray{NTuple{1, Nstages}, RT, 1, Nstages} 
  c̃::SArray{NTuple{1, Nstages}, RT, 1, Nstages} 
  function MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod, nsubsteps,
                                      α, β, γ,
                                      Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)
    dt = [dt]
    t0 = [t0]

    Nstages = size(α, 1)

    yn = similar(Q)
    ΔYnj = ntuple(_ -> similar(Q), Nstages-2)
    fYnj = ntuple(_ -> similar(Q), Nstages-1)
    offset = similar(Q)
    fastrhs! = TimeScaledRHS(RT(0), RT(0), fastrhs!)

    d = sum(β, dims=2)

    c = similar(d)
    for i = eachindex(c)
      c[i] = d[i] 
      if i > 1
        c[i] += sum(j-> (α[i,j] + γ[i,j])*c[j], 1:i-1)
      end
    end
    c̃ = α*c
    
    new{T, RT, AT, Nstages, Nstages-1, Nstages-2, Nstages ^ 2}(dt, t0, yn, ΔYnj, fYnj, offset,
                                         slowrhs!, fastrhs!, fastmethod,nsubsteps,
                                         α, β, γ, d, c, c̃)
  end
end


# TODO almost identical functions seem to be defined for every ode solver,
# define a common one in ODEsolvers ?
function ODEs.dostep!(Q, mis::MultirateInfinitesimalStep, param,
                      timeend::AbstractFloat, adjustfinalstep::Bool)
  time, dt = mis.t[1], mis.dt[1]
  @assert dt > 0
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end

  ODEs.dostep!(Q, mis, param, time, dt)

  if dt == mis.dt[1]
    mis.t[1] += dt
  else
    mis.t[1] = timeend
  end
  return mis.t[1]
end

function ODEs.dostep!(Q, mis::MultirateInfinitesimalStep, p,
                      time::AbstractFloat, dt::AbstractFloat)
  FT = eltype(dt)
  α = mis.α
  β = mis.β
  γ = mis.γ
  yn = mis.yn
  ΔYnj = mis.ΔYnj
  fYnj = mis.fYnj
  offset = mis.offset
  c = mis.c
  c̃ = mis.c̃
  slowrhs! = mis.slowrhs!
  fastmethod = mis.fastmethod
  fastrhs! = mis.fastrhs!

  nstages = size(α, 1)

  # first stage
  copyto!(yn, Q)
  for i = 2:nstages
    d_i = sum(j -> β[i, j], 1:i-1)

    slowrhs!(fYnj[i-1], Q, p, time + c[i-1]*dt, increment=false)

    # TODO write a kernel to do this
    #=
    if i > 2
      @. ΔYnj[i-2] = Q - yn     # == 0 for i == 2
    end
    Q .= yn
    offset .= (β[i,1]/d_i) .* fYnj[1]
    for j = 2:i-1 
     Q .+= α[i, j] .* ΔYnj[j-1]
     offset .+= (γ[i, j]/d_i) .* ΔYnj[j-1] ./ dt .+ (β[i,j]/d_i) .* fYnj[j]
    end  
    =#

    threads = 256
    blocks = div(length(realview(Q)) + threads - 1, threads)
    @launch(device(Q), threads=threads, blocks=blocks,
            update!(realview(Q), realview(offset), Val(i), realview(yn), map(realview, ΔYnj[1:i-2]), map(realview, fYnj[1:i-1]), α[i,:], β[i,:], γ[i,:], d_i, dt))


    fastrhs!.α = time + c̃[i]*dt
    fastrhs!.β = (c[i] - c̃[i]) / d_i

    τ = zero(FT)
    dτ = d_i * dt / mis.nsubsteps
    fastsolver = fastmethod(fastrhs!, Q; dt=dτ, t0=τ)
    #solve!(Q, fastsolver, p; timeend = d_i * dτ) #(1c)
    for k = 1:mis.nsubsteps
      ODEs.dostep!(Q, fastsolver, p, τ, dτ, FT(1), offset)
      τ += dτ
    end
  end
end


# TODO write this function
#function MultirateInfinitesimalStep(spacedisc::AbstractSpaceMethod, RKA, RKB, RKC,
#                                Q::AT; dt=0, t0=0) where {AT<:AbstractArray}
#  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x..., increment = increment)
#  MultirateInfinitesimalStep(rhs!, RKA, RKB, RKC, Q; dt=dt, t0=t0)
#end

function MIS2(slowrhs!, fastrhs!, fastmethod, nsubsteps, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}
  α = [0 0              0              0;
       0 0              0              0;
       0 0.536946566710 0              0;
       0 0.480892968551 0.500561163566 0]

  β = [ 0                0                0              0;
        0.126848494553   0                0              0;
       -0.784838278826   1.37442675268    0              0;
       -0.0456727081749 -0.00875082271190 0.524775788629 0]

  γ = [0  0               0              0;
       0  0               0              0;
       0  0.652465126004  0              0;
       0 -0.0732769849457 0.144902430420 0]

  MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod, nsubsteps,  α, β, γ, Q; dt=dt, t0=t0)
end

function MIS3C(slowrhs!, fastrhs!, fastmethod, nsubsteps,  Q::AT; dt=0, t0=0) where {AT <: AbstractArray}
  α = [0 0              0              0;
       0 0              0              0;
       0 0.589557277145 0              0;
       0 0.544036601551 0.565511042564 0]

  β = [ 0                 0                0              0;
        0.397525189225    0                0              0;
       -0.227036463644    0.624528794618   0              0;
       -0.00295238076840 -0.270971764284   0.671323159437 0]

  γ = [0  0               0               0;
       0  0               0               0;
       0  0.142798786398  0               0;
       0 -0.0428918957402 0.0202720980282 0]

  MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod, nsubsteps,  α, β, γ, Q; dt=dt, t0=t0)
end

function MIS4(slowrhs!, fastrhs!, fastmethod, nsubsteps,  Q::AT; dt=0, t0=0) where {AT <: AbstractArray}
  α = [0 0              0              0              0;
       0 0              0              0              0;
       0 0.914092810304 0              0              0;
       0 1.14274417397 -0.295211246188 0              0;
       0 0.112965282231 0.337369411296 0.503747183119 0]

  β = [ 0                0               0              0              0;
        0.136296478423   0               0              0              0;
        0.280462398979  -0.0160351333596 0              0              0;
        0.904713355208  -1.04011183154   0.652337563489 0              0;
        0.0671969845546 -0.365621862610 -0.154861470835 0.970362444469 0]

  γ = [0  0              0               0              0;
       0  0              0               0              0;
       0  0.678951983291 0               0              0;
       0 -1.38974164070  0.503864576302  0              0;
       0 -0.375328608282 0.320925021109 -0.158259688945 0]

  MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod, nsubsteps,  α, β, γ, Q; dt=dt, t0=t0)
end

function MIS4a(slowrhs!, fastrhs!, fastmethod, nsubsteps,  Q::AT; dt=0, t0=0) where {AT <: AbstractArray}
  α = [0 0                     0                   0                   0;
       0 0                     0                   0                   0;
       0 0.52349249922385610   0                   0                   0;
       0 1.1683374366893629   -0.75762080241712637 0                   0;
       0 -0.036477233846797109 0.56936148730740477 0.47746263002599681 0]

  # β[5,1] in the paper is incorrect
  # the correct value is used below (from authors)
  β = [ 0                    0                   0                   0                   0;
        0.38758444641450318  0                   0                   0                   0;
       -0.025318448354142823 0.38668943087310403 0                   0                   0;
        0.20899983523553325 -0.45856648476371231 0.43423187573425748 0                   0;
       -0.10048822195663100 -0.46186171956333327 0.83045062122462809 0.27014914900250392 0]

  γ = [0  0                    0                    0                    0;
       0  0                    0                    0                    0;
       0  0.13145089796226542  0                    0                    0;
       0 -0.36855857648747881  0.33159232636600550  0                    0;
       0 -0.065767130537473045 0.040591093109036858 0.064902111640806712 0]

  MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod, nsubsteps, α, β, γ, Q; dt=dt, t0=t0)
end

end
