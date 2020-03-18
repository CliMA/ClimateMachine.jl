
include("MultirateInfinitesimalStepMethod_kernels.jl")

export MIS2, MIS3C, MIS4, MIS4a, TVDMISA, TVDMISB

"""
    TimeScaledRHS(a, b, rhs!)

When evaluate at time `t`, evaluates `rhs!` at time `a + bt`.
"""
mutable struct TimeScaledRHS{RT}
  a::RT
  b::RT
  rhs!
end

function (o::TimeScaledRHS)(dQ, Q, params, tau; increment)
  o.rhs!(dQ, Q, params, o.a + o.b*tau; increment=increment)
end

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
mutable struct MultirateInfinitesimalStep{T, RT, AT, FS, Nstages, Nstagesm1, Nstagesm2, Nstages_sq} <: AbstractODESolver
  "time step"
  dt::RT
  "time"
  t::RT
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
  tsfastrhs!::TimeScaledRHS{RT}
  "fast rhs method"
  fastsolver::FS
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

    Nstages = size(α, 1)

    yn = similar(Q)
    ΔYnj = ntuple(_ -> similar(Q), Nstages-2)
    fYnj = ntuple(_ -> similar(Q), Nstages-1)
    offset = similar(Q)
    tsfastrhs! = TimeScaledRHS(RT(0), RT(0), fastrhs!)
    fastsolver = fastmethod(tsfastrhs!, Q)

    d = sum(β, dims=2)

    c = similar(d)
    for i = eachindex(c)
      c[i] = d[i]
      if i > 1
        c[i] += sum(j-> (α[i,j] + γ[i,j])*c[j], 1:i-1)
      end
    end
    c̃ = α*c

    new{T, RT, AT, typeof(fastsolver), Nstages, Nstages-1, Nstages-2, Nstages ^ 2}(RT(dt), RT(t0), yn, ΔYnj, fYnj, offset,
                                         slowrhs!, tsfastrhs!, fastsolver, nsubsteps,
                                         α, β, γ, d, c, c̃)
  end
end


# TODO almost identical functions seem to be defined for every ode solver,
# define a common one in ODEsolvers ?
function dostep!(Q, mis::MultirateInfinitesimalStep, param,
                      timeend::Real, adjustfinalstep::Bool)
  time, dt = mis.t, mis.dt
  @assert dt > 0
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end

  dostep!(Q, mis, param, time, dt)

  if dt == mis.dt
    mis.t += dt
  else
    mis.t = timeend
  end
  return mis.t
end

function dostep!(Q, mis::MultirateInfinitesimalStep, p,
                      time::Real, dt::AbstractFloat)
  FT = eltype(dt)
  α = mis.α
  β = mis.β
  γ = mis.γ
  yn = mis.yn
  ΔYnj = mis.ΔYnj
  fYnj = mis.fYnj
  offset = mis.offset
  d = mis.d
  c = mis.c
  c̃ = mis.c̃
  slowrhs! = mis.slowrhs!
  fastsolver = mis.fastsolver
  fastrhs! = mis.tsfastrhs!

  nstages = size(α, 1)

  copyto!(yn, Q) # first stage
  for i = 2:nstages
    slowrhs!(fYnj[i-1], Q, p, time + c[i-1]*dt, increment=false)

    threads = 256
    blocks = div(length(realview(Q)) + threads - 1, threads)
    @launch(device(Q), threads=threads, blocks=blocks,
            update!(realview(Q), realview(offset), Val(i), realview(yn), map(realview, ΔYnj[1:i-2]), map(realview, fYnj[1:i-1]),
            α[i,:], β[i,:], γ[i,:], d[i], dt))

    fastrhs!.a = time + c̃[i]*dt
    fastrhs!.b = (c[i] - c̃[i]) / d[i]

    τ = zero(FT)
    dτ = d[i] * dt / mis.nsubsteps
    updatetime!(fastsolver, τ)
    updatedt!(fastsolver, dτ)
    # TODO: we want to be able to write
    #   solve!(Q, fastsolver, p; numberofsteps = mis.nsubsteps)  #(1c)
    # especially if we want to use StormerVerlet, but need some way to pass in `offset`
    for k = 1:mis.nsubsteps
      dostep!(Q, fastsolver, p, τ, dτ, FT(1), realview(offset))  #(1c)
      τ += dτ
    end
  end
end

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

function TVDMISA(slowrhs!, fastrhs!, fastmethod, nsubsteps,  Q::AT; dt=0, t0=0) where {AT <: AbstractArray}
  FT = eltype(Q)
  RT = real(FT)

  α = [0 0                    0                  0;
       0 0                    0                  0;
       0 0.1946360605647457   0                  0;
       0 0.3971200136786614   0.2609434606211801 0]

  β = [ 0                    0                   0                   0;
        RT(2//3)             0                   0                   0;
       -0.28247174703488398  RT(4//9)            0                   0;
       -0.31198081960042401  0.18082737579913699 RT(9//16)           0]

  γ = [0  0                    0                    0;
       0  0                    0                    0;
       0  0.5624048933209129   0                    0;
       0  0.4408467475713277  -0.2459300561692391   0]

  MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod, nsubsteps, α, β, γ, Q; dt=dt, t0=t0)
end

function TVDMISB(slowrhs!, fastrhs!, fastmethod, nsubsteps,  Q::AT; dt=0, t0=0) where {AT <: AbstractArray}
  FT = eltype(Q)
  RT = real(FT)

  α = [0 0                     0                   0;
       0 0                     0                   0;
       0 0.42668232863311001   0                   0;
       0 0.26570779016173801   0.41489966891866698 0]

  β = [ 0                    0                   0                   0;
        RT(2//3)             0                   0                   0;
       -0.25492859100078202  RT(4//9)            0                   0;
       -0.26452517179288798  0.11424084424766399 RT(9//16)           0]

  γ = [0  0                    0                    0;
       0  0                    0                    0;
       0  0.28904389120139701  0                    0;
       0  0.45113560071334202 -0.25006656847591002  0]

  MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod, nsubsteps, α, β, γ, Q; dt=dt, t0=t0)
end
