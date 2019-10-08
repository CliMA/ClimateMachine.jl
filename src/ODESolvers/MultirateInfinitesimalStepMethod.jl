module MultirateInfinitesimalStepMethod
using ..ODESolvers
using ..MPIStateArrays: device, realview

using StaticArrays

export MIS2, MIS3C

ODEs = ODESolvers

struct OffsetRHS{O}
  "storage for offset term"
  offset::O
  rhs!
end

function (o::OffsetRHS)(dQ, Q, params, t; increment)
  o.rhs!(dQ, Q, params, t; increment=increment)
  dQ .+= o.offset
end

using GPUifyLoops

struct MultirateInfinitesimalStep{T, RT, AT, Nstages, Nstages_sq} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT, 1}
  "time"
  t::Array{RT, 1}
  "storage for y_n"
  yn::AT
  "Storage for ``Y_nj - y_n``"
  ΔYnj::NTuple{Nstages, AT}
  "Storage for ``f(Y_nj)``"
  fYnj::NTuple{Nstages, AT}
  "slow rhs function"
  slowrhs!
  "RHS for fast solver"
  fastrhs!::OffsetRHS{AT}
  "fast rhs method"
  fastmethod
  α::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq} 
  β::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  γ::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq} 

  function MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod,
                                      α, β, γ,
                                      Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)
    dt = [dt]
    t0 = [t0]

    Nstages = size(α, 1)

    yn = similar(Q)
    ΔYnj = ntuple(_ -> similar(Q), Nstages)
    fYnj = ntuple(_ -> similar(Q), Nstages)
    fastrhs! = OffsetRHS(similar(Q), fastrhs!)
    
    new{T, RT, AT, Nstages, Nstages ^ 2}(dt, t0, yn, ΔYnj, fYnj,
                                         slowrhs!, fastrhs!, fastmethod,
                                         α, β, γ)
  end
end

# TODO write this function
#function MultirateInfinitesimalStep(spacedisc::AbstractSpaceMethod, RKA, RKB, RKC,
#                                Q::AT; dt=0, t0=0) where {AT<:AbstractArray}
#  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x..., increment = increment)
#  MultirateInfinitesimalStep(rhs!, RKA, RKB, RKC, Q; dt=dt, t0=t0)
#end

function MIS2(slowrhs!, fastrhs!, fastmethod, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}
  T = eltype(Q)
  RT = real(T)

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

  MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod, α, β, γ, Q; dt=dt, t0=t0)
end

function MIS3C(slowrhs!, fastrhs!, fastmethod, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}
  T = eltype(Q)
  RT = real(T)

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

  MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod, α, β, γ, Q; dt=dt, t0=t0)
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
  DT = eltype(dt)
  α = mis.α
  β = mis.β
  γ = mis.γ
  yn = mis.yn
  ΔYnj = mis.ΔYnj
  fYnj = mis.fYnj
  slowrhs! = mis.slowrhs!
  fastmethod = mis.fastmethod
  fastrhs! = mis.fastrhs!

  nstages = size(α, 1)

  # first stage
  copyto!(yn, Q)
  fill!(ΔYnj[1], 0)
  slowrhs!(fYnj[1], yn, p, time, increment=false)

  for i = 2:nstages
    d_i = sum(j -> β[i, j], 1:i-1)

    # TODO write a kernel to do this
    Q .= yn .+ sum(j -> α[i, j] .* ΔYnj[j], 1:i-1)  # (1a) 
    fastrhs!.offset .= sum(j -> γ[i, j] .* ΔYnj[j] ./ dt .+ β[i,j] .* fYnj[j], 1:i-1) ./ d_i # (1b) 

    tau = zero(DT)
    fastdt = d_i * dt # TODO substepping
    fastsolver = fastmethod(fastrhs!, Q; dt=fastdt, t0=tau)

    solve!(Q, fastsolver, p; timeend = d_i * dt) #(1c)
    #while tau < t + ??
    #  ODEs.dostep!(Q, fastsolver, fastrhs!, param, tau, dt)
    #end

    if i < nstages
      t = time # TODO evaluate slow rhs at proper times
      slowrhs!(fYnj[i], Q, p, t, increment=false)
      @. ΔYnj[i] = Q - yn
    end
  end
end

end
