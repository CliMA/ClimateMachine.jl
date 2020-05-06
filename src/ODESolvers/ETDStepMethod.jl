
export EB4, EB1, ETDRK3

"""
ETDStep(slowrhs!, fastrhs!, fastmethod,
                           α, β, γ,
                           Q::AT; dt=0, t0=0) where {AT<:AbstractArray}
This is a time stepping object for explicitly time stepping the partitioned differential
equation given by right-hand-side functions `f_fast` and `f_slow` with the state `Q`, i.e.,
```math
  \\dot{Q} = f_fast(Q, t) + f_slow(Q, t)
```
with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.
The constructor builds an exponential time differencing  Runge-Kutta scheme
based on the provided `β` tableaux and `fastmethod` for solving
the fast modes.
### References
    @article{Krogstad2005,
      title={Generalized integrating factor methods for stiff PDEs},
      author={Krogstad, Stein},
      journal= {Journal of Computational Physics},
      volume= {203},
      number= {1},
      pages = {72 - 88},
      year = {2005},
    }
"""
mutable struct ETDStep{
  T,
  RT,
  AT,
  FS,
  Nstages,
  Nstagesm1,
  Nstagesm2,
  Nstages_sq
} <: AbstractODESolver
  "time step"
  dt::RT
  "time"
  t::RT
  #"storage for y_n"
  #yn::AT
  "Storage for ``f(Y_nj)``"
  fYnj::NTuple{Nstagesm1, AT}
  "Storage for offset"
  offset::AT
  "slow rhs function"
  slowrhs!
  "RHS for fast solver"
  tsfastrhs!::TimeScaledRHS{N,RT} where N
  "fast rhs method"
  fastsolver::FS
  "number of steps"
  nsteps::Int

  nStages::Int64
  nPhi::Int64
  nPhiStages::Array{Int64,1}

  #α::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  #β::SArray{NTuple{2, Nstages}, AT, 2, Nstages_sq}
  #βS::SArray{NTuple{2, Nstages}, AT, 2, Nstages_sq}
  β::Array{Array{Float64,1},2};
  βS::Array{Array{Float64,1},2};

  #γ::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  d::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
  c::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
  #c̃::SArray{NTuple{1, Nstages}, RT, 1, Nstages}

  function ETDStep(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsteps,
    Nstages,
    nPhi,
    nPhiStages,
    β,
    βS,
    c,
    Q::AT;
    dt=0,
    t0=0
  ) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)


    #yn = similar(Q)
    #ΔYnj = ntuple(_ -> similar(Q), Nstages-2)
    fYnj = ntuple(_ -> similar(Q), Nstages-1)
    offset = similar(Q)
    tsfastrhs! = TimeScaledRHS(RT(0), RT(0), fastrhs!)
    fastsolver = fastmethod(tsfastrhs!, Q)

    #d = sum(β, dims=2)

    #c = similar(d)
    d=copy(c);

    #for i = eachindex(c)
      #c[i] = d[i]
      #if i > 1
      #   c[i] += sum(j-> (α[i,j] + γ[i,j])*c[j], 1:i-1)
      #end
    #end
    #c̃ = α*c

    #new{T, RT, AT, typeof(fastsolver), Nstages, Nstages-1, Nstages-2, Nstages ^ 2}(RT(dt), RT(t0), yn, ΔYnj, fYnj, offset,
    #                                       slowrhs!, tsfastrhs!, fastsolver,
    #                                       α, β, γ, d, c, c̃)
    new{
      T,
      RT,
      AT,
      typeof(fastsolver),
      Nstages,
      Nstages-1,
      Nstages-2,
      Nstages ^ 2
    }(
      RT(dt),
      RT(t0),
      fYnj,
      offset,
      slowrhs!,
      tsfastrhs!,
      fastsolver,
      nsteps,
      Nstages,
      nPhi,
      nPhiStages,
      β,
      βS,
      d,
      c
    )
  end
end

function dostep!(Q, etd::ETDStep, p, time)

  dt = etd.dt
  FT = eltype(dt)
  β = etd.β
  βS = etd.βS
  nPhi = etd.nPhi
  #yn = etd.yn
  fYnj = etd.fYnj
  offset = etd.offset
  d = etd.d
  c = etd.c
  #c̃ = etd.c̃
  slowrhs! = etd.slowrhs!
  fastsolver = etd.fastsolver
  fastrhs! = etd.tsfastrhs!
  nsteps=etd.nsteps

  nStages = etd.nStages

  #copyto!(yn, Q) # first stage
  for iStage = 1:nStages-1
    slowrhs!(fYnj[iStage], Q, p, time + c[iStage]*dt, increment=false)

    nstepsLoc=ceil(Int,nsteps*c[iStage+1]);
    dτ=dt*c[iStage+1]/nstepsLoc;

    dostep!(Q, fastsolver, p, time, dτ, nstepsLoc, iStage, β, βS, nPhi, fYnj, FT(1), realview(offset), nothing)  #(1c)
  end
end

function EB1(slowrhs!, fastrhs!, fastmethod, nsteps, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}

  nStages=2;
  nPhi=1;

  nPhiStages=[0, 1]; #???

  β = [[[0.0]] [[0.0]];
      [[1.0]] [[0.0]]];
  βS =[[[0.0]] [[0.0]];
      [[0.0]] [[0.0]]];


  c = [0.0, 1.0, 1.0];

  for i=2:2
    for j=1:i-1
    kFac=1;
      for k=1:1
        kFac=kFac*max(k-1,1)*c[i];
        βS[i,j][k]=β[i,j][k]/kFac;
      end
    end
  end

  ETDStep(slowrhs!, fastrhs!, fastmethod, nsteps, nStages, nPhi, nPhiStages, β, βS, c, Q; dt=dt, t0=t0)
end

function ETDRK3(slowrhs!, fastrhs!, fastmethod, nsteps, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}

    nStages=4;
    nPhi=1;

    nPhiStages=[0,1,1,1];

    β = [[[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
         [[1.0/3.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
         [[0.0,0.0,0.0]] [[1.0/2.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
         [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[1.0,0.0,0.0]] [[0.0,0.0,0.0]]];

    #βS=similar(β);

    βS = [[[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
          [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
          [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
          [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]]];

    c = [0.0, 0.33333333333333333, 0.5, 1.0];
       #c[i] is usually sum of first elements in i-th row)

    for i=2:nStages
      for j=1:i-1
        kFac=1;
          for k=1:nPhi
            kFac=kFac*max(k-1,1);
            βS[i,j][k]=β[i,j][k]/(kFac*c[i]);
            β[i,j][k]=β[i,j][k]/c[i];
          end
      end
    end

    #=γ = [0  0               0              0;
         0  0               0              0;
         0  0.652465126004  0              0;
         0 -0.0732769849457 0.144902430420 0]=# #not needed (yet?)

    ETDStep(slowrhs!, fastrhs!, fastmethod, nsteps, nStages, nPhi, nPhiStages, β, βS, c, Q; dt=dt, t0=t0)
end

function EB4(slowrhs!, fastrhs!, fastmethod, nsteps, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}

  nStages=5;
  nPhi=3;

  nPhiStages=[0,1,2,2,3];

  β = [[[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]];
       [[0.5,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]];
       [[0.5,-1.0,0.0]] [[0.0,1.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]];
       [[1.0,-2.0,0.0]] [[0.0,0.0,0.0]]  [[0.0,2.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]];
       [[1.0,-3.0,4.0]] [[0.0,2.0,-4.0]] [[0.0,2.0,-4.0]] [[0.0,-1.0,4.0]] [[0.0,0.0,0.0]]];

  #βS=similar(β);
  βS = [[[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
        [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
        [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
        [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
        [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]]];

  c = [0.0, 0.5, 0.5, 1.0, 1.0];
     #c[i] is usually sum of first elements in i-th row)

  for i=2:nStages
    for j=1:i-1
      kFac=1;
        for k=1:nPhi
          kFac=kFac*max(k-1,1);
          βS[i,j][k]=β[i,j][k]/(kFac*c[i]);
          β[i,j][k]=β[i,j][k]/c[i];
        end
    end
  end

  #=γ = [0  0               0              0;
       0  0               0              0;
       0  0.652465126004  0              0;
       0 -0.0732769849457 0.144902430420 0]=# #not needed (yet?)

  ETDStep(slowrhs!, fastrhs!, fastmethod, nsteps, nStages, nPhi, nPhiStages, β, βS, c, Q; dt=dt, t0=t0)
end
