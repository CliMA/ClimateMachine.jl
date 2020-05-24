
export ETDStep, EB4, EB1, ETDRK3

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
  Nstages_sq,
  Nβ,
} <: AbstractODESolver
  "time step"
  dt::RT
  "time"
  t::RT
  "storage for y_n"
  yn::AT
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

  nStages::Int
  nPhi::Int
  nPhiStages::SArray{NTuple{1, Nstages}, Int, 1, Nstages}

  β::NTuple{Nβ, SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}};
  βS::NTuple{Nβ, SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}};

  c::SArray{NTuple{1, Nstages}, RT, 1, Nstages}

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

    yn = similar(Q)
    fYnj = ntuple(_ -> similar(Q), Nstages-1)
    offset = similar(Q)
    tsfastrhs! = TimeScaledRHS(RT(0), RT(0), fastrhs!)
    fastsolver = fastmethod(tsfastrhs!, Q)

    for i=2:Nstages
      for j=1:i-1
        kFac=1;
          for k=1:nPhi
            kFac=kFac*max(k-1,1);
            βS[k][i,j]=β[k][i,j]/(kFac*c[i]);
            β[k][i,j]=β[k][i,j]/c[i];
          end
      end
    end

    new{
      T,
      RT,
      AT,
      typeof(fastsolver),
      Nstages,
      Nstages-1,
      Nstages ^ 2,
      length(β),
    }(
      RT(dt),
      RT(t0),
      yn,
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
  yn = etd.yn
  fYnj = etd.fYnj
  offset = etd.offset
  c = etd.c
  slowrhs! = etd.slowrhs!
  fastsolver = etd.fastsolver
  fastrhs! = etd.tsfastrhs!
  nsteps=etd.nsteps

  nStages = etd.nStages

  copyto!(yn, Q) # first stage
  for iStage = 1:nStages-1
    slowrhs!(fYnj[iStage], Q, p, time + c[iStage]*dt, increment=false)

    nstepsLoc=ceil(Int,nsteps*c[iStage+1]);
    dτ=dt*c[iStage+1]/nstepsLoc;

    copyto!(Q, yn)
    dostep!(Q, fastsolver, p, time, dτ, nstepsLoc, iStage, β, βS, nPhi, fYnj, FT(1), realview(offset), nothing)
  end
end


function EB1(slowrhs!, fastrhs!, fastmethod, nsteps, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}

  T = eltype(Q)
  RT = real(T)

  nStages=2;
  nPhi=1;

  nPhiStages=[0, 1];

  β0 = [RT(0) RT(0)
        RT(1) RT(0)];
  βS0 = zeros(RT,2,2);

  c = [RT(0) RT(1) RT(1)];

  ETDStep(slowrhs!, fastrhs!, fastmethod, nsteps, nStages, nPhi, nPhiStages, (β0,), (βS0,), c, Q; dt=dt, t0=t0)
end

function ETDRK3(slowrhs!, fastrhs!, fastmethod, nsteps, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}

    T = eltype(Q)
    RT = real(T)

    nStages=4;
    nPhi=1;

    nPhiStages=[0,1,1,1];

    β0 = [
            RT(0)         RT(0)         RT(0)     RT(0)
            RT(1 // 3)    RT(0)         RT(0)     RT(0)
            RT(0)         RT(1 // 2)    RT(0)     RT(0)
            RT(0)         RT(0)         RT(1)     RT(0)
    ];
    βS0 = zeros(RT,4,4);
    β1 = zeros(RT,4,4);
    βS1 = zeros(RT,4,4);
    β2 = zeros(RT,4,4);
    βS2 = zeros(RT,4,4);

    c = [RT(0) RT(1 // 3) RT(1 // 2) RT(1)];

    ETDStep(slowrhs!, fastrhs!, fastmethod, nsteps, nStages, nPhi, nPhiStages, (β0, β1, β2), (βS0, βS1, βS2), c, Q; dt=dt, t0=t0)
end

function EB4(slowrhs!, fastrhs!, fastmethod, nsteps, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}

  T = eltype(Q)
  RT = real(T)

  nStages=5;
  nPhi=3;

  nPhiStages=[0,1,2,2,3];

  β0 = [
          RT(0)         RT(0)     RT(0)     RT(0)     RT(0)
          RT(1 // 2)    RT(0)     RT(0)     RT(0)     RT(0)
          RT(1 // 2)    RT(0)     RT(0)     RT(0)     RT(0)
          RT(1)         RT(0)     RT(1)     RT(0)     RT(0)
          RT(1)         RT(0)     RT(0)     RT(0)     RT(0)
  ];
  βS0 = zeros(RT,5,5);
  β1 = [
          RT(0)       RT(0)     RT(0)     RT(0)     RT(0)
          RT(0)       RT(0)     RT(0)     RT(0)     RT(0)
          RT(-1)      RT(1)     RT(0)     RT(0)     RT(0)
          RT(-2)      RT(0)     RT(2)     RT(0)     RT(0)
          RT(-3)      RT(2)     RT(2)     RT(-1)    RT(0)
  ];
  βS1 = zeros(RT,5,5);
  β1 = [
          RT(0)      RT(0)     RT(0)     RT(0)     RT(0)
          RT(0)      RT(0)     RT(0)     RT(0)     RT(0)
          RT(0)      RT(0)     RT(0)     RT(0)     RT(0)
          RT(0)      RT(0)     RT(0)     RT(0)     RT(0)
          RT(4)      RT(-4)    RT(-4)    RT(4)     RT(0)
  ];
  βS2 = zeros(RT,5,5);

  c = [RT(0) RT(1 // 2) RT(1 // 2) RT(1) RT(1)];

  ETDStep(slowrhs!, fastrhs!, fastmethod, nsteps, nStages, nPhi, nPhiStages, (β0, β1, β2), (βS0, βS1, βS2), c, Q; dt=dt, t0=t0)
end
