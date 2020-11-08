
export ETDStep, EB4, EB1, ETDRK3

"""
ETDStep(slowrhs!, fastrhs!, fastmethod, α, β, γ, Q; dt=0, t0=0)

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
    Nβ
} <: AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "elapsed time steps"
    steps::Int
    "storage for y_n"
    yn::AT
    "Storage for ``f(Y_nj)``"
    fYnj::NTuple{Nstagesm1,AT}
    "slow rhs function"
    slowrhs!::Any
    "RHS for fast solver"
    tsfastrhs!::TimeScaledRHS{N,RT} where {N}
    "fast rhs method"
    fastsolver::FS
    "number of steps"
    nsubsteps::Int
    nPhi::Int
    nPhiStages::SArray{NTuple{1,Nstages},Int,1,Nstages}
    β::NTuple{Nβ,SArray{NTuple{2,Nstages},RT,2,Nstages_sq}}
    βs::NTuple{Nβ,SArray{NTuple{2,Nstages},RT,2,Nstages_sq}}
    c::SArray{NTuple{1,Nstages},RT,1,Nstages}

    function ETDStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        Nstages,
        nPhi,
        nPhiStages,
        β,
        βs,
        c,
        Q::AT;
        dt = 0,
        t0 = 0,
    ) where {AT<:AbstractArray}

        T = eltype(Q)
        RT = real(T)

        Nstages = size(β[1], 1)
        Nβ = length(β)

        yn = similar(Q)
        fYnj = ntuple(_ -> similar(Q), Nstages - 1)
        tsfastrhs! = TimeScaledRHS(RT(0), RT(0), fastrhs!)
        fastsolver = fastmethod(tsfastrhs!, Q)

        for i = 2:Nstages
            for j = 1:i-1
                kFac = 1
                for k = 1:nPhi
                    kFac = kFac * max(k - 1, 1)
                    βs[k][i, j] = β[k][i, j] / (kFac * c[i])
                    β[k][i, j] /= c[i]
                end
            end
        end

        new{T,RT,AT,typeof(fastsolver),Nstages,Nstages - 1,Nstages^2,Nβ}(
            RT(dt),
            RT(t0),
            0,
            yn,
            fYnj,
            slowrhs!,
            tsfastrhs!,
            fastsolver,
            nsubsteps,
            Nstages,
            nPhi,
            nPhiStages,
            β,
            βs,
            c,
        )
    end
end

function dostep!(Q, etd::ETDStep, p, time)
    dt = etd.dt
    FT = eltype(dt)
    β = etd.β
    βs = etd.βs
    c = etd.c
    nPhi = etd.nPhi
    yn = etd.yn
    fYnj = etd.fYnj
    slowrhs! = etd.slowrhs!
    fastsolver = etd.fastsolver
    fastrhs! = etd.tsfastrhs!
    nsubsteps = etd.nsubsteps

    Nstages = size(β[1], 1)
    Nβ = length(β)
    ts = time

    copyto!(yn, Q) # first stage
    for iStage = 1:Nstages-1
        dts = c[iStage+1] * dt

        slowrhs!(fYnj[iStage], Q, p, time + c[iStage] * dt, increment = false)

        τ = zero(FT)
        nsubstepsLoc = ceil(Int, nsubsteps * c[iStage+1])
        dτ = dt * c[iStage+1] / nsubstepsLoc
        updatetime!(fastsolver, τ)
        updatedt!(fastsolver, dτ)

        βrow = ntuple(k -> ntuple(j -> βs[k][iStage+1, j], iStage), Nβ)
        mriparam = MRIParam(p, βrow, realview.(fYnj[1:iStage]), ts, dts)

        copyto!(Q, yn)
        dostep!(
            Q,
            fastsolver,
            mriparam,
            ts,
            nsubstepsLoc,
            iStage,
            nothing,
            nothing,
            nothing,
        )
    end
end


function EB1(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)

    nPhi = 1
    nPhiStages = [0, 1]

    β0 = [
        RT(0) RT(0)
        RT(1) RT(0)
    ]
    βs0 = zeros(RT, 2, 2)

    c = [RT(0) RT(1) RT(1)]

    ETDStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        nPhi,
        nPhiStages,
        (β0,),
        (βs0,),
        c,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function ETDRK3(slowrhs!, fastrhs!, fastmethod, nsubsteps, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}

    T = eltype(Q)
    RT = real(T)

    nPhi=1;
    nPhiStages=[0,1,1,1];

    β0 = [
            RT(0)         RT(0)         RT(0)     RT(0)
            RT(1 // 3)    RT(0)         RT(0)     RT(0)
            RT(0)         RT(1 // 2)    RT(0)     RT(0)
            RT(0)         RT(0)         RT(1)     RT(0)
    ];
    βs0 = zeros(RT,4,4);
    β1 = zeros(RT,4,4);
    βs1 = zeros(RT,4,4);
    β2 = zeros(RT,4,4);
    βs2 = zeros(RT,4,4);

    c = [RT(0) RT(1 // 3) RT(1 // 2) RT(1)];

    ETDStep(slowrhs!, fastrhs!, fastmethod, nsubsteps, nPhi, nPhiStages, (β0, β1, β2), (βs0, βs1, βs2), c, Q; dt=dt, t0=t0)
end

function EB4(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)

    nPhi = 3
    nPhiStages = [0, 1, 2, 2, 3]

    β0 = [
        RT(0) RT(0) RT(0) RT(0) RT(0)
        RT(1 // 2) RT(0) RT(0) RT(0) RT(0)
        RT(1 // 2) RT(0) RT(0) RT(0) RT(0)
        RT(1) RT(0) RT(1) RT(0) RT(0)
        RT(1) RT(0) RT(0) RT(0) RT(0)
    ]
    βs0 = zeros(RT, 5, 5)
    β1 = [
        RT(0) RT(0) RT(0) RT(0) RT(0)
        RT(0) RT(0) RT(0) RT(0) RT(0)
        RT(-1) RT(1) RT(0) RT(0) RT(0)
        RT(-2) RT(0) RT(2) RT(0) RT(0)
        RT(-3) RT(2) RT(2) RT(-1) RT(0)
    ]
    βs1 = zeros(RT, 5, 5)
    β2 = [
        RT(0) RT(0) RT(0) RT(0) RT(0)
        RT(0) RT(0) RT(0) RT(0) RT(0)
        RT(0) RT(0) RT(0) RT(0) RT(0)
        RT(0) RT(0) RT(0) RT(0) RT(0)
        RT(4) RT(-4) RT(-4) RT(4) RT(0)
    ]
    βs2 = zeros(RT, 5, 5)

    c = [RT(0) RT(1 // 2) RT(1 // 2) RT(1) RT(1)]

    ETDStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        nPhi,
        nPhiStages,
        (β0, β1, β2),
        (βs0, βs1, βs2),
        c,
        Q;
        dt = dt,
        t0 = t0,
    )
end
