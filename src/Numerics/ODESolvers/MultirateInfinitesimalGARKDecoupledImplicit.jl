export MRIGARKDecoupledImplicit
export MRIGARKIRK21aSandu,
    MRIGARKESDIRK34aSandu,
    MRIGARKESDIRK46aSandu,
    MRIGARKESDIRK23LSA,
    MRIGARKESDIRK24LSA

"""
    MRIGARKDecoupledImplicit(f!, backward_euler_solver, fastsolver, Γs, γ̂s, Q,
                             Δt, t0)

Construct a decoupled implicit MultiRate Infinitesimal General-structure
Additive Runge--Kutta (MRI-GARK) scheme to solve

```math
    \\dot{y} = f(y, t) + g(y, t)
```

where `f` is the slow tendency function and `g` is the fast tendency function;
see Sandu (2019).

The fast tendency is integrated using the `fastsolver` and the slow tendency
using the MRI-GARK scheme. Since this is a decoupled, implicit MRI-GARK there is no implicit coupling between the fast and slow tendencies.

The `backward_euler_solver` should be of type `AbstractBackwardEulerSolver` or
`LinearBackwardEulerSolver`, and is used to perform the backward Euler solves 
for `y` given the slow tendency function, namely

```math
   y = z + α f(y, t; p)
```

Currently only ['LowStorageRungeKutta2N`](@ref) schemes are supported for
`fastsolver`

The coefficients defined by `γ̂s` can be used for an embedded scheme (only the
last stage is different).

The available concrete implementations are:

  - [`MRIGARKIRK21aSandu`](@ref)
  - [`MRIGARKESDIRK34aSandu`](@ref)
  - [`MRIGARKESDIRK46aSandu`](@ref)

### References

    @article{Sandu2019,
        title={A class of multirate infinitesimal gark methods},
        author={Sandu, Adrian},
        journal={SIAM Journal on Numerical Analysis},
        volume={57},
        number={5},
        pages={2300--2327},
        year={2019},
        publisher={SIAM},
        doi={10.1137/18M1205492}
    }
"""
mutable struct MRIGARKDecoupledImplicit{
    T,
    RT,
    AT,
    Nstages,
    NΓ,
    FS,
    Nx,
    Ny,
    Nx_Ny,
    BE,
} <: AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "rhs function"
    slowrhs!
    "backwark Euler solver"
    besolver!::BE
    "Storage for RHS during the `MRIGARKDecoupledImplicit` update"
    Rstages::NTuple{Nstages, AT}
    "Storage for the implicit solver data vector"
    Qhat::AT
    "RK coefficient matrices for coupling coefficients"
    Γs::NTuple{NΓ, SArray{Tuple{Nx, Ny}, RT, 2, Nx_Ny}}
    "RK coefficient matrices for embedded scheme"
    γ̂s::NTuple{NΓ, SArray{NTuple{1, Ny}, RT, 1, Ny}}
    "RK coefficient vector C (time scaling)"
    Δc::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
    "fast solver"
    fastsolver::FS

    function MRIGARKDecoupledImplicit(
        slowrhs!,
        backward_euler_solver,
        fastsolver,
        Γs,
        γ̂s,
        Q::AT,
        dt,
        t0,
    ) where {AT <: AbstractArray}
        NΓ = length(Γs)

        T = eltype(Q)
        RT = real(T)

        # Compute the Δc coefficients (only explicit values kept)
        Δc = sum(Γs[1], dims = 2)

        # Couple of sanity checks on the assumptions of coefficients being of
        # the decoupled implicit structure of Sandu (2019)
        @assert all(isapprox.(Δc[2:2:end], 0; atol = 2 * eps(RT)))

        Δc = Δc[1:2:(end - 1)]

        # number of slow RHS values we need to keep
        Nstages = length(Δc)

        # Couple more sanity checks on the decoupled implicit structure
        @assert Nstages == size(Γs[1], 2) - 1
        @assert Nstages == div(size(Γs[1], 1), 2)

        # Scale in the Δc to the Γ and γ̂, and convert to real type
        Γs = ntuple(k -> RT.(Γs[k]), NΓ)
        γ̂s = ntuple(k -> RT.(γ̂s[k]), NΓ)

        # Convert to real type
        Δc = RT.(Δc)

        # create storage for the stage values
        Rstages = ntuple(i -> similar(Q), Nstages)
        Qhat = similar(Q)

        FS = typeof(fastsolver)
        Nx, Ny = size(Γs[1])

        # Set up the backward Euler solver with the initial value of α
        α = dt * Γs[1][2, 2]
        besolver! =
            setup_backward_Euler_solver(backward_euler_solver, Q, α, slowrhs!)
        @assert besolver! isa AbstractBackwardEulerSolver
        BE = typeof(besolver!)

        new{T, RT, AT, Nstages, NΓ, FS, Nx, Ny, Nx * Ny, BE}(
            RT(dt),
            RT(t0),
            slowrhs!,
            besolver!,
            Rstages,
            Qhat,
            Γs,
            γ̂s,
            Δc,
            fastsolver,
        )
    end
end

function updatedt!(mrigark::MRIGARKDecoupledImplicit, dt)
    @assert Δt_is_adjustable(mrigark.besolver!)
    α = dt * mrigark.Γs[1][2, 2]
    update_backward_Euler_solver!(mrigark.besolver!, mrigark.Qhat, α)
    mrigark.dt = dt
end

function dostep!(Q, mrigark::MRIGARKDecoupledImplicit, param, time::Real)
    dt = mrigark.dt
    fast = mrigark.fastsolver

    Rs = mrigark.Rstages
    Δc = mrigark.Δc

    Nstages = length(Δc)
    groupsize = 256

    slowrhs! = mrigark.slowrhs!
    Γs = mrigark.Γs
    NΓ = length(Γs)

    ts = time
    groupsize = 256

    besolver! = mrigark.besolver!
    Qhat = mrigark.Qhat
    # Since decoupled implicit methods are being used, there is an purely
    # explicit stage followed by an implicit correction stage, hence the divide
    # by two
    for s in 1:Nstages
        # Stage dt
        dts = Δc[s] * dt
        stage_end_time = ts + dts

        # initialize the slow tendency stage value
        slowrhs!(Rs[s], Q, param, ts, increment = false)

        # # advance fast solution to time stage_end_time
        γs = ntuple(k -> ntuple(j -> Γs[k][2s - 1, j] / Δc[s], s), NΓ)
        mriparam = MRIParam(param, γs, realview.(Rs[1:s]), ts, dts)
        updatetime!(mrigark.fastsolver, ts)
        solve!(Q, mrigark.fastsolver, mriparam; timeend = stage_end_time)

        # correct with implicit slow solve
        # Qhat = Q + ∑_j Σ_k Γ_{sjk} dt Rs[j] / k
        # (Divide by k arises from the integration γ_{ij}(τ) in Sandu (2019);
        # see Equation (2.2b) and Definition 2.2
        γs = ntuple(k -> ntuple(j -> dt * Γs[k][2s, j] / k, s), NΓ)
        event = Event(array_device(Q))
        event = mri_create_Qhat!(array_device(Q), groupsize)(
            realview(Qhat),
            realview(Q),
            γs,
            mriparam.Rs;
            ndrange = length(realview(Q)),
            dependencies = (event,),
        )
        wait(array_device(Q), event)

        # Solve: Q = Qhat + α fslow(Q, stage_end_time)
        α = dt * Γs[1][2s, s + 1]
        besolver!(Q, Qhat, α, param, stage_end_time)

        # update time
        ts += dts
    end
end

# Compute: Qhat = Q + ∑_j Σ_k Γ_{sjk} dt Rs[j] / k
@kernel function mri_create_Qhat!(Qhat, Q, γs, Rs)
    i = @index(Global, Linear)
    @inbounds begin
        NΓ = length(γs)
        Ns = length(γs[1])
        qhat = Q[i]

        for s in 1:Ns
            ri = Rs[s][i]
            sc = γs[1][s]
            for k in 2:NΓ
                sc += γs[k][s]
            end
            qhat += sc * ri
        end
        Qhat[i] = qhat
    end
end

"""
    MRIGARKIRK21aSandu(f!, fastsolver, Q; dt, t0 = 0)

The 2rd order, 2 stage implicit scheme from Sandu (2019).
"""
function MRIGARKIRK21aSandu(
    slowrhs!,
    backward_euler_solver,
    fastsolver,
    Q;
    dt,
    t0 = 0,
)
    #! format: off
    Γ0 = [ 1 // 1 0 // 1
          -1 // 2 1 // 2 ]
    γ̂0 = [-1 // 2 1 // 2 ]
    #! format: on
    MRIGARKDecoupledImplicit(
        slowrhs!,
        backward_euler_solver,
        fastsolver,
        (Γ0,),
        (γ̂0,),
        Q,
        dt,
        t0,
    )
end

"""
    MRIGARKESDIRK34aSandu(f!, fastsolver, Q; dt, t0=0)

The 3rd order, 4 stage decoupled implicit scheme from Sandu (2019).
"""
function MRIGARKESDIRK34aSandu(
    slowrhs!,
    backward_euler_solver,
    fastsolver,
    Q;
    dt,
    t0 = 0,
)
    T = real(eltype(Q))
    μ = acot(2 * sqrt(T(2))) / 3
    λ = 1 - cos(μ) / sqrt(T(2)) + sqrt(T(3 // 2)) * sin(μ)
    @assert isapprox(-1 + 9λ - 18 * λ^2 + 6 * λ^3, 0, atol = 2 * eps(T))

    #! format: off
    Γ0 = [
          T(1 // 3)               0                        0           0
          -λ                      λ                        0           0
          (3-10λ) / (24λ-6)       (5-18λ) / (6-24λ)        0           0
          (-24λ^2+6λ+1) / (6-24λ) (-48λ^2+12λ+1) / (24λ-6) λ           0
          (3-16λ) / (12-48λ)      (48λ^2-21λ+2) / (12λ-3)  (3-16λ) / 4 0
          -λ                      0                        0           λ
         ]
    γ̂0 = [ 0                      0                        0           0]
    #! format: on
    MRIGARKDecoupledImplicit(
        slowrhs!,
        backward_euler_solver,
        fastsolver,
        (Γ0,),
        (γ̂0,),
        Q,
        dt,
        t0,
    )
end

"""
    MRIGARKESDIRK46aSandu(f!, fastsolver, Q; dt, t0=0)

The 4th order, 6 stage decoupled implicit scheme from Sandu (2019).
"""
function MRIGARKESDIRK46aSandu(
    slowrhs!,
    implicitsolve!,
    fastsolver,
    Q;
    dt,
    t0 = 0,
)
    T = real(eltype(Q))
    μ = acot(2 * sqrt(T(2))) / 3
    λ = 1 - cos(μ) / sqrt(T(2)) + sqrt(T(3 // 2)) * sin(μ)
    @assert isapprox(-1 + 9λ - 18 * λ^2 + 6 * λ^3, 0, atol = 2 * eps(T))

    #! format: off
    Γ0 = [
                         1 // 5                             0 // 1                             0 // 1                              0 // 1                          0 // 1           0 // 1
                        -1 // 4                             1 // 4                             0 // 1                              0 // 1                          0 // 1           0 // 1
             1771023115159 // 1929363690800    -1385150376999 // 1929363690800                 0 // 1                              0 // 1                          0 // 1           0 // 1
                    914009 // 345800                 -1000459 // 345800                        1 // 4                              0 // 1                          0 // 1           0 // 1
            18386293581909 // 36657910125200       5506531089 // 80566835440       -178423463189 // 482340922700                   0 // 1                          0 // 1           0 // 1 
                  36036097 // 8299200                    4621 // 118560                -38434367 // 8299200                        1 // 4                          0 // 1           0 // 1
          -247809665162987 // 146631640500800  10604946373579 // 14663164050080   10838126175385 // 5865265620032    -24966656214317 // 36657910125200             0 // 1           0 // 1
                  38519701 // 11618880               10517363 // 9682400               -23284701 // 19364800               -10018609 // 2904720                    1 // 4           0 // 1
           -52907807977903 // 33838070884800   74846944529257 // 73315820250400  365022522318171 // 146631640500800  -20513210406809 // 109973730375600  -2918009798 // 1870301537  0 // 1
                        19 // 100                         -73 // 300                         127 // 300                          127 // 300                     -313 // 300         1 // 4
         ]

    Γ1 = [
                         0 // 1                             0 // 1                               0 // 1                              0 // 1                          0 // 1           0 // 1
                         0 // 1                             0 // 1                               0 // 1                              0 // 1                          0 // 1           0 // 1
            -1674554930619 // 964681845400      1674554930619 // 964681845400                    0 // 1                              0 // 1                          0 // 1           0 // 1
                  -1007739 // 172900                  1007739 // 172900                          0 // 1                              0 // 1                          0 // 1           0 // 1
            -8450070574289 // 18328955062600     -39429409169 // 40283417720          173621393067 // 120585230675                   0 // 1                          0 // 1           0 // 1
                -122894383 // 16598400                  14501 // 237120                  121879313 // 16598400                       0 // 1                          0 // 1           0 // 1
            32410002731287 // 15434909526400  -46499276605921 // 29326328100160    -34914135774643 // 11730531240064    45128506783177 // 18328955062600             0 // 1           0 // 1
                -128357303 // 23237760              -35433927 // 19364800                 71038479 // 38729600                 8015933 // 1452360                    0 // 1           0 // 1
           136721604296777 // 67676141769600 -349632444539303 // 146631640500800 -1292744859249609 // 293263281001600    8356250416309 // 54986865187800   17282943803 // 3740603074  0 // 1
                         3 // 25                          -29 // 300                            71 // 300                           71 // 300                     -149 // 300         0 // 1
         ]

    γ̂0 = [-1 // 4 5595 // 8804 -2445 // 8804 -4225 // 8804 2205 // 4402 -567 // 4402]
    γ̂1 = [ 0 // 1    0 // 1        0 // 1        0 // 1       0 // 1       0 // 1   ]
    #! format: on
    MRIGARKDecoupledImplicit(
        slowrhs!,
        implicitsolve!,
        fastsolver,
        (Γ0, Γ1),
        (γ̂0, γ̂1),
        Q,
        dt,
        t0,
    )
end

"""
    MRIGARKESDIRK23LSA(f!, fastsolver, Q; dt, t0 = 0, δ = 0

A 2nd order, 3 stage decoupled implicit scheme. It is based on L-Stable,
stiffly-accurate ESDIRK scheme of Bank et al (1985); see also Kennedy and
Carpenter (2016).

The free parameter `δ` can take any values for accuracy.

### References

    @article{Bank1985,
        title={Transient simulation of silicon devices and circuits},
        author={R. E. Bank and W. M. Coughran and W. Fichtner and
                E. H. Grosse and D. J. Rose and R. K. Smith},
        journal={IEEE Transactions on Computer-Aided Design of Integrated
                 Circuits and Systems},
        volume={4},
        number={4},
        pages={436-451},
        year={1985},
        publisher={IEEE},
        doi={10.1109/TCAD.1985.1270142}
    }

    @techreport{KennedyCarpenter2016,
        title = {Diagonally implicit Runge-Kutta methods for ordinary
                 differential equations. A review},
                 author = {C. A. Kennedy and M. H. Carpenter},
        institution = {National Aeronautics and Space Administration},
        year = {2016},
        number = {NASA/TM–2016–219173},
        address = {Langley Research Center, Hampton, VA}
    }
"""
function MRIGARKESDIRK23LSA(
    slowrhs!,
    implicitsolve!,
    fastsolver,
    Q;
    dt,
    t0 = 0,
    δ = 0,
)
    T = real(eltype(Q))
    rt2 = sqrt(T(2))

    #! format: off
    Γ0 = [
          2 - rt2                      0                       0
          (1 - rt2) / rt2              (rt2 - 1) / rt2         0
          δ                            rt2 - 1 - δ             0
          (3 - 2rt2 * (1 + δ)) / 2rt2  (δ * 2rt2 - 1) / 2rt2   (rt2 - 1) / rt2
         ]

    γ̂0 = [0 0 0]
    #! format: on
    Δc = sum(Γ0, dims = 2)

    # Check that the explicit steps match the Δc values:
    @assert Γ0[1, 1] ≈ Δc[1]
    @assert Γ0[3, 1] + Γ0[3, 2] ≈ Δc[3]

    # Check the implicit stages have no Δc
    @assert isapprox(Γ0[2, 1] + Γ0[2, 2], 0, atol = eps(T))
    @assert isapprox(Γ0[4, 1] + Γ0[4, 2] + Γ0[4, 3], 0, atol = eps(T))

    # Check consistency with the original scheme
    @assert Γ0[1, 1] + Γ0[2, 1] ≈ 1 - 1 / rt2
    @assert Γ0[2, 2] ≈ 1 - 1 / rt2

    @assert Γ0[1, 1] + Γ0[2, 1] + Γ0[3, 1] + Γ0[4, 1] ≈ 1 / (2 * rt2)
    @assert Γ0[2, 2] + Γ0[3, 2] + Γ0[4, 2] ≈ 1 / (2 * rt2)
    @assert Γ0[4, 3] ≈ 1 - 1 / rt2


    MRIGARKDecoupledImplicit(
        slowrhs!,
        implicitsolve!,
        fastsolver,
        (Γ0,),
        (γ̂0,),
        Q,
        dt,
        t0,
    )
end

"""
    MRIGARKESDIRK24LSA(f!,
                       fastsolver,
                       Q;
                       dt,
                       t0 = 0,
                       γ = 0.2,
                       c3 = (2γ + 1) / 2,
                       a32 = 0.2,
                       α = -0.1,
                       β1 = c3 / 10,
                       β2 = c3 / 10,
                       )

A 2nd order, 4 stage decoupled implicit scheme. It is based on an L-Stable,
stiffly-accurate ESDIRK.
"""
function MRIGARKESDIRK24LSA(
    slowrhs!,
    implicitsolve!,
    fastsolver,
    Q;
    dt,
    t0 = 0,
    γ = 0.2,
    c3 = (2γ + 1) / 2,
    a32 = 0.2,
    α = -0.1,
    β1 = c3 / 10,
    β2 = c3 / 10,
)
    T = real(eltype(Q))

    # Check L-Stability constraint; bound comes from Kennedy and Carpenter
    # (2016) Table 5.
    @assert 0.1804253064293985641345831 ≤ γ < 1 // 2

    # check the stage times are increasing
    @assert 2γ < c3 < 1

    # Original RK scheme
    # Enforce L-Stability
    b3 = (2 * (1 - γ)^2 - 1) / 4 / a32

    # Enforce 2nd order accuracy
    b2 = (1 - 2γ - 2b3 * c3) / 4γ

    A = [
        0 0 0 0
        γ γ 0 0
        c3 - a32 - γ a32 γ 0
        1 - b2 - b3 - γ b2 b3 γ
    ]
    c = sum(A, dims = 2)
    b = A[end, :]

    # Check 2nd order accuracy
    @assert sum(b) ≈ 1
    @assert 2 * sum(A' * b) ≈ 1

    # Setup the GARK Tableau
    Δc = [c[2], 0, c[3] - c[2], 0, c[4] - c[3], 0]

    Γ0 = zeros(T, 6, 4)
    Γ0[1, 1] = Δc[1]

    Γ0[2, 1] = A[2, 1] - Γ0[1, 1]
    Γ0[2, 2] = A[2, 2]

    Γ0[3, 1] = α
    Γ0[3, 2] = Δc[3] - Γ0[3, 1]

    Γ0[4, 1] = A[3, 1] - Γ0[1, 1] - Γ0[2, 1] - Γ0[3, 1]
    Γ0[4, 2] = A[3, 2] - Γ0[1, 2] - Γ0[2, 2] - Γ0[3, 2]
    Γ0[4, 3] = A[3, 3]

    Γ0[5, 1] = β1
    Γ0[5, 2] = β2
    Γ0[5, 3] = Δc[5] - Γ0[5, 1] - Γ0[5, 2]

    Γ0[6, 1] = A[4, 1] - Γ0[1, 1] - Γ0[2, 1] - Γ0[3, 1] - Γ0[4, 1] - Γ0[5, 1]
    Γ0[6, 2] = A[4, 2] - Γ0[1, 2] - Γ0[2, 2] - Γ0[3, 2] - Γ0[4, 2] - Γ0[5, 2]
    Γ0[6, 3] = A[4, 3] - Γ0[1, 3] - Γ0[2, 3] - Γ0[3, 3] - Γ0[4, 3] - Γ0[5, 3]
    Γ0[6, 4] = A[4, 4]

    γ̂0 = [0 0 0 0]

    # Check consistency with original scheme
    @assert all(A ≈ [0 0 0 0; accumulate(+, Γ0, dims = 1)[2:2:end, :]])
    @assert all(Δc ≈ sum(Γ0, dims = 2))

    MRIGARKDecoupledImplicit(
        slowrhs!,
        implicitsolve!,
        fastsolver,
        (Γ0,),
        (γ̂0,),
        Q,
        dt,
        t0,
    )
end
