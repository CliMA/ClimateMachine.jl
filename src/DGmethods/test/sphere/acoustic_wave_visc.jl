#@article{TOMITA2004357,
#title = "A new dynamical framework of nonhydrostatic global model using the icosahedral grid",
#journal = "Fluid Dynamics Research",
#volume = "34",
#number = "6",
#pages = "357 - 400",
#year = "2004",
#issn = "0169-5983",
#doi = "https://doi.org/10.1016/j.fluiddyn.2004.03.003",
#url = "http://www.sciencedirect.com/science/article/pii/S0169598304000310",
#author = "Hirofumi Tomita and Masaki Satoh",
#keywords = "Atmospheric general circulation model, Icosahedral grid, Nonhydrostatic equations",
#}
#
# This version runs the acoustic wave on the sphere as a stand alone test (no dependence
# on CLIMA moist thermodynamics). The point of this file is to show users how to use the GLOBAL_MAX, GLOBAL_MIN, and GLOBAL_EXTREMA_DIFF function.
#The GLOBAL_EXTREMA_DIFF function is used to extract the perturbation values from the total state minus the reference state which is stored in AUX.
#--------------------------------#
#--------------------------------#
#Can be run with:
# mpirun -n 1 julia --project=@. acoustic_wave_visc.jl
# mpirun -n 2 julia --project=@. acoustic_wave_visc.jl
#--------------------------------#
#--------------------------------#

using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using Random

#Earth Constants
const gravity = 9.80616
const earth_radius = 6.37122e6
const rgas = 287.17
const cp = 1004.67
const cv = 717.5
const p0 = 1.0e5
const ztop = 10.0e3 #10km
#Earth Constants

#Simulation Constants
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const statenames = ("ρ", "U", "V", "W", "E")
const γ_exact = 7 // 5
const μ_exact = 0 // 1
const radians = true
const _nviscstates = 6
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23 = 1:_nviscstates
const _ngradstates = 3
const _states_for_gradient_transform = (_ρ, _U, _V, _W)
const iperturbation = 0
#Simulation Constants

const integration_testing = false

# preflux computation
@inline function preflux(Q, VF, aux, t)
  γ::eltype(Q) = γ_exact
  @inbounds ρ, U, V, W, E, ϕ = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], aux[_a_ϕ]
  ρinv = 1 / ρ
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  ((γ-1)*(E - 0.5*ρ*(u^2 + v^2 + w^2) - ρ*ϕ), u, v, w, ρinv)
end

# max eigenvalue
@inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv)
  γ::eltype(Q) = γ_exact
  @inbounds abs(n[1] * u + n[2] * v + n[3] * w) + sqrt(ρinv * γ * P)
end

# Euler flux function
euler_flux!(F, Q, VF, aux, t) = euler_flux!(F, Q, VF, aux, t, preflux(Q,VF,aux,t)...)
@inline function euler_flux!(F,Q,VF,aux,t,P,u,v,w,ρinv)
  @inbounds begin
      ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
      τ11, τ22, τ33 = VF[_τ11], VF[_τ22], VF[_τ33]
      τ12 = τ21 = VF[_τ12]
      τ13 = τ31 = VF[_τ13]
      τ23 = τ32 = VF[_τ23]

      # inviscid terms
      F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
      F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
      F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
      F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
      F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)

      # viscous terms
      F[1, _U] -= τ11; F[2, _U] -= τ12; F[3, _U] -= τ13
      F[1, _V] -= τ21; F[2, _V] -= τ22; F[3, _V] -= τ23
      F[1, _W] -= τ31; F[2, _W] -= τ32; F[3, _W] -= τ33

      F[1, _E] -= u * τ11 + v * τ12 + w * τ13
      F[2, _E] -= u * τ21 + v * τ22 + w * τ23
      F[3, _E] -= u * τ31 + v * τ32 + w * τ33
  end
end

# Compute the velocity from the state
@inline function velocities!(vel, Q, _...)
  @inbounds begin
    # ordering should match states_for_gradient_transform
    ρ, U, V, W = Q[1], Q[2], Q[3], Q[4]
    ρinv = 1 / ρ
    vel[1], vel[2], vel[3] = ρinv * U, ρinv * V, ρinv * W
  end
end

# Visous flux
@inline function compute_stresses!(VF, grad_vel, _...)
  μ::eltype(VF) = μ_exact
  @inbounds begin
    dudx, dudy, dudz = grad_vel[1, 1], grad_vel[2, 1], grad_vel[3, 1]
    dvdx, dvdy, dvdz = grad_vel[1, 2], grad_vel[2, 2], grad_vel[3, 2]
    dwdx, dwdy, dwdz = grad_vel[1, 3], grad_vel[2, 3], grad_vel[3, 3]

    # strains
    ϵ11 = dudx
    ϵ22 = dvdy
    ϵ33 = dwdz
    ϵ12 = (dudy + dvdx) / 2
    ϵ13 = (dudz + dwdx) / 2
    ϵ23 = (dvdz + dwdy) / 2

    # deviatoric stresses
    VF[_τ11] = 2μ * (ϵ11 - (ϵ11 + ϵ22 + ϵ33) / 3)
    VF[_τ22] = 2μ * (ϵ22 - (ϵ11 + ϵ22 + ϵ33) / 3)
    VF[_τ33] = 2μ * (ϵ33 - (ϵ11 + ϵ22 + ϵ33) / 3)
    VF[_τ12] = 2μ * ϵ12
    VF[_τ13] = 2μ * ϵ13
    VF[_τ23] = 2μ * ϵ23
  end
end

@inline function stresses_penalty!(VF, nM, velM, QM, aM, velP, QP, aP, t)
  @inbounds begin
    n_Δvel = similar(VF, Size(3, 3))
    for j = 1:3, i = 1:3
      n_Δvel[i, j] = nM[i] * (velP[j] - velM[j]) / 2
    end
    compute_stresses!(VF, n_Δvel)
  end
end

@inline stresses_boundary_penalty!(VF, _...) = VF.=0

#=
# initial condition
function initialcondition!(dim, Q, t, x, y, z, _...)
  DFloat = eltype(Q)
  ρ::DFloat = ρ_g(t, x, y, z, dim)
  U::DFloat = U_g(t, x, y, z, dim)
  V::DFloat = V_g(t, x, y, z, dim)
  W::DFloat = W_g(t, x, y, z, dim)
  E::DFloat = E_g(t, x, y, z, dim)

  if integration_testing
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
  else
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] =
    10+rand(), rand(), rand(), rand(), 10+rand()
  end
end
=#

# Boundary flux function
@inline function euler_bc_flux!(QP, _, _, nM, QM, _, auxM, bctype, t,  PM, uM, vM, wM, ρMinv)
    DFloat = eltype(QM)
    γ:: DFloat = γ_exact
    @inbounds begin
        if bctype == 1 #no-flux
            #Store values at left of boundary ("-" values)
            ρM, UM, VM, WM, EM = QM[_ρ], QM[_U], QM[_V], QM[_W], QM[_E]
            ϕM=auxM[_a_ϕ]

            #Scalars are the same on both sides of the boundary
            ρP=ρM; PP=PM; ϕP=ϕM
            nx, ny, nz = nM[1], nM[2], nM[3]

            #reflect velocities
            uN=nx*uM + ny*vM + nz*wM
            uP=uM - 2*uN*nx
            vP=vM - 2*uN*ny
            wP=wM - 2*uN*nz

            #Construct QP state
            QP[_ρ], QP[_U], QP[_V], QP[_W] = ρP, ρP*uP, ρP*vP, ρP*wP
            QP[_E]= PP/(γ-1) + 0.5*ρP*( uP^2 + vP^2 + wP^2) + ρP*ϕP
        end
    nothing
  end
end

#{{{ Cartesian->Spherical
@inline function cartesian_to_spherical(x,y,z,radians)
    #Conversion Constants
    if radians
        c=1.0
    else
        c=180/π
    end
    λ_max=2π*c
    λ_min=0*c
    ϕ_max=+0.5*π*c
    ϕ_min=-0.5*π*c

    #Conversion functions
    r = hypot(x,y,z)
    λ=atan(y,x)*c
    ϕ=asin(z/r)*c
    return (r, λ, ϕ)
end
#}}} Cartesian->Spherical

# Construct Geopotential
const _nauxstate = 12
const _ρ, _U, _V, _W, _E, _a_ϕ, _a_ϕx, _a_ϕy, _a_ϕz, _a_x, _a_y, _a_z = 1:_nauxstate
const auxnames = ("ρ_ref", "U_ref", "V_ref", "W_ref", "E_ref", "ϕ", "ϕx", "ϕy", "ϕz", "x", "y", "z")
@inline function auxiliary_state_initialization!(aux, x, y, z)
    DFloat = eltype(aux)
    γ:: DFloat = γ_exact

    #Test Case Constants
    a :: DFloat = earth_radius
    R_pert :: DFloat = a/3.0  #!Radius of perturbation
    nv :: DFloat = 1.0
    T0 :: DFloat = 300
    #Test Case Constants

    #Convert to Spherical Coords
    (r, λ, ϕ) = cartesian_to_spherical(x,y,z,radians)
    h = r - a
    cosϕ = cos(ϕ)
    cosλ = cos(λ)
    sinλ = sin(λ)

    #Potential Temperature for an isothermal atmosphere
    θ_ref = T0*exp(gravity*h/(cp*T0))
    #Hydrostatic pressure from the def. of potential temp
    p_ref = p0*(T0/θ_ref)^(cp/rgas)
    #Density from the ideal gas law
    ρ_ref = (p0/(rgas*θ_ref))*(p_ref/p0)^(cv/cp)

    #Fields
    ϕ = gravity*h
    u, v, w = 0, 0, 0
    #Reference State
    U_ref = ρ_ref*u
    V_ref = ρ_ref*v
    W_ref = ρ_ref*w
    E_ref = p_ref/(γ-1) + 0.5*ρ_ref*(u^2 + v^2 + w^2) + ρ_ref*ϕ

    #Store Reference State and other Constant Fields
    @inbounds begin
        aux[_ρ] = ρ_ref
        aux[_U] = U_ref
        aux[_V] = V_ref
        aux[_W] = W_ref
        aux[_E] = E_ref
        aux[_a_ϕ] = ϕ
        aux[_a_x] = x
        aux[_a_y] = y
        aux[_a_z] = z
    end
end

# Construct Euler Source which is ρ*grad(ϕ)
@inline function euler_source!(S, Q, aux, t)
  @inbounds begin
      ρ, ϕx, ϕy, ϕz =  Q[_ρ], aux[_a_ϕx], aux[_a_ϕy], aux[_a_ϕz]
      S[_ρ] = 0
      S[_U] = - ρ*ϕx
      S[_V] = - ρ*ϕy
      S[_W] = - ρ*ϕz
      S[_E] = 0
  end
end

# initial condition
function acoustic_wave!(Q, t, x, y, z, aux, _...)
    DFloat = eltype(Q)
    γ:: DFloat = γ_exact

    #Test Case Constants
    a :: DFloat = earth_radius
    R_pert :: DFloat = a/3.0  #!Radius of perturbation
    nv :: DFloat = 1.0
    T0 :: DFloat = 300
    #Test Case Constants

    #Convert to Spherical Coords
    (r, λ, ϕ) = cartesian_to_spherical(x,y,z,radians)
    h = r - a
    cosϕ = cos(ϕ)
    cosλ = cos(λ)
    sinλ = sin(λ)

    #Potential Temperature for an isothermal atmosphere
    θ_ref = T0*exp(gravity*h/(cp*T0))
    #Hydrostatic pressure from the def. of potential temp
    p_ref = p0*(T0/θ_ref)^(cp/rgas)
    #Density from the ideal gas law
    ρ_ref = (p0/(rgas*θ_ref))*(p_ref/p0)^(cv/cp)

    #Pressure Perturbation
    r1 = a*acos(cosϕ*cosλ)
    if (r1 < R_pert)
        f = 0.5*(1 + cos(π*r1/R_pert))
    else
        f = 0
    end

    #vertical profile
    g = sin(nv*π*h/ztop)

    dp = 100*f*g #Original
    p = p_ref + dp*iperturbation
    ρ = p0/(rgas*θ_ref)*(p/p0)^(cv/cp)

    #Fields
    ϕ = aux[_a_ϕ]
    u, v, w = 0, 0, 0
    #Initial Condition: perturbation + Reference State
    U = ρ*u
    V = ρ*v
    W = ρ*w
    E = p/(γ-1) + 0.5*ρ*(u^2 + v^2 + w^2) + ρ*ϕ

    #Store Initial conditions
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
end

#{{{ Main
function main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt, nsimstep)

    grid = DiscontinuousSpectralElementGrid(topl,
                                            FloatType = DFloat,
                                            DeviceArray = ArrayType,
                                            polynomialorder = N,
                                            meshwarp = Topologies.cubedshellwarp)

    #{{{ numerical fluxex
    numflux!(x...) = NumericalFluxes.rusanov!(x...,euler_flux!,wavespeed,preflux)
    numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., euler_flux!,
                                                              euler_bc_flux!,
                                                              wavespeed,preflux)

    #Define Spatial Discretization
    spacedisc = DGBalanceLaw(grid = grid,
                             length_state_vector = _nstate,
                             flux! = euler_flux!,
                             numerical_flux! = numflux!,
                             numerical_boundary_flux! = numbcflux!,
                             number_gradient_states = _ngradstates,
                             states_for_gradient_transform =
                             _states_for_gradient_transform,
                             number_viscous_states = _nviscstates,
                             gradient_transform! = velocities!,
                             viscous_transform! = compute_stresses!,
                             viscous_penalty! = stresses_penalty!,
                             viscous_boundary_penalty! = stresses_boundary_penalty!,
                             auxiliary_state_length = _nauxstate,
                             auxiliary_state_initialization! =
                             auxiliary_state_initialization!,
                             source! = euler_source!)

    #Compute Gradient of Geopotential
    DGBalanceLawDiscretizations.grad_auxiliary_state!(spacedisc, _a_ϕ, (_a_ϕx, _a_ϕy, _a_ϕz))

    #Initial Condition
    initialcondition(Q, x...) = acoustic_wave!(Q, DFloat(0), x...)
    Q = MPIStateArray(spacedisc, initialcondition)

    #Store Initial Condition as Exact Solution
    Qe = copy(Q)

    #Define Time-Integration Method
    lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

    #------------Set Callback Info--------------------------------#
#    nsimstep=1000
    nwalltime=100
    nsimstepvtk=1000
    # Set up General information callback
    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXSimulationSteps(nsimstep) do
        (δρ_max,δρ_min)=global_extrema_diff(Q,spacedisc.auxstate,_ρ)
        (δU_max,δU_min)=global_extrema_diff(Q,spacedisc.auxstate,_U)
        (δV_max,δV_min)=global_extrema_diff(Q,spacedisc.auxstate,_V)
        (δW_max,δW_min)=global_extrema_diff(Q,spacedisc.auxstate,_W)
        (δE_max,δE_min)=global_extrema_diff(Q,spacedisc.auxstate,_E)
        @info @sprintf """Update
        simtime = %.16e
        runtime = %s
        Δmass   = %.16e
        Δenergy = %.16e
        δρ_max = %.16e
        δρ_min = %.16e
        δU_max = %.16e
        δU_min = %.16e
        δV_max = %.16e
        δV_min = %.16e
        δW_max = %.16e
        δW_min = %.16e
        δE_max = %.16e
        δE_min = %.16e""" ODESolvers.gettime(lsrk) Dates.format(convert(Dates.DateTime, Dates.now()-starttime[]), Dates.dateformat"HH:MM:SS") abs(weightedsum(Q,_ρ) - weightedsum(Qe,_ρ)) / weightedsum(Qe,_ρ) abs(weightedsum(Q,_E) - weightedsum(Qe,_E)) / weightedsum(Qe,_E) δρ_max δρ_min δU_max δU_min δV_max δV_min δW_max δW_min δE_max δE_min
        nothing
    end

    # Set up Mass/Energy conservation callback
    cbmass = GenericCallbacks.EveryXSimulationSteps(nsimstep) do
        @info @sprintf """Conservation Metrics
            Δmass   = %.16e
            Δenergy = %.16e""" abs(weightedsum(Q,_ρ) - weightedsum(Qe,_ρ)) / weightedsum(Qe,_ρ) abs(weightedsum(Q,_E) - weightedsum(Qe,_E)) / weightedsum(Qe,_E)
    end

    # Set up Extrema difference callback
    cbextrema = GenericCallbacks.EveryXSimulationSteps(nsimstep) do
        (δρ_max,δρ_min)=global_extrema_diff(Q,spacedisc.auxstate,_ρ)
        (δU_max,δU_min)=global_extrema_diff(Q,spacedisc.auxstate,_U)
        (δV_max,δV_min)=global_extrema_diff(Q,spacedisc.auxstate,_V)
        (δW_max,δW_min)=global_extrema_diff(Q,spacedisc.auxstate,_W)
        (δE_max,δE_min)=global_extrema_diff(Q,spacedisc.auxstate,_E)
        @info @sprintf """Extrema Values
            δρ_max = %.16e
            δρ_min = %.16e
            δE_max = %.16e
            δE_min = %.16e""" δρ_max δρ_min δE_max δE_min
    end

    # Set up VTK information callback
    step = [0]
    mkpath("vtk")
    cbvtk = GenericCallbacks.EveryXSimulationSteps(nsimstepvtk) do (init=false)
        outprefix = @sprintf("vtk/acoustic_wave_visc_%dD_mpirank%04d_step%04d",
                             3, MPI.Comm_rank(mpicomm), step[1])
        @debug "doing VTK output" outprefix
        DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, statenames, spacedisc.auxstate, auxnames)
        step[1] += 1
        nothing
    end
    #------------Set Callback Info--------------------------------#

    #Perform Time-Integration
    solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

    # Print some end of the simulation information
    if integration_testing
        error = euclidean_distance(Q, Qe) / norm(Qe)
        Δmass = abs(weightedsum(Q,_ρ) - weightedsum(Qe,_ρ)) / weightedsum(Qe,_ρ)
        Δenergy = abs(weightedsum(Q,_E) - weightedsum(Qe,_E)) / weightedsum(Qe,_E)
        @info @sprintf """Finished
            error = %.16e
            Δmass = %.16e
            Δenergy = %.16e
            """ error Δmass Δenergy
    else
        error = euclidean_distance(Q, Qe) / norm(Qe)
        Δmass = abs(weightedsum(Q,_ρ) - weightedsum(Qe,_ρ)) / weightedsum(Qe,_ρ)
        Δenergy = abs(weightedsum(Q,_E) - weightedsum(Qe,_E)) / weightedsum(Qe,_E)
        @info @sprintf """Finished
            error = %.16e
            Δmass = %.16e
            Δenergy = %.16e
            """ error Δmass Δenergy
    end

    #return diagnostics
    return (error, Δmass, Δenergy)
end
#}}} Main

#{{{ Run Script
function run(mpicomm, Nhorizontal, Nvertical, N, timeend, DFloat, dt, nsimstep, ArrayType)
    height_min=earth_radius
    height_max=earth_radius + ztop
    Rrange=range(DFloat(height_min); length=Nhorizontal+1, stop=height_max)
    topl = StackedCubedSphereTopology(mpicomm,Nhorizontal,Rrange; boundary=(1,1))
    (error, Δmass, Δenergy) = main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt, nsimstep)
end
#}}} Run Script

#{{{ Run Program
using Test
let
    MPI.Initialized() || MPI.Init()
    Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
    mpicomm=MPI.COMM_WORLD

    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))
    @static if Base.find_package("CUDAnative") !== nothing
      device!(MPI.Comm_rank(mpicomm) % length(devices()))
    end

    #This snippet of code allows one to run just one instance/configuration. Before running this, Comment the Integration Testing block above
    DFloat = Float64
    N=4
    ArrayType = Array
    dt=1 #stable dt for N=4 and Nhorizontal=5
    timeend=1000
    nsimstep=100
    Nhorizontal = 4 #number of horizontal elements per face of cubed-sphere grid
    Nvertical = 4 #number of horizontal elements per face of cubed-sphere grid
    #    dt=dt/Nhorizontal
    nsteps = ceil(Int64, timeend / dt)
    dt = timeend / nsteps
    (error, Δmass,  Δenergy) = run(mpicomm, Nhorizontal, Nvertical, N, timeend, DFloat, dt, nsimstep, ArrayType)

end #Test

isinteractive() || MPI.Finalize()


