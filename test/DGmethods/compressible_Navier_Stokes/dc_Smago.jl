# Load Modules 
using MPI
using CLIMA
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
using CLIMA.Vtk

#using CLIMA.ReadConfigurationFile
using CLIMA.Topography

if haspkg("CuArrays")
    using CUDAdrv
    using CUDAnative
    using CuArrays
    CuArrays.allowscalar(false)
    const ArrayType = CuArray
else
    const ArrayType = Array
end

# Prognostic equations: ρ, (ρu), (ρv), (ρw), (ρe_tot), (ρq_tot)
# For the dry example shown here, we load the moist thermodynamics module 
# and consider the dry equation set to be the same as the moist equations but
# with total specific humidity = 0. 
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0

# State labels 
const _nstate = 6
const _ρ, _U, _V, _W, _E, _QT = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E, QTid = _QT)
const statenames = ("RHO", "U", "V", "W", "E", "QT")

# Viscous state labels
const _nviscstates = 16
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23, _qx, _qy, _qz, _Tx, _Ty, _Tz, _θx, _θy, _θz, _SijSij = 1:_nviscstates

# Gradient state labels
const _ngradstates = 6
const _states_for_gradient_transform = (_ρ, _U, _V, _W, _E, _QT)

if !@isdefined integration_testing
    const integration_testing =
        parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
    using Random
end

# Problem constants (TODO: parameters module (?))
const μ_sgs           = 100.0
const Prandtl         = 71 // 100
const Prandtl_t       = 1 // 3
const cp_over_prandtl = cp_d / Prandtl_t

# Problem description 
# --------------------
# 2D thermal perturbation (cold bubble) in a neutrally stratified atmosphere
# No wall-shear, lateral periodic boundaries with no-flux walls at the domain
# top and bottom. 
# Inviscid, Constant viscosity, StandardSmagorinsky, MinimumDissipation
# filters are tested against this benchmark problem
# TODO: link to module SubGridScaleTurbulence

#
# Read user configuration file:
#
# dict_user_input = read_configuration_file()

##Print dictionary keys and values:
#for key in dict_user_input
#    @info @sprintf """ Dictionaryxx: %s %s %s""" (key[1]) (key[2]) typeof((key[2]))
#end

#const numdims = 3
const numdims = get(dict_user_input, "nsd", "nsd NOT DEFINED")
const Npoly   = get(dict_user_input, "Npoly", "Npoly NOT DEFINED")

Δx      = get(dict_user_input, "Δx", "Δx NOT DEFINED")
Δy      = get(dict_user_input, "Δy", "Δy NOT DEFINED")
Δz      = get(dict_user_input, "Δz", "Δz NOT DEFINED")
xmin    = get(dict_user_input, "xmin", "xmin NOT DEFINED")
xmax    = get(dict_user_input, "xmax", "xmax NOT DEFINED")
ymin    = get(dict_user_input, "ymin", "ymax NOT DEFINED")
ymax    = get(dict_user_input, "ymax", "ymax NOT DEFINED")
zmin    = get(dict_user_input, "zmin", "zmin NOT DEFINED")
zmax    = get(dict_user_input, "zmax", "zmax NOT DEFINED")
dt      = get(dict_user_input, "dt", "dt NOT DEFINED")
timeend = get(dict_user_input, "tfinal", "tfinal NOT DEFINED")


#@info @sprintf """ NumDims from dict %d""" numdims
#@info @sprintf """ dx dy dz          %.16e %.16e %.16e""" Δx Δy Δz
#@info @sprintf """ Npoly             %d""" Npoly
#@info @sprintf """ (xmin xmax)       %.16e %.16e""" xmin xmax
#@info @sprintf """ (ymin ymax)       %.16e %.16e""" ymin ymax
#@info @sprintf """ (zmin zmax)       %.16e %.16e""" zmin zmax
#@info @sprintf """ dt                %.16e""" dt
#@info @sprintf """ timeend           %.16e""" timeend

#Get Nex, Ney from resolution
Lx = xmax - xmin
Ly = ymax - ymin
Lz = zmax - ymin

if ( Δx > 0)
    #
    # User defines the grid size:
    #
    ratiox = (Lx/Δx - 1)/Npoly
    ratioy = (Ly/Δy - 1)/Npoly
    ratioz = (Lz/Δz - 1)/Npoly
    Nex = ceil(Int64, ratiox)
    Ney = ceil(Int64, ratioy)
    Nez = ceil(Int64, ratioz)
    
else
    #
    # User defines the number of elements:
    #
    Δx = Lx / ((Nex * Npoly) + 1)
    Δy = Ly / ((Ney * Npoly) + 1)
    Δz = Lz / ((Nez * Npoly) + 1)
end

# Smagorinsky model requirements : TODO move to SubgridScaleTurbulence module 
const C_smag = 0.23
# Equivalent grid-scale
#Δ = sqrt(Δx * Δy)
#const Δsqr = Δ * Δ

# Anisotropic grid computation
function anisotropic_coefficient_sgs(Δx, Δy, Δz)

    Δ = (Δx * Δy *  Δz)^(1/3)
    
    Δ_sorted = sort([Δx, Δy, Δz])  
    Δ_s1 = Δ_sorted[1]
    Δ_s2 = Δ_sorted[2]
    a1 = Δ_s1 / max(Δx,Δy,Δz) / (Npoly + 1)
    a2 = Δ_s2 / max(Δx,Δy,Δz) / (Npoly + 1)
    f_anisotropic = 1 + 2/27 * ((log(a1))^2 - log(a1)*log(a2) + (log(a2))^2 )
    
    Δ = Δ*f_anisotropic
    Δsqr = Δ * Δ
    
    return Δsqr
end

const Δsqr = anisotropic_coefficient_sgs(Δx, Δy, Δz)

# -------------------------------------------------------------------------
# Preflux calculation: This function computes parameters required for the 
# DG RHS (but not explicitly solved for as a prognostic variable)
# In the case of the rising_thermal_bubble example: the saturation
# adjusted temperature and pressure are such examples. Since we define
# the equation and its arguments here the user is afforded a lot of freedom
# around its behaviour. 
# The preflux function interacts with the following  
# Modules: NumericalFluxes.jl 
# functions: wavespeed, cns_flux!, bcstate!
# -------------------------------------------------------------------------
@inline function preflux(Q,VF, aux, _...)
    gravity::eltype(Q) = grav
    R_gas::eltype(Q) = R_d
    @inbounds ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    ρinv = 1 / ρ
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    u, v, w = ρinv * U, ρinv * V, ρinv * W
    e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * y) / ρ
    q_tot = QT / ρ
    # Establish the current thermodynamic state using the prognostic variables
    TS = PhaseEquil(e_int, q_tot, ρ)
    T = air_temperature(TS)
    P = air_pressure(TS) # Test with dry atmosphere
    q_liq = PhasePartition(TS).liq
    θ = virtual_pottemp(TS)
    (P, u, v, w, ρinv, q_liq,T,θ)
end

#-------------------------------------------------------------------------
#md # Soundspeed computed using the thermodynamic state TS
# max eigenvalue
@inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv, q_liq, T, θ)
    gravity::eltype(Q) = grav
    @inbounds begin 
        ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
        x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
        u, v, w = ρinv * U, ρinv * V, ρinv * W
        e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * y) / ρ
        q_tot = QT / ρ
        TS = PhaseEquil(e_int, q_tot, ρ)
        (n[1] * u + n[2] * v + n[3] * w) + soundspeed_air(TS)
    end
end

# -------------------------------------------------------------------------
# ### Physical Flux (Required)
#md # Here, we define the physical flux function, i.e. the conservative form
#md # of the equations of motion for the prognostic variables ρ, U, V, W, E, QT
#md # $\frac{\partial Q}{\partial t} + \nabla \cdot \boldsymbol{F} = \boldsymbol {S}$
#md # $\boldsymbol{F}$ contains both the viscous and inviscid flux components
#md # and $\boldsymbol{S}$ contains source terms.
#md # Note that the preflux calculation is splatted at the end of the function call
#md # to cns_flux!
# -------------------------------------------------------------------------
cns_flux!(F, Q, VF, aux, t) = cns_flux!(F, Q, VF, aux, t, preflux(Q,VF, aux)...)
@inline function cns_flux!(F, Q, VF, aux, t, P, u, v, w, ρinv, q_liq, T, θ)
    gravity::eltype(Q) = grav
    @inbounds begin
        ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
        # Inviscid contributions
        F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
        F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
        F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
        F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
        F[1, _E], F[2,       = cv_d
    p0::DFloat            = MSLP
    gravity::DFloat       = grav
    # initialise with dry domain 
    q_tot::DFloat         = 0
    q_liq::DFloat         = 0
    q_ice::DFloat         = 0 
    # perturbation parameters for rising bubble
    rx                    = 4000
    ry                    = 2000
    xc                    = 0
    yc                    = 3000
    r                     = sqrt( (x - xc)^2/rx^2 + (y - yc)^2/ry^2)
    θ_ref::DFloat         = 300
    θ_c::DFloat           = -15.0
    Δθ::DFloat            = 0.0
    if r <= 1
        Δθ = θ_c * (1 + cospi(r))/2
    end
    qvar                  = PhasePartition(q_tot)
    θ                     = θ_ref + Δθ # potential temperature
    π_exner               = 1.0 - gravity / (c_p * θ) * y # exner pressure
    ρ                     = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

    P                     = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
    T                     = P / (ρ * R_gas) # temperature
    U, V, W               = 0.0 , 0.0 , 0.0  # momentum components
    # energy definitions
    e_kin                 = (U^2 + V^2 + W^2) / (2*ρ)/ ρ
    e_pot                 = gravity * y
    e_int                 = internal_energy(T, qvar)
    E                     = ρ * (e_int + e_kin + e_pot)  #* total_energy(e_kin, e_pot, T, q_tot, q_liq, q_ice)
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]= ρ, U, V, W, E, ρ * q_tot
end

function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)

    brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                  range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)),
                  range(DFloat(zmin), length=Ne[3]+1, DFloat(zmax)))
    
    
    # User defined periodicity in the topl assignment
    # brickrange defines the domain extents
    topl = StackedBrickTopology(mpicomm, brickrange, periodicity=(false,false,false))

    grid = DiscontinuousSpectralElementGrid(topl,
                                            FloatType = DFloat,
                                            DeviceArray = ArrayType,
                                            polynomialorder = N)
    
    numflux!(x...) = NumericalFluxes.rusanov!(x..., cns_flux!, wavespeed, preflux)
    numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., cns_flux!, bcstate!, wavespeed, preflux)

    # spacedisc = data needed for evaluating the right-hand side function
    spacedisc = DGBalanceLaw(grid = grid,
                             length_state_vector = _nstate,
                             flux! = cns_flux!,
                             numerical_flux! = numflux!,
                             numerical_boundary_flux! = numbcflux!, 
                             number_gradient_states = _ngradstates,
                             states_for_gradient_transform =
                             _states_for_gradient_transform,
                             number_viscous_states = _nviscstates,
                             gradient_transform! = gradient_vars!,
                             viscous_transform! = compute_stresses!,
                             viscous_penalty! = stresses_penalty!,
                             viscous_boundary_penalty! = stresses_boundary_penalty!,
                             auxiliary_state_length = _nauxstate,
                             auxiliary_state_initialization! =
                             auxiliary_state_initialization!,
                             source! = source!)

    # This is a actual state/function that lives on the grid
    initialcondition(Q, x...) = density_current!(Val(dim), Q, DFloat(0), x...)
    Q = MPIStateArray(spacedisc, initialcondition)

    lsrk = LSRK54CarpenterKennedy(spacedisc, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
          norm(Q₀) = %.16e""" eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            #globmean = global_mean(Q, _ρ)
            @info @sprintf("""Update
                             simtime = %.16e
                             runtime = %s
                             norm(Q) = %.16e""", 
                           ODESolvers.gettime(lsrk),
                           Dates.format(convert(Dates.DateTime,
                                                Dates.now()-starttime[]),
                                        Dates.dateformat"HH:MM:SS"),
                           energy )#, globmean)
        end
    end

    npoststates = 9
    _post_sgs, _P, _u, _v, _w, _ρinv, _q_liq, _T, _θ = 1:npoststates
    postnames = ("SGS", "P", "u", "v", "w", "rhoinv", "_q_liq", "T", "THETA")
    postprocessarray = MPIStateArray(spacedisc; nstate=npoststates)

    step = [0]
    mkpath("vtk-DC-smago")
    cbvtk = GenericCallbacks.EveryXSimulationSteps(2500) do (init=false)
        DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, spacedisc,
                                                   Q) do R, Q, QV, aux
                                                       @inbounds let
                                                           (R[_P], R[_u], R[_v], R[_w], R[_ρinv], R[_q_liq], R[_T], R[_θ]) = (preflux(Q, QV, aux))
                                                       end
                                                   end

        outprefix = @sprintf("vtk-DC-smago/cns_%dD_mpirank%04d_step%04d", dim,
                             MPI.Comm_rank(mpicomm), step[1])
        @debug "doing VTK output" outprefix
        writevtk(outprefix, Q, spacedisc, statenames,
                 postprocessarray, postnames)
        #= 
        pvtuprefix = @sprintf("vtk/cns_%dD_step%04d", dim, step[1])
        prefixes = ntuple(i->
        @sprintf("vtk/cns_%dD_mpirank%04d_step%04d",
        dim, i-1, step[1]),
        MPI.Comm_size(mpicomm))
        writepvtu(pvtuprefix, prefixes, postnames)
        =# 
        step[1] += 1
        nothing
    end
    
    solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))


    # Print some end of the simulation information
    engf = norm(Q)
    if integration_testing
        Qe = MPIStateArray(spacedisc,
                           (Q, x...) -> initialcondition!(Val(dim), Q,
                                                          DFloat(timeend), x...))
        engfe = norm(Qe)
        errf = euclidean_distance(Q, Qe)
        @info @sprintf """Finished
            norm(Q)                 = %.16e
            norm(Q) / norm(Q₀)      = %.16e
            norm(Q) - norm(Q₀)      = %.16e
            norm(Q - Qe)            = %.16e
            norm(Q - Qe) / norm(Qe) = %.16e
            """ engf engf/eng0 engf-eng0 errf errf / engfe
    else
        @info @sprintf """Finished
            norm(Q)            = %.16e
            norm(Q) / norm(Q₀) = %.16e
            norm(Q) - norm(Q₀) = %.16e""" engf engf/eng0 engf-eng0
    end
integration_testing ? errf : (engf / eng0)
end

using Test
let
    MPI.Initialized() || MPI.Init()
    Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
    mpicomm = MPI.COMM_WORLD
    if MPI.Comm_rank(mpicomm) == 0
        ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
        loglevel = ll == "DEBUG" ? Logging.Debug :
            ll == "WARN"  ? Logging.Warn  :
            ll == "ERROR" ? Logging.Error : Logging.Info
        global_logger(ConsoleLogger(stderr, loglevel))
    else
        global_logger(NullLogger())
    end
    # User defined number of elements
    # User defined timestep estimate
    # User defined simulation end time
    # User defined polynomial order 
    numelem = (Nex,Ney,Nez)
    #dt = 0.0125
    #timeend = 900
    polynomialorder = Npoly
    DFloat = Float64
    dim = numdims

    #header_file_in = joinpath(@__DIR__, "../../TopographyFiles/NOAA-text-files/monterey.hdr"))
    #body_file_in   = joinpath(@__DIR__, "../../TopographyFiles/NOAA-text-files/monterey.xyz"))
    #TopographyReadExternal("NOAA", header_file_in, body_file_in)
    
    
    engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                    DFloat, dt)
end

isinteractive() || MPI.Finalize()

nothing
