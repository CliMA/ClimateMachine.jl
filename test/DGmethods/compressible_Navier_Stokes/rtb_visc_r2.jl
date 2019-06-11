# Load modules that are used in the CliMA project.
# These are general modules not necessarily specific
# to CliMA
using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

# Load modules specific to CliMA project
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Vtk

# Prognostic equations: ρ, (ρu), (ρv), (ρw), (ρe_tot), (ρq_tot)
# Even for the dry example shown here, we load the moist thermodynamics module 
# and consider the dry equation set to be the same as the moist equations but
# with total specific humidity = 0. 
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0

# For a three dimensional problem 
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const statenames = ("ρ", "U", "V", "W", "E")

const _nviscstates = 9
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23, _Tx, _Ty, _Tz = 1:_nviscstates

const _ngradstates = 5
const _states_for_gradient_transform = (_ρ, _U, _V, _W, _E)

if !@isdefined integration_testing
    const integration_testing =
        parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end


const Prandtl = 71 // 100
const Prandtl_t = 1 // 3
const k_μ = cp_d / Prandtl_t
const μ_exact = 2.5
const γ_exact = 7 // 5

const xmin = 0
const ymin = 0
const zmin = 0
const xmax = 1000
const ymax = 1500
const zmax =  150


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
    γ::eltype(Q) = γ_exact
    gravity::eltype(Q) = grav
    R_gas::eltype(Q) = R_d
    @inbounds ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
    ρinv = 1 / ρ
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    u, v, w = ρinv * U, ρinv * V, ρinv * W
    e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * y) / ρ
    qt = 0.0
    # Establish the current thermodynamic state using the prognostic variables
    TS    = PhaseEquil(e_int, qt, ρ)
    T     = air_temperature(TS)    
    theta = dry_pottemp(TS)
    P     = air_pressure(TS) # Test with dry atmosphere
    
    (P, u, v, w, ρinv, theta)
    # Preflux returns pressure, 3 velocity components, and 1/ρ
end

# -------------------------------------------------------------------------
#md # Soundspeed computed using the thermodynamic state TS
# max eigenvalue
@inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv, _)
  gravity::eltype(Q) = grav
  γ::eltype(Q) = γ_exact
  @inbounds begin 
    ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    u, v, w = ρinv * U, ρinv * V, ρinv * W
    e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * y) / ρ
    n[1] * u + n[2] * v + n[3] * w + sqrt(γ * P / ρ)
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
@inline function cns_flux!(F, Q, VF, aux, t, P, u, v, w, ρinv, _)
    @inbounds begin
        ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
        # Inviscid contributions 
        F[1, _ρ], F[2, _ρ], F[3, _ρ]    = U          , V          , W
        F[1, _U], F[2, _U], F[3, _U]    = u * U  + P , v * U      , w * U
        F[1, _V], F[2, _V], F[3, _V]    = u * V      , v * V + P  , w * V
        F[1, _W], F[2, _W], F[3, _W]    = u * W      , v * W      , w * W + P
        F[1, _E], F[2, _E], F[3, _E]    = u * (E + P), v * (E + P), w * (E + P)

        # Stress tensor
        τ11, τ22, τ33 = VF[_τ11], VF[_τ22], VF[_τ33]
        τ12 = τ21 = VF[_τ12]
        τ13 = τ31 = VF[_τ13]
        τ23 = τ32 = VF[_τ23]        
        vTx, vTy, vTz = VF[_Tx], VF[_Ty], VF[_Tz]
        
        # Viscous contributions
        F[1, _U] -= τ11; F[2, _U] -= τ12; F[3, _U] -= τ13
        F[1, _V] -= τ21; F[2, _V] -= τ22; F[3, _V] -= τ23
        F[1, _W] -= τ31; F[2, _W] -= τ32; F[3, _W] -= τ33
        # Energy dissipation
        F[1, _E] -= u * τ11 + v * τ12 + w * τ13 + vTx
        F[2, _E] -= u * τ21 + v * τ22 + w * τ23 + vTy
        F[3, _E] -= u * τ31 + v * τ32 + w * τ33 + vTz
    end
end


# -------------------------------------------------------------------------
#md # Here we define a function to extract the velocity components from the 
#md # prognostic equations (i.e. the momentum and density variables). This 
#md # function is not required in general, but provides useful functionality 
#md # in some cases. 
# -------------------------------------------------------------------------
# Compute the velocity from the state
gradient_vars!(vel, Q, aux, t, _...) = gradient_vars!(vel, Q, aux, t, preflux(Q,~,aux)...)
@inline function gradient_vars!(vel, Q, aux, t, P, u, v, w, ρinv, _)
  R_gas::eltype(Q) = R_d 
  @inbounds begin
    y = aux[_a_y]
    # ordering should match states_for_gradient_transform
    ρ, U, V, W, E= Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
    ρinv = 1 / ρ
    T = P / ρ / R_gas
    vel[1], vel[2], vel[3] = u, v, w
    vel[4], vel[5] = E, T
  end
end


# -------------------------------------------------------------------------
#md ### Auxiliary Function (Not required)
#md # In this example the auxiliary function is used to store the spatial
#md # coordinates. This may also be used to store variables for which gradients
#md # are needed, but are not available through teh prognostic variable 
#md # calculations. (An example of this will follow - in the Smagorinsky model, 
#md # where a local Richardson number via potential temperature gradient is required)
# -------------------------------------------------------------------------
const _nauxstate = 3
const _a_x, _a_y, _a_z, = 1:_nauxstate
@inline function auxiliary_state_initialization!(aux, x, y, z)
    @inbounds begin
        aux[_a_x] = x
        aux[_a_y] = y
        aux[_a_z] = z
    end
end

# -------------------------------------------------------------------------
#md ### Viscous fluxes. 
#md # The viscous flux function compute_stresses computes the components of 
#md # the velocity gradient tensor, and the corresponding strain rates to
#md # populate the viscous flux array VF. SijSij is calculated in addition
#md # to facilitate implementation of the constant coefficient Smagorinsky model
#md # (pending)
@inline function compute_stresses!(VF, grad_vel, _...)
  gravity::eltype(VF) = grav
  @inbounds begin
    dudx, dudy, dudz = grad_vel[1, 1], grad_vel[2, 1], grad_vel[3, 1]
    dvdx, dvdy, dvdz = grad_vel[1, 2], grad_vel[2, 2], grad_vel[3, 2]
    dwdx, dwdy, dwdz = grad_vel[1, 3], grad_vel[2, 3], grad_vel[3, 3]
    # compute gradients of moist vars and temperature
    dTdx, dTdy, dTdz = grad_vel[1, 5], grad_vel[2, 5], grad_vel[3, 5]
    # virtual potential temperature gradient: for richardson calculation
    # strains
    # --------------------------------------------
    # SMAGORINSKY COEFFICIENT COMPONENTS
    # --------------------------------------------
    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = (dudy + dvdx) / 2
    S13 = (dudz + dwdx) / 2
    S23 = (dvdz + dwdy) / 2
    # --------------------------------------------
    # SMAGORINSKY COEFFICIENT COMPONENTS
    # --------------------------------------------
    # FIXME: Grab functions from module SubgridScaleTurbulence 
    SijSij = (S11^2 + S22^2 + S33^2
                          + 2.0 * S12^2
                          + 2.0 * S13^2 
                          + 2.0 * S23^2) 

    modSij = sqrt(2.0 * SijSij) 
    #ν_e = modSij * C_smag^2 * Δsqr
    ν_e = μ_exact
    D_e= ν_e / Prandtl_t
    #--------------------------------------------
    # deviatoric stresses
    # Fix up index magic numbers
    VF[_τ11] = 2 * ν_e * (S11 - (S11 + S22 + S33) / 3)
    VF[_τ22] = 2 * ν_e * (S22 - (S11 + S22 + S33) / 3)
    VF[_τ33] = 2 * ν_e * (S33 - (S11 + S22 + S33) / 3)
    VF[_τ12] = 2 * ν_e * S12
    VF[_τ13] = 2 * ν_e * S13
    VF[_τ23] = 2 * ν_e * S23
    k_e = k_μ * ν_e 
    # TODO: Viscous stresse come from SubgridScaleTurbulence module
    VF[_Tx], VF[_Ty], VF[_Tz] = k_e * dTdx, k_e * dTdy, k_e * dTdz
  end
end


# -------------------------------------------------------------------------
# generic bc for 2d , 3d
@inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t, PM, uM, vM, wM, ρinvM, _)
    @inbounds begin
        x, y, z = auxM[_a_x], auxM[_a_y], auxM[_a_z]
        ρM, UM, VM, WM, EM = QM[_ρ], QM[_U], QM[_V], QM[_W], QM[_E]
        UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
        QP[_U] = UM - 2 * nM[1] * UnM
        QP[_V] = VM - 2 * nM[2] * UnM
        QP[_W] = WM - 2 * nM[3] * UnM
        QP[_ρ] = ρM
        QP[_E] = EM

        VFP .= VFM
        # To calculate PP, uP, vP, wP, ρinvP we use the preflux function 
        nothing
        #preflux(QP, auxP, t)
        # Required return from this function is either nothing or preflux with plus state as arguments
    end
end
# -------------------------------------------------------------------------

@inline stresses_boundary_penalty!(VF, _...) = VF.=0

@inline function stresses_penalty!(VF, nM, velM, QM, aM, velP, QP, aP, t)
    @inbounds begin
        n_Δvel = similar(VF, Size(3, 3))
        for j = 1:3, i = 1:3
            n_Δvel[i, j] = nM[i] * (velP[j] - velM[j]) / 2
        end
        compute_stresses!(VF, n_Δvel)
    end
end
# -------------------------------------------------------------------------

# --------------DEFINE SOURCES HERE -------------------------------#
#  TODO: Make sure that the source values are not being over-written
# ------------------------------------------------------------------
"""
    The function source! collects all the individual source terms 
    associated with a given problem. We do not define sources here, 
    rather we only call those source terms which are necessary based
    on the governing equations. 
    by terms defined elsewhere
    """
@inline function source!(S,Q,aux,t)

    # Initialise the final block source term 
    S .= 0

    # Typically these sources are imported from modules
    @inbounds begin
        #source_squircle_sponge!(S,Q,aux,t)
        source_geopot!(S, Q, aux, t)
        #source_radiation!(S,Q,aux,t)
        #source_ls_subsidence!(S,Q,aux,t)
    end
end

"""
    Rayleigh sponge function: Linear damping / relaxation to specified
    reference values. In the current implementation we relax velocities
    at the boundaries to a still atmosphere.
    """
@inline function source_squircle_sponge!(S,Q,aux,t)
    gravity::eltype(Q) = grav
    α = 1.0
    U, V, W = Q[_U], Q[_V], Q[_W]
    x, y, z = aux[_a_x], aux[_a_y], aux[_a_z]
    rp = (x^4 + y^4 + z^4)^(1/4) 
    rsponge = 0.85 * xmax # Sponge damper extent  
    @inbounds begin
        if rp > rsponge
            S[_U] += -α * sinpi(1/2 * (rp-rsponge)/rsponge) ^ 4 * U
            S[_V] += -α * sinpi(1/2 * (rp-rsponge)/rsponge) ^ 4 * V
            S[_W] += -α * sinpi(1/2 * (rp-rsponge)/rsponge) ^ 4 * W
        elseif rp > 2 * rsponge
            S[_U] -= U
            S[_V] -= V
            S[_W] -= W
        end
    end
end

"""
    Geopotential source term. Gravity forcing applied to the vertical
    momentum equation
    """
@inline function source_geopot!(S,Q,aux,t)
    gravity::eltype(Q) = grav
    @inbounds begin
        ρ, U, V, W, E  = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
        S[_V] += - ρ * gravity
    end
end

"""
    Large scale subsidence common to several atmospheric observational
    campaigns. In the absence of a GCM to drive the flow we may need to 
    specify a large scale forcing function.
    """
@inline function source_ls_subsidence!(S,Q,aux,t)
    @inbounds begin
        nothing
    end
end

# ------------------------------------------------------------------
# -------------END DEF SOURCES-------------------------------------# 

# initial condition
"""
    User-specified. Required. 
    This function specifies the initial conditions for the Rising Thermal
    Bubble driver. 
    """
function rising_thermal_bubble!(dim, Q, t, x, y, z, _...)
    DFloat                = eltype(Q)
    γ::DFloat             = γ_exact
    # can override default gas constants 
    # to moist values later in the driver 
    R_gas::DFloat         = R_d
    c_p::DFloat           = cp_d
    c_v::DFloat           = cv_d
    p0::DFloat            = MSLP
    gravity::DFloat       = grav
    # initialise with dry domain 
    q_tot::DFloat         = 0 
    q_liq::DFloat         = 0
    q_ice::DFloat         = 0

    xc::DFloat            = 500
    yc::DFloat            = 300
    
    # perturbation parameters for rising bubble
    r                     = sqrt((x - xc)^2 + (y - yc)^2)
    rc::DFloat            = 250
    θ_ref::DFloat         = 300
    θ_c::DFloat           = 0.5
    Δθ::DFloat            = 0.0
    if r <= rc 
        Δθ = θ_c * (1 + cospi(r/rc))/2
    end
    θ                     = θ_ref + Δθ # potential temperature
    π_exner               = 1.0 - gravity / (c_p * θ) * y # exner pressure
    ρ                     = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

    P                     = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
    T                     = P / (ρ * R_gas) # temperature
    U, V, W               = 0.0 , 0.0 , 0.0  # momentum components
    # energy definitions
    e_kin                 = (U^2 + V^2 + W^2) / (2*ρ)/ ρ
    e_pot                 = gravity * y
    e_int                 = c_v * (T - T_0) #internal_energy(T, q_tot, q_liq, q_ice)
    E                     = ρ * (e_int + e_kin + e_pot)  #* total_energy(e_kin, e_pot, T, q_tot, q_liq, q_ice)
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
end

function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)

    ArrayType = Array
    # CuArray option (TODO merge new master)

    brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                  range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)))
    
    # User defined periodicity in the topl assignment
    # brickrange defines the domain extents
    topl = BrickTopology(mpicomm, brickrange, periodicity=(false,false))

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
    initialcondition(Q, x...) = rising_thermal_bubble!(Val(dim), Q, DFloat(0), x...)
    Q = MPIStateArray(spacedisc, initialcondition)

    lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
      norm(Q₀) = %.16e""" eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(5, mpicomm) do (s=false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            @info @sprintf("""Update
                         simtime = %.16e
                         runtime = %s
                         norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                           Dates.format(convert(Dates.DateTime,
                                                Dates.now()-starttime[]),
                                        Dates.dateformat"HH:MM:SS"),
                           energy)
        end
    end

    #Define auxiliary states for post-processing
    npoststates = 6
    _P, _u, _v, _w, _rhoinv, _theta = 1:npoststates
    postnames = ("P", "u", "v", "w", "rhoinv", "theta")
    postprocessarray = MPIStateArray(spacedisc; nstate=npoststates)


    #Start computation
    step = [0]
    mkpath("vtk")
    cbvtk = GenericCallbacks.EveryXSimulationSteps(500) do (init=false)
        
        # Gather the values of the auxiliary quantities to write to VTK later
        # Store them into 
        DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, spacedisc,  Q) do postaux, Q, QV, aux            
            @inbounds let
                (postaux[_P], postaux[_u], postaux[_v], postaux[_w], postaux[_rhoinv], postaux[_theta])= preflux(Q, QV, aux)
            end
        end
        
        outprefix = @sprintf("vtk/cns_%dD_mpirank%04d_step%04d", dim,
                             MPI.Comm_rank(mpicomm), step[1])
        @debug "doing VTK output" outprefix
        writevtk(outprefix, Q, spacedisc, statenames,
                 postprocessarray, postnames)
        step[1] += 1
        nothing
    end
    
    # solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, ))
    solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))


    # Print some end of the simulation information
    engf = norm(Q)
    @info @sprintf """Finished
      norm(Q)            = %.16e
      norm(Q) / norm(Q₀) = %.16e
      norm(Q) - norm(Q₀) = %.16e""" engf engf/eng0 engf-eng0
    engf/eng0
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
    numelem = (20,30)
    dt = 0.01
    timeend = 800
    polynomialorder = 4
    for DFloat in (Float64,) #Float32)
        for dim = 2:2
            engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                            DFloat, dt)
        end
    end
end

isinteractive() || MPI.Finalize()

nothing
