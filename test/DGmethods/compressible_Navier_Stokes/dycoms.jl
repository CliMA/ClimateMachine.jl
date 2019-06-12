# # DYCOMS:
#
## Introduction
#
# This driver defines the initial condition for the
# Dynamics and Chemistry of Marine Stratocumulus (DYCOMS)
# LES test described in
#
# [1] B. Stevens et al. (2005) Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus, Mon. Wea. Rev. 133:1443-1462
#
#
# Below is a program interspersed with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref ex_001_periodic_advection-plain-program).
#
# ## Commented Program

#------------------------------------------------------------------------------

# ### Preliminaries
# Load modules that are used in the CliMA project.
# These are general modules not necessarily specific
# to CliMA
using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using DelimitedFiles
using Dierckx

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
#md nothing # hide

# The prognostic equations are conservations laws solved with respect to
# the dynamics and moisture quantities:
# ```math
#  ρ,\;(ρu),\;(ρv),\;(ρw),;(ρe_tot),\;(ρq_tot)
# ```
#md nothing # hide

# Load the MoistThermodynamics and PlanetParameters modules:
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters: R_d, cp_d, grav, cv_d, MSLP, T_0
#md nothing # hide

# Define the ids for each dynamics and moist states:
const _nstate = 6
const _ρ, _U, _V, _W, _E, _QT = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E, QTid = _QT)
const statenames    = ("ρ", "U", "V", "W", "E", "Qtot")
const auxstatenames = ("ax","ay","az","Qliq")

const _nviscstates = 6
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23 = 1:_nviscstates

const _ngradstates = 3
const _states_for_gradient_transform = (_ρ, _U, _V, _W)
#md nothing # hide

if !@isdefined integration_testing
    const integration_testing =
        parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

const γ_exact = 7 // 5
const μ_exact = 10

# Domain:
const xmin =    0
const zmin =    0
const ymin =    0
const xmax = 1500 #domain length
const zmax =  150 #domain depth
const ymax = 1500 #domain height (vertical)
const xc   = (xmax + xmin) / 2
const zc   = (zmax + zmin) / 2
const yc   = (ymax + ymin) / 2

# NOTICE: Gravity in the current setting acts along y
# This will be changed to z in the next revision
#md nothing # hide


# -------------------------------------------------------------------------
# Preflux calculation: This function computes parameters required for the 
# DG RHS (but not explicitly solved for as a prognostic variable)
# In the case of the dycoms test: the saturation
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
    @inbounds ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    ρinv = 1 / ρ

    xvert = aux[_a_y]
    
    u, v, w = ρinv * U, ρinv * V, ρinv * W
    e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * xvert) / ρ
    qt = QT / ρ
    # Establish the current thermodynamic state using the prognostic variables
    TS           = PhaseEquil(e_int, qt, ρ)
    T            = air_temperature(TS)
    P            = air_pressure(TS) # Test with dry atmosphere
    q_phase_part = PhasePartition(TS)
        
    (P, u, v, w, ρinv)
    # Preflux returns pressure, 3 velocity components, and 1/ρ
end

# -------------------------------------------------------------------------
# max eigenvalue
@inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv)
    γ::eltype(Q) = γ_exact
    @inbounds abs(n[1] * u + n[2] * v + n[3] * w) + sqrt(ρinv * γ * P)
end


# -------------------------------------------------------------------------
# ### read sounding
#md # 
#md # The sounding file contains the following quantities along a 1D column.
#md # It needs to have the following structure:
#md #
#md # z[m]   theta[K]  q[g/kg]   u[m/s]   v[m/s]   p[Pa]
#md # ...      ...       ...      ...      ...      ...
#md #
#md #
# -------------------------------------------------------------------------
function read_sounding()
    #read in the original squal sounding
    fsounding  = open(joinpath(@__DIR__, "../soundings/sounding_DYCOMS_TEST1.dat"))
    sounding = readdlm(fsounding)
    close(fsounding)
    (nzmax, ncols) = size(sounding)
    if nzmax == 0
        error("SOUNDING ERROR: The Sounding file is empty!")
    end
    return (sounding, nzmax, ncols)
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
@inline function cns_flux!(F, Q, VF, aux, t, P, u, v, w, ρinv)
    @inbounds begin
        ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
        # Inviscid contributions 
        F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
        F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
        F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
        F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
        F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
        F[1, _QT], F[2, _QT], F[3, _QT] = u * QT  , v * QT     , w * QT 
        # Stress tensor
        τ11, τ22, τ33 = VF[_τ11], VF[_τ22], VF[_τ33]
        τ12 = τ21 = VF[_τ12]
        τ13 = τ31 = VF[_τ13]
        τ23 = τ32 = VF[_τ23]
        # Viscous contributions
        F[1, _U] -= τ11; F[2, _U] -= τ12; F[3, _U] -= τ13
        F[1, _V] -= τ21; F[2, _V] -= τ22; F[3, _V] -= τ23
        F[1, _W] -= τ31; F[2, _W] -= τ32; F[3, _W] -= τ33
        # Energy dissipation
        F[1, _E] -= u * τ11 + v * τ12 + w * τ13
        F[2, _E] -= u * τ21 + v * τ22 + w * τ23
        F[3, _E] -= u * τ31 + v * τ32 + w * τ33
    end
end

# -------------------------------------------------------------------------
#md # Here we define a function to extract the velocity components from the 
#md # prognostic equations (i.e. the momentum and density variables). This 
#md # function is not required in general, but provides useful functionality 
#md # in some cases. 
# -------------------------------------------------------------------------
# Compute the velocity from the state
@inline function velocities!(vel, Q, _...)
    @inbounds begin
        # ordering should match states_for_gradient_transform
        ρ, U, V, W = Q[_ρ], Q[_U], Q[_V], Q[_W]
        ρinv = 1 / ρ
        vel[1], vel[2], vel[3] = ρinv * U, ρinv * V, ρinv * W
    end
end

# -------------------------------------------------------------------------
#md ### Auxiliary Function (Not required)
#md # In this example the auxiliary function is used to store the spatial
#md # coordinates. This may also be used to store variables for which gradients
#md # are needed, but are not available through the prognostic variable 
#md # calculations. (An example of this will follow - in the Smagorinsky model, 
#md # where a local Richardson number via potential temperature gradient is required)
# -------------------------------------------------------------------------
const _nauxstate = 4
const _a_x, _a_y, _a_z, _a_ql = 1:_nauxstate
@inline function auxiliary_state_initialization!(aux, x, y, z)
    @inbounds begin
        aux[_a_x]  = x
        aux[_a_y]  = y
        aux[_a_z]  = z
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
        # --------------------------------------------
        # SMAGORINSKY COEFFICIENT COMPONENTS
        # --------------------------------------------
        SijSij = (ϵ11 + ϵ22 + ϵ33
                  + 2.0 * ϵ12
                  + 2.0 * ϵ13 
                  + 2.0 * ϵ23) 
        # mod strain rate ϵ ---------------------------
        # deviatoric stresses
        VF[_τ11] = 2μ * (ϵ11 - (ϵ11 + ϵ22 + ϵ33) / 3)
        VF[_τ22] = 2μ * (ϵ22 - (ϵ11 + ϵ22 + ϵ33) / 3)
        VF[_τ33] = 2μ * (ϵ33 - (ϵ11 + ϵ22 + ϵ33) / 3)
        VF[_τ12] = 2μ * ϵ12
        VF[_τ13] = 2μ * ϵ13
        VF[_τ23] = 2μ * ϵ23

    end
end
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# generic bc for 2d , 3d
@inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t, PM, uM, vM, wM, ρinvM)
    @inbounds begin
        x, y, z = auxM[_a_x], auxM[_a_y], auxM[_a_z]
        ρM, UM, VM, WM, EM, QTM = QM[_ρ], QM[_U], QM[_V], QM[_W], QM[_E], QM[_QT]
        UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
        QP[_U] = UM - 2 * nM[1] * UnM
        QP[_V] = VM - 2 * nM[2] * UnM
        QP[_W] = WM - 2 * nM[3] * UnM
        QP[_ρ] = ρM
        QP[_E] = EM
        QP[_QT] = QTM
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
# -----------------------------------------------------------------
# --------------DEFINE SOURCES HERE -------------------------------
# -----------------------------------------------------------------
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
        source_geopot!(S, Q, aux, t)
    end
end

# Sponge: classical Rayleigh type absorbing layers:
@inline function sponge_rectangular(S, Q, aux)

    
    U, V, W = Q[_U], Q[_V], Q[_W]
    x, y, z = aux[_a_x], aux[_a_y], aux[_a_z]
    
    xmin = brickrange[1][1]
    xmax = brickrange[1][end]
    ymin = brickrange[2][1]
    ymax = brickrange[2][end]
    zmin = brickrange[3][1]
    zmax = brickrange[3][end]
    
    # Define Sponge Boundaries      
    xc       = (xmax + xmin)/2
    yc       = (ymax + ymin)/2
    
    zsponge  = 0.85 * zmax
    xsponger = xmax - 0.15*abs(xmax - xc)
    xspongel = xmin + 0.15*abs(xmin - xc)
    ysponger = ymax - 0.15*abs(ymax - yc)
    yspongel = ymin + 0.15*abs(ymin - yc)
    
    csxl, csxr  = 0.0, 0.0
    csyl, csyr  = 0.0, 0.0
    ctop        = 0.0
    
    csx         = 0.0
    csy         = 0.0
    ct          = 1.0
       
    #x left and right
    #xsl
    if (x <= xspongel)
        csxl = csx * sinpi(1/2 * (x - xspongel)/(xmin - xspongel))^4
    end
    #xsr
    if (x >= xsponger)
        csxr = csx * sinpi(1/2 * (x - xsponger)/(xmax - xsponger))^4
    end        
    #y left and right
    #ysl
    if (y <= yspongel)
        csyl = csy * sinpi(1/2 * (y - yspongel)/(ymin - yspongel))^4
    end
    #ysr
    if (y >= ysponger)
        csyr = csy * sinpi(1/2 * (y - ysponger)/(ymay - ysponger))^4
    end
    
    #Vertical sponge:         
    if (z >= zsponge)
        ctop = ct * sinpi(1/2 * (z - zsponge)/(zmax - zsponge))^4
    end

    beta  = 1.0 - (1.0 - ctop) * (1.0 - csxl)*(1.0 - csxr) * (1.0 - csyl)*(1.0 - csyr)
    beta  = min(beta, 1.0)
    alpha = 1.0 - beta

    @inbounds begin
        S[_U] -= beta * U
        S[_V] -= beta * V
        S[_W] -= beta * W
    end
    
end
#---END SPONGE

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
This function specifies the initial conditions
for the dycoms driver. 
"""
function dycoms!(dim, Q, t, x, y, z, _...)
    DFloat         = eltype(Q)
    p0::DFloat      = MSLP
    gravity::DFloat = grav
    
    # ----------------------------------------------------
    # GET DATA FROM INTERPOLATED ARRAY ONTO VECTORS
    # This driver accepts data in 6 column format
    # ----------------------------------------------------
    (sounding, _, ncols) = read_sounding()
    
    # WARNING: Not all sounding data is formatted/scaled 
    # the same. Care required in assigning array values
    # height theta qv    u     v     pressure
    zinit, tinit, qinit, uinit, vinit, pinit  = sounding[:, 1],
    sounding[:, 2],
    sounding[:, 3],
    sounding[:, 4],
    sounding[:, 5],
    sounding[:, 6]    
    #------------------------------------------------------
    # GET SPLINE FUNCTION
    #------------------------------------------------------
    spl_tinit    = Spline1D(zinit, tinit; k=1)
    spl_qinit    = Spline1D(zinit, qinit; k=1)
    spl_uinit    = Spline1D(zinit, uinit; k=1)
    spl_vinit    = Spline1D(zinit, vinit; k=1)
    spl_pinit    = Spline1D(zinit, pinit; k=1)
    # --------------------------------------------------
    # INITIALISE ARRAYS FOR INTERPOLATED VALUES
    # --------------------------------------------------
    xvert          = y
    
    datat          = spl_tinit(xvert)
    dataq          = spl_qinit(xvert)
    datau          = spl_uinit(xvert)
    datav          = spl_vinit(xvert)
    datap          = spl_pinit(xvert)
    dataq          = dataq * 1.0e-3
    
    randnum   = rand(1)[1] / 100

    θ_liq = datat
    q_tot = dataq + randnum * dataq
    P     = datap    
    T     = air_temperature_from_liquid_ice_pottemp(θ_liq, P, PhasePartition(q_tot))
    ρ     = air_density(T, P)
        
    # energy definitions
    u, v, w     = 0*datau, 0*datav, 0.0 #geostrophic. TO BE BUILT PROPERLY if Coriolis is considered
    U           = ρ * u
    V           = ρ * v
    W           = ρ * w
    e_kin       = (u^2 + v^2 + w^2) / 2  
    e_pot       = gravity * xvert
    e_int       = internal_energy(T, PhasePartition(q_tot))
    E           = ρ * total_energy(e_kin, e_pot, T, PhasePartition(q_tot))
    
    #Get q_liq and q_ice
    TS           = PhaseEquil(e_int, q_tot, ρ)
    q_phase_part = PhasePartition(TS)
    
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]= ρ, U, V, W, E, ρ * q_tot
    
end

function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)

    ArrayType = Array
    # CuArray option (TODO merge new master)

    brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                  range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)),
                  range(DFloat(zmin), length=Ne[3]+1, DFloat(zmax)))
    
    # User defined periodicity in the topl assignment
    # brickrange defines the domain extents
    topl = BrickTopology(mpicomm, brickrange, periodicity=(false,false,false))

    grid = DiscontinuousSpectralElementGrid(topl,
                                            FloatType = DFloat,
                                            DeviceArray = ArrayType,
                                            polynomialorder = N)
    
    numflux!(x...)   = NumericalFluxes.rusanov!(x..., cns_flux!, wavespeed, preflux)
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
                             gradient_transform! = velocities!,
                             viscous_transform! = compute_stresses!,
                             viscous_penalty! = stresses_penalty!,
                             viscous_boundary_penalty! = stresses_boundary_penalty!,
                             auxiliary_state_length = _nauxstate,
                             auxiliary_state_initialization! =
                             auxiliary_state_initialization!,
                             source! = source!)

    # This is a actual state/function that lives on the grid
    initialcondition(Q, x...) = dycoms!(Val(dim), Q, DFloat(0), x...)
    Q = MPIStateArray(spacedisc, initialcondition)

    lsrk = LSRK54CarpenterKennedy(spacedisc, Q; dt = dt, t0 = 0)
    
    eng0 = norm(Q)
    @info @sprintf """Starting norm(Q₀) = %.16e""" eng0

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
                                                Dates.dateformat"HH:MM:SS"),energy)
        end
    end

    step = [0]
    mkpath("vtk")
    cbvtk = GenericCallbacks.EveryXSimulationSteps(10) do (init=false)
        outprefix = @sprintf("vtk/cns_%dD_mpirank%04d_step%04d", dim,
                             MPI.Comm_rank(mpicomm), step[1])
        @debug "doing VTK output" outprefix
        writevtk(outprefix, Q, spacedisc, statenames, spacedisc.auxstate, auxstatenames)
        
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
    engf / eng0
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
    numelem = (10,10, 1)
    dt = 1e-2
    timeend = 10
    polynomialorder = 5
    for DFloat in (Float64,) #Float32)
        for dim = 3:3
            engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                            DFloat, dt)
        end
    end
end

isinteractive() || MPI.Finalize()

nothing
