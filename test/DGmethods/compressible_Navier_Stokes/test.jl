diff --git a/test/DGmethods/compressible_Navier_Stokes/dc_smagorinsky_sgs.jl b/test/DGmethods/compressible_Navier_Stokes/dc_smagorinsky_sgs.jl
index 4a376ab..3a822f4 100644
--- a/test/DGmethods/compressible_Navier_Stokes/dc_smagorinsky_sgs.jl
+++ b/test/DGmethods/compressible_Navier_Stokes/dc_smagorinsky_sgs.jl
@@ -31,10 +31,9 @@ end
 State labels
 """
 const _nstate = 6
-const _ρ, _U, _V, _W, _E, _QT = 1:_nstate
-const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E, QTid = _QT)
-const statenames = ("RHO", "U", "V", "W", "E", "QT")
-
+const _ρ, _ρu, _ρv, _ρw, _ρe_tot, _ρq_tot = 1:_nstate
+const stateid = (ρid = _ρ, ρuid = _ρu, ρvid = _ρv, ρwid = _ρw, ρe_totid = _ρe_tot, ρq_totid = _ρq_tot)
+const statenames = ("RHO", "ρu", "ρv", "ρw", "ρe_tot", "ρq_tot")
 
 """
 Viscous state labels
@@ -50,8 +49,13 @@ const _ngradstates = 7
 """
 Number of states being loaded for gradient computation
 """
-const _states_for_gradient_transform = (_ρ, _U, _V, _W, _E, _QT)
+const _states_for_gradient_transform = (_ρ, _ρu, _ρv, _ρw, _ρe_tot, _ρq_tot)
 
+"""
+Auxiliary States
+"""
+const _nauxstate = 7
+const _a_x, _a_y, _a_z, _a_dx, _a_dy, _a_dz, _a_Δsqr = 1:_nauxstate
 
 if !@isdefined integration_testing
     const integration_testing =
@@ -72,7 +76,7 @@ Problem Description
 """
 
 const numdims = 2
-Δx    = 50
+Δx    = 100
 Δy    = 50
 Δz    = 50
 Npoly = 4
@@ -96,33 +100,14 @@ const Nex = ceil(Int64, ratiox)
 const Ney = ceil(Int64, ratioy)
 const Nez = ceil(Int64, ratioz)
 
-# Equivalent grid-scale
-
-@info @sprintf """ ----------------------------------------------------"""
-@info @sprintf """   ______ _      _____ __  ________                  """     
-@info @sprintf """  |  ____| |    |_   _|  ...  |  __  |               """  
-@info @sprintf """  | |    | |      | | |   .   | |  | |               """ 
-@info @sprintf """  | |    | |      | | | |   | | |__| |               """
-@info @sprintf """  | |____| |____ _| |_| |   | | |  | |               """
-@info @sprintf """  | _____|______|_____|_|   |_|_|  |_|               """
-@info @sprintf """                                                     """
-@info @sprintf """ ----------------------------------------------------"""
-@info @sprintf """ Density Current                                     """
-@info @sprintf """   Resolution:                                       """ 
-@info @sprintf """     (Δx, Δy)   = (%.2e, %.2e)                       """ Δx Δy
-@info @sprintf """     (Nex, Ney) = (%d, %d)                           """ Nex Ney
-@info @sprintf """ ----------------------------------------------------"""
-
+DoF = (Nex*Ney*Nez)*(Npoly+1)^numdims*(_nstate)
+DoFstorage = (Nex*Ney*Nez)*(Npoly+1)^numdims*(_nstate + _nviscstates + _nauxstate + CLIMA.Grids._nvgeo) +
+             (Nex*Ney*Nez)*(Npoly+1)^(numdims-1)*2^numdims*(CLIMA.Grids._nsgeo)
 # -------------------------------------------------------------------------
 #md ### Auxiliary Function (Not required)
-#md # In this example the auxiliary function is used to store the spatial
-#md # coordinates. This may also be used to store variables for which gradients
-#md # are needed, but are not available through teh prognostic variable 
-#md # calculations. (An example of this will follow - in the Smagorinsky model, 
-#md # where a local Richardson number via potential temperature gradient is required)
+# We use the auxiliary function to carry the Cartesian coordinates x, y, z 
+# and the local grid spacing dx, dy, dz 
 # -------------------------------------------------------------------------
-const _nauxstate = 7
-const _a_x, _a_y, _a_z, _a_dx, _a_dy, _a_dz, _a_Δsqr = 1:_nauxstate
 @inline function auxiliary_state_initialization!(aux, x, y, z, dx, dy, dz)
     @inbounds begin
         aux[_a_x] = x
@@ -138,10 +123,9 @@ end
 # -------------------------------------------------------------------------
 # Preflux calculation: This function computes parameters required for the 
 # DG RHS (but not explicitly solved for as a prognostic variable)
-# In the case of the rising_thermal_bubble example: the saturation
-# adjusted temperature and pressure are such examples. Since we define
-# the equation and its arguments here the user is afforded a lot of freedom
-# around its behaviour. 
+# In the case of the density current example: the saturation
+# adjusted temperature and pressure as such examples. Note that in future
+# versions the preflux! function will be removed
 # The preflux function interacts with the following  
 # Modules: NumericalFluxes.jl 
 # functions: wavespeed, cns_flux!, bcstate!
@@ -149,12 +133,12 @@ end
 @inline function preflux(Q,VF, aux, _...)
     gravity::eltype(Q) = grav
     R_gas::eltype(Q) = R_d
-    @inbounds ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
+    @inbounds ρ, ρu, ρv, ρw, ρe_tot, ρq_tot = Q[_ρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_ρe_tot], Q[_ρq_tot]
     ρinv = 1 / ρ
     x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
-    u, v, w = ρinv * U, ρinv * V, ρinv * W
-    e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * y) / ρ
-    q_tot = QT / ρ
+    u, v, w = ρinv * ρu, ρinv * ρv, ρinv * ρw
+    e_int = (ρe_tot - (ρu^2 + ρv^2+ ρw^2)/(2*ρ) - ρ * gravity * y) / ρ
+    q_tot = ρq_tot / ρ
     # Establish the current thermodynamic state using the prognostic variables
     TS = PhaseEquil(e_int, q_tot, ρ)
     T = air_temperature(TS)
@@ -170,11 +154,11 @@ end
 @inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv, q_liq, T, θ)
   gravity::eltype(Q) = grav
   @inbounds begin 
-    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
+    ρ, ρu, ρv, ρw, ρe_tot, ρq_tot = Q[_ρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_ρe_tot], Q[_ρq_tot]
     x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
-    u, v, w = ρinv * U, ρinv * V, ρinv * W
-    e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * y) / ρ
-    q_tot = QT / ρ
+    u, v, w = ρinv * ρu, ρinv * ρv, ρinv * ρw
+    e_int = (ρe_tot - (ρu^2 + ρv^2+ ρw^2)/(2*ρ) - ρ * gravity * y) / ρ
+    q_tot = ρq_tot / ρ
     TS = PhaseEquil(e_int, q_tot, ρ)
     abs(n[1] * u + n[2] * v + n[3] * w) + soundspeed_air(TS)
   end
@@ -182,8 +166,9 @@ end
 
 # -------------------------------------------------------------------------
 # ### Physical Flux (Required)
-#md # Here, we define the physical flux function, i.e. the conservative form
-#md # of the equations of motion for the prognostic variables ρ, U, V, W, E, QT
+#md # Here, we define the physical flux function (compressible Navier Stokes), 
+#md # Discontinuous Galerkin requires the conservative form
+#md # of the equations of motion for the prognostic variables ρ, ρu, ρv, ρw, E, ρq_tot
 #md # $\frac{\partial Q}{\partial t} + \nabla \cdot \boldsymbol{F} = \boldsymbol {S}$
 #md # $\boldsymbol{F}$ contains both the viscous and inviscid flux components
 #md # and $\boldsymbol{S}$ contains source terms.
@@ -194,14 +179,14 @@ cns_flux!(F, Q, VF, aux, t) = cns_flux!(F, Q, VF, aux, t, preflux(Q,VF, aux)...)
 @inline function cns_flux!(F, Q, VF, aux, t, P, u, v, w, ρinv, q_liq, T, θ)
   gravity::eltype(Q) = grav
   @inbounds begin
-    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
+    ρ, ρu, ρv, ρw, ρe_tot, ρq_tot = Q[_ρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_ρe_tot], Q[_ρq_tot]
     # Inviscid contributions
-    F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
-    F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
-    F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
-    F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
-    F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
-    F[1, _QT], F[2, _QT], F[3, _QT] = u * QT  , v * QT     , w * QT 
+    F[1, _ρ], F[2, _ρ], F[3, _ρ]                = ρu          , ρv          , ρw
+    F[1, _ρu], F[2, _ρu], F[3, _ρu]             = u * ρu  + P , v * ρu      , w * ρu
+    F[1, _ρv], F[2, _ρv], F[3, _ρv]             = u * ρv      , v * ρv + P  , w * ρv
+    F[1, _ρw], F[2, _ρw], F[3, _ρw]             = u * ρw      , v * ρw      , w * ρw + P
+    F[1, _ρe_tot], F[2, _ρe_tot], F[3, _ρe_tot]                = u * (ρe_tot + P) , v * (ρe_tot + P) , w * (ρe_tot + P)
+    F[1, _ρq_tot], F[2, _ρq_tot], F[3, _ρq_tot] = u * ρq_tot  , v * ρq_tot  , w * ρq_tot 
 
     #Derivative of T and Q:
     vqx, vqy, vqz = VF[_qx], VF[_qy], VF[_qz]        
@@ -223,45 +208,35 @@ cns_flux!(F, Q, VF, aux, t) = cns_flux!(F, Q, VF, aux, t, preflux(Q,VF, aux)...)
     τ23 = τ32 = VF[_τ23] * ν_e
     
     # Viscous velocity flux (i.e. F^visc_u in Giraldo Restelli 2008)
-    F[1, _U] -= τ11 * f_R ; F[2, _U] -= τ12 * f_R ; F[3, _U] -= τ13 * f_R
-    F[1, _V] -= τ21 * f_R ; F[2, _V] -= τ22 * f_R ; F[3, _V] -= τ23 * f_R
-    F[1, _W] -= τ31 * f_R ; F[2, _W] -= τ32 * f_R ; F[3, _W] -= τ33 * f_R
+    F[1, _ρu] -= τ11 * f_R ; F[2, _ρu] -= τ12 * f_R ; F[3, _ρu] -= τ13 * f_R
+    F[1, _ρv] -= τ21 * f_R ; F[2, _ρv] -= τ22 * f_R ; F[3, _ρv] -= τ23 * f_R
+    F[1, _ρw] -= τ31 * f_R ; F[2, _ρw] -= τ32 * f_R ; F[3, _ρw] -= τ33 * f_R
 
     # Viscous Energy flux (i.e. F^visc_e in Giraldo Restelli 2008)
-    F[1, _E] -= u * τ11 + v * τ12 + w * τ13 + ν_e * k_μ * vTx 
-    F[2, _E] -= u * τ21 + v * τ22 + w * τ23 + ν_e * k_μ * vTy
-    F[3, _E] -= u * τ31 + v * τ32 + w * τ33 + ν_e * k_μ * vTz 
+    F[1, _ρe_tot] -= u * τ11 + v * τ12 + w * τ13 + ν_e * k_μ * vTx 
+    F[2, _ρe_tot] -= u * τ21 + v * τ22 + w * τ23 + ν_e * k_μ * vTy
+    F[3, _ρe_tot] -= u * τ31 + v * τ32 + w * τ33 + ν_e * k_μ * vTz 
   end
 end
 
 # -------------------------------------------------------------------------
-#md # Here we define a function to extract the velocity components from the 
-#md # prognostic equations (i.e. the momentum and density variables). This 
-#md # function is not required in general, but provides useful functionality 
-#md # in some cases. 
+#md # Here we specify the variables (in terms of the prognostic equations)
+#md # for which gradients are required. Required for viscous flow problems
 # -------------------------------------------------------------------------
 # Compute the velocity from the state
 gradient_vars!(gradient_list, Q, aux, t, _...) = gradient_vars!(gradient_list, Q, aux, t, preflux(Q,~,aux)...)
 @inline function gradient_vars!(gradient_list, Q, aux, t, P, u, v, w, ρinv, q_liq, T, θ)
     @inbounds begin
-        y = aux[_a_y]
-        # ordering should match states_for_gradient_transform
-        ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
-        E, QT = Q[_E], Q[_QT]
-        ρinv = 1 / ρ
+        ρ, ρu, ρv, ρw, ρe_tot, ρq_tot = Q[_ρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_ρe_tot], Q[_ρq_tot]
         gradient_list[1], gradient_list[2], gradient_list[3] = u, v, w
-        gradient_list[4], gradient_list[5], gradient_list[6] = E, QT, T
+        gradient_list[4], gradient_list[5], gradient_list[6] = ρe_tot, ρq_tot, T
         gradient_list[7] = ρ
     end
 end
 
 # -------------------------------------------------------------------------
 #md ### Viscous fluxes. 
-#md # The viscous flux function compute_stresses computes the components of 
-#md # the velocity gradient tensor, and the corresponding strain rates to
-#md # populate the viscous flux array VF. SijSij is calculated in addition
-#md # to facilitate implementation of the constant coefficient Smagorinsky model
-#md # (pending)
+# -------------------------------------------------------------------------
 @inline function compute_stresses!(VF, grad_vars, _...)
     gravity::eltype(VF) = grav
     @inbounds begin
@@ -269,15 +244,14 @@ end
         dvdx, dvdy, dvdz = grad_vars[1, 2], grad_vars[2, 2], grad_vars[3, 2]
         dwdx, dwdy, dwdz = grad_vars[1, 3], grad_vars[2, 3], grad_vars[3, 3]
         # compute gradients of moist vars and temperature
-        dqdx, dqdy, dqdz = grad_vars[1, 5], grad_vars[2, 5], grad_vars[3, 5]
         dTdx, dTdy, dTdz = grad_vars[1, 6], grad_vars[2, 6], grad_vars[3, 6]
         dρdx, dρdy, dρdz = grad_vars[1, 7], grad_vars[2, 7], grad_vars[3, 7]
         # virtual potential temperature gradient: for richardson calculation
         # strains
         # --------------------------------------------
         (S11,S22,S33,S12,S13,S23,SijSij) = SubgridScaleTurbulence.strainrate_tensor_components(dudx, dudy, dudz,
-                                                                                            dvdx, dvdy, dvdz,
-                                                                                            dwdx, dwdy, dwdz)
+                                                                                              dvdx, dvdy, dvdz,
+                                                                                              dwdx, dwdy, dwdz)
         #--------------------------------------------
         # deviatoric stresses
         VF[_τ11] = 2 * (S11 - (S11 + S22 + S33) / 3)
@@ -299,15 +273,13 @@ end
 @inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t, PM, uM, vM, wM, ρinvM, q_liqM, TM, θM)
     @inbounds begin
         x, y, z = auxM[_a_x], auxM[_a_y], auxM[_a_z]
-        ρM, UM, VM, WM, EM, QTM = QM[_ρ], QM[_U], QM[_V], QM[_W], QM[_E], QM[_QT]
-        # No flux boundary conditions
-        # No shear on walls (free-slip condition)
-        UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
-        QP[_U] = UM - 2 * nM[1] * UnM
-        QP[_V] = VM - 2 * nM[2] * UnM
-        QP[_W] = WM - 2 * nM[3] * UnM
-        #QP[_ρ] = ρM
-        #QP[_QT] = QTM
+        ρM, ρuM, ρvM, ρwM, ρe_totM, ρq_totM = QM[_ρ], QM[_ρu], QM[_ρv], QM[_ρw], QM[_ρe_tot], QM[_ρq_tot]
+        ρunM = nM[1] * ρuM + nM[2] * ρvM + nM[3] * ρwM
+        QP[_ρu] = ρuM - 2 * nM[1] * ρunM
+        QP[_ρv] = ρvM - 2 * nM[2] * ρunM
+        QP[_ρw] = ρwM - 2 * nM[3] * ρunM
+        QP[_ρ] = ρM
+        QP[_ρq_tot] = ρq_totM
         VFP .= 0 
         nothing
     end
@@ -320,7 +292,6 @@ Boundary correction for Neumann boundaries
   compute_stresses!(VF, 0) 
 end
 
-
 """
 Gradient term flux correction 
 """
@@ -328,72 +299,35 @@ Gradient term flux correction
     @inbounds begin
         n_Δgradient_list = similar(VF, Size(3, _ngradstates))
         for j = 1:_ngradstates, i = 1:3
+            # Viscous penalty for i=3 coordinate directions and _ngradstates gradient terms
             n_Δgradient_list[i, j] = nM[i] * (gradient_listP[j] - gradient_listM[j]) / 2
         end
         compute_stresses!(VF, n_Δgradient_list)
     end
 end
-# -------------------------------------------------------------------------
 
+# -------------------------------------------------------------------------
+# md ### Source Terms
+# -------------------------------------------------------------------------
 @inline function source!(S,Q,aux,t)
-    # Initialise the final block source term 
-    S .= 0
-
-    # Typically these sources are imported from modules
     @inbounds begin
         source_geopot!(S, Q, aux, t)
     end
 end
 
-@inline function source_sponge!(S, Q, aux, t)
-    y = aux[_a_y]
-    x = aux[_a_x]
-    U = Q[_U]
-    V = Q[_V]
-    W = Q[_W]
-    # Define Sponge Boundaries      
-    xc       = (xmax + xmin)/2
-    ysponge  = 0.85 * ymax
-    xsponger = xmax - 0.15*abs(xmax - xc)
-    xspongel = xmin + 0.15*abs(xmin - xc)
-    csxl  = 0.0
-    csxr  = 0.0
-    ctop  = 0.0
-    csx   = 1.0
-    ct    = 1.0 
-    #x left and right
-    #xsl
-    if (x <= xspongel)
-        csxl = csx * sinpi(1/2 * (x - xspongel)/(xmin - xspongel))^4
-    end
-    #xsr
-    if (x >= xsponger)
-        csxr = csx * sinpi(1/2 * (x - xsponger)/(xmax - xsponger))^4
-    end
-    #Vertical sponge:         
-    if (y >= ysponge)
-        ctop = ct * sinpi(1/2 * (y - ysponge)/(ymax - ysponge))^4
-    end
-    beta  = 1.0 - (1.0 - ctop)*(1.0 - csxl)*(1.0 - csxr)
-    beta  = min(beta, 1.0)
-    S[_U] -= beta * U  
-    S[_V] -= beta * V  
-    S[_W] -= beta * W
-end
-
 @inline function source_geopot!(S,Q,aux,t)
     gravity::eltype(Q) = grav
     @inbounds begin
-        ρ, U, V, W, E  = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
-        S[_V] += - ρ * gravity
+        # FIXME: coordinate direction needs to be consistent with Overleaf docs.
+        ρ, ρu, ρv, ρw, ρe_tot  = Q[_ρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_ρe_tot]
+        S[_ρv] += - ρ * gravity
     end
 end
 
 
-# ------------------------------------------------------------------
-# -------------END DEF SOURCES-------------------------------------# 
-
-# initial condition
+# -------------------------------------------------------------------------
+# md ### Initial Condition
+# -------------------------------------------------------------------------
 function density_current!(dim, Q, t, x, y, z, _...)
     DFloat                = eltype(Q)
     R_gas::DFloat         = R_d
@@ -405,7 +339,7 @@ function density_current!(dim, Q, t, x, y, z, _...)
     q_tot::DFloat         = 0
     q_liq::DFloat         = 0
     q_ice::DFloat         = 0 
-    # perturbation parameters for rising bubble
+    # thermal perturbation parameters for falling bubble
     rx                    = 4000
     ry                    = 2000
     xc                    = 0
@@ -424,13 +358,13 @@ function density_current!(dim, Q, t, x, y, z, _...)
 
     P                     = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
     T                     = P / (ρ * R_gas) # temperature
-    U, V, W               = 0.0 , 0.0 , 0.0  # momentum components
+    ρu, ρv, ρw            = 0.0 , 0.0 , 0.0  # momentum components
     # energy definitions
-    e_kin                 = (U^2 + V^2 + W^2) / (2*ρ)/ ρ
+    e_kin                 = (ρu^2 + ρv^2 + ρw^2) / (2*ρ)/ ρ
     e_pot                 = gravity * y
     e_int                 = internal_energy(T, qvar)
-    E                     = ρ * (e_int + e_kin + e_pot)  #* total_energy(e_kin, e_pot, T, q_tot, q_liq, q_ice)
-    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]= ρ, U, V, W, E, ρ * q_tot
+    ρe_tot                = ρ * (e_int + e_kin + e_pot)  #* total_energy(e_kin, e_pot, T, q_tot, q_liq, q_ice)
+    @inbounds Q[_ρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_ρe_tot], Q[_ρq_tot] = ρ, ρu, ρv, ρw, ρe_tot, ρ * q_tot
 end
 
 function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)
@@ -438,7 +372,7 @@ function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)
     brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                   range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)))
     
-    # User defined periodicity in the topl assignment
+    # User defined periodicity
     # brickrange defines the domain extents
     topl = StackedBrickTopology(mpicomm, brickrange, periodicity=(false,false))
 
@@ -450,7 +384,6 @@ function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)
     numflux!(x...) = NumericalFluxes.rusanov!(x..., cns_flux!, wavespeed, preflux)
     numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., cns_flux!, bcstate!, wavespeed, preflux)
 
-    # spacedisc = data needed for evaluating the right-hand side function
     spacedisc = DGBalanceLaw(grid = grid,
                              length_state_vector = _nstate,
                              flux! = cns_flux!,
@@ -469,12 +402,11 @@ function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)
                              auxiliary_state_initialization!,
                              source! = source!)
 
-    # This is a actual state/function that lives on the grid
     initialcondition(Q, x...) = density_current!(Val(dim), Q, DFloat(0), x...)
     Q = MPIStateArray(spacedisc, initialcondition)
 
     lsrk = LSRK54CarpenterKennedy(spacedisc, Q; dt = dt, t0 = 0)
-
+    # dt can be updated via callback in solve! and updatedt! function 
     eng0 = norm(Q)
     @info @sprintf """Starting
       norm(Q₀) = %.16e""" eng0
@@ -572,16 +504,35 @@ let
     else
         global_logger(NullLogger())
     end
-    # User defined number of elements
-    # User defined timestep estimate
-    # User defined simulation end time
-    # User defined polynomial order 
+    
     numelem = (Nex,Ney)
     dt = 0.01
     timeend = 900
     polynomialorder = Npoly
     DFloat = Float64
     dim = numdims
+  
+    if MPI.Comm_rank(mpicomm) == 0
+      @info @sprintf """ ------------------------------------------------------"""
+      @info @sprintf """   ______ _      _____ __  ________                    """     
+      @info @sprintf """  |  ____| |    |_   _|  ...  |  __  |                 """  
+      @info @sprintf """  | |    | |      | | |   .   | |  | |                 """ 
+      @info @sprintf """  | |    | |      | | | |   | | |__| |                 """
+      @info @sprintf """  | |____| |____ _| |_| |   | | |  | |                 """
+      @info @sprintf """  | _____|______|_____|_|   |_|_|  |_|                 """
+      @info @sprintf """                                                       """
+      @info @sprintf """ ------------------------------------------------------"""
+      @info @sprintf """ Dycoms                                                """
+      @info @sprintf """   Resolution:                                         """ 
+      @info @sprintf """     (Δx, Δy, Δz)   = (%.2e, %.2e, %.2e)               """ Δx Δy Δz
+      @info @sprintf """     (Nex, Ney, Nez) = (%d, %d, %d)                    """ Nex Ney Nez
+      @info @sprintf """     DoF = %d                                          """ DoF
+      @info @sprintf """     Minimum necessary memory to run this test: %g GBs """ (DoFstorage * sizeof(DFloat))/1000^3
+      @info @sprintf """     Time step dt: %.2e                                """ dt
+      @info @sprintf """     End time  t : %.2e                                """ timeend
+      @info @sprintf """ ------------------------------------------------------"""
+    end
+    
     engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                     DFloat, dt)
 end
