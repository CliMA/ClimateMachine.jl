```math
\newcommand{\paramT}[1]{     \textcolor{blue}{  \text{#1}}}
\newcommand{\hyperparamT}[1]{\textcolor{orange}{\text{#1}}}
\newcommand{\simparamT}[1]{  \textcolor{purple}{\text{#1}}}

\newcommand{\exp}[1]{\mathrm{exp}\left(#1\right)}
\newcommand{\atan}[1]{\mathrm{atan}\left(#1\right)}
\newcommand{\sign}[1]{\mathrm{sign}\left(#1\right)}
\newcommand{\erf}[1]{\mathrm{erf}\left(#1\right)}
\newcommand{\erfinv}[1]{\mathrm{erfinv}\left(#1\right)}

\newcommand{\param}[1]{     \textcolor{blue}{  #1}}
\newcommand{\hyperparam}[1]{\textcolor{orange}{#1}}
\newcommand{\simparam}[1]{  \textcolor{purple}{#1}}

\newcommand{\CROSS}{\times}
\newcommand{\GRAD}{\nabla}
\newcommand{\DOT}{\bullet}
\newcommand{\PD}{\partial}
\newcommand{\DM}[1]{\langle #1 \rangle}
\newcommand{\iEnv}{e}
\newcommand{\SD}[2]{{\overline{#1}}_{#2}}
\newcommand{\SDi}[1]{{\SD{#1}{i}}}
\newcommand{\SDj}[1]{{\SD{#1}{j}}}
\newcommand{\SDe}[1]{{\SD{#1}{\iEnv}}}
\newcommand{\SDiog}[2]{#1_{#2}}
\newcommand{\SDio}[1]{{\SDiog{#1}{i}}}
\newcommand{\SDjo}[1]{{\SDiog{#1}{j}}}
\newcommand{\SDeo}[1]{{\SDiog{#1}{\iEnv}}}
\newcommand{\aSD}[2]{{#1}_{#2}}
\newcommand{\aSDi}[1]{\aSD{#1}{i}}
\newcommand{\aSDj}[1]{\aSD{#1}{j}}
\newcommand{\aSDe}[1]{\aSD{#1}{\iEnv}}
\newcommand{\otherDefs}{where additional variable definitions are in:}

\newcommand{\IntraCVSDi}[2]{\overline{{#1}_{i      }'{#2}_{i      }'}}
\newcommand{\IntraCVSDj}[2]{\overline{{#1}_{j      }'{#2}_{j      }'}}
\newcommand{\IntraCVSDe}[2]{\overline{{#1}_{\iEnv{}}'{#2}_{\iEnv{}}'}}

\newcommand{\InterCVSDi}[2]{\overline{{#1}_{i      }'}\overline{{#2}_{i      }'}}
\newcommand{\InterCVSDj}[2]{\overline{{#1}_{j      }'}\overline{{#2}_{j      }'}}
\newcommand{\InterCVSDe}[2]{\overline{{#1}_{\iEnv{}}'}\overline{{#2}_{\iEnv{}}'}}

\newcommand{\TCV}[2]{\langle {#1}^*{#2}^* \rangle}

\newcommand{\BC}[1]{{#1|_{z_{min}}}}
\newcommand{\BCT}[1]{{#1|_{z_{max}}}}
\newcommand{\BCB}[1]{{#1|_{z_{min}}}}
\newcommand{\BCG}[1]{{#1|_{z_{boundary}}}}

\newcommand{\WindSpeed}{|u|}
\newcommand{\LayerThickness}{\param{\Delta z}}
\newcommand{\SurfaceRoughness}[1]{\param{z_{0#1}}}
\newcommand{\SensibleSurfaceHeatFlux}{F_{\mathrm{sensible}''}}
\newcommand{\LatentSurfaceHeatFlux}{F_{\mathrm{latent}''}}
\newcommand{\FrictionVelocity}{u_*}
\newcommand{\Buoyancy}{b}
\newcommand{\BuoyancyGrad}{\PD_z \Buoyancy}
\newcommand{\BuoyancyFlux}{\IntraCVSDi{w}{\theta}}
\newcommand{\TemperatureScale}{\theta_*}
\newcommand{\SurfaceMomentumFlux}{\BC{\overline{w'u'}}}
\newcommand{\SurfaceHeatFlux}{\BC{\overline{w'\theta'}}}
\newcommand{\SurfaceBuoyancyFlux}{\BC{\IntraCVSDi{w}{\theta}}}
\newcommand{\ConvectiveVelocity}{{w_*}} % Convective velocity near the surface
\newcommand{\InversionHeight}{{z_*}}
\newcommand{\MOLen}{\Lambda_{M-O}}
\newcommand{\zLL}{\param{z_{||}}} % z at the first surface level (we should make this grid-independent)

\newcommand{\qt}{q_{\mathrm{tot}}}
\newcommand{\ql}{q_{\mathrm{liq}}}
\newcommand{\qi}{q_{\mathrm{ice}}}
\newcommand{\qv}{q_{\mathrm{vap}}}
\newcommand{\qvsat}{q_{\mathrm{vap}}^*}
\newcommand{\pvsat}{p_{\mathrm{vap}}^*}
\newcommand{\qc}{q_{\mathrm{con}}}
\newcommand{\ThetaL}{{\theta_{\mathrm{liq}}}}
\newcommand{\ThetaVap}{{\theta_{\mathrm{vap}}}}
\newcommand{\ThetaVirt}{{\theta_{\mathrm{virt}}}}
\newcommand{\ThetaRho}{{\theta_{\rho}}}
\newcommand{\ThetaLiqIce}{{\theta_{\mathrm{liq-ice}}}}
\newcommand{\ThetaLiqIceSat}{{\theta^*_{\mathrm{liq-ice}}}}
\newcommand{\ThetaDry}{{\theta_{\mathrm{dry}}}}
\newcommand{\h}{{e_{\mathrm{int}}}}
\newcommand{\hint}{e_{\mathrm{int}}}
\newcommand{\htot}{e_{\mathrm{tot}}}

\newcommand{\alphaRef}{{\alpha}_0}
\newcommand{\rhoRef}{{\rho}_0}
\newcommand{\pRef}{{p}_0}
\newcommand{\Heaviside}{\mathcal H}

\newcommand{\alphaLL}{\alphaRef|_{\zLL}}
\newcommand{\uH}{\simparam{\mathbf{u}_h}}

\newcommand{\CoriolisParam}{\hyperparam{\mathrm{coriolis\_param}}}
\newcommand{\SubsidenceParam}{\hyperparam{\mathrm{subsidence}}}
\newcommand{\betaM}{\hyperparam{\beta_m}}
\newcommand{\betaH}{\hyperparam{\beta_h}}
\newcommand{\gammaM}{\hyperparam{\gamma_m}}
\newcommand{\gammaH}{\hyperparam{\gamma_h}}

\newcommand{\PTilde}{\param{\tilde{p}}}
\newcommand{\VKConst}{\param{\kappa_{\mathrm{Von-Karman}}}}
\newcommand{\Nsd}{\hyperparam{N_{sd}}}
\newcommand{\grav}{\param{g}}
\newcommand{\TZero}{\param{T_{0}}}
\newcommand{\RefHintV}{\param{{\hint}_{v,0}}}
\newcommand{\RefHintI}{\param{{\hint}_{i,0}}}

\newcommand{\epsvi}{\param{\varepsilon_{vi}}}
\newcommand{\MRatio}{\param{M_{\mathrm{ratio}}}}
\newcommand{\Rd}{\param{R_d}}
\newcommand{\Rv}{\param{R_v}}
\newcommand{\Cp}[1]{\param{c_{p#1}}}
\newcommand{\Cv}[1]{\param{c_{v#1}}}
\newcommand{\Cvd}{\Cv{d}}
\newcommand{\Cvv}{\Cv{v}}
\newcommand{\Cvl}{\Cv{l}}
\newcommand{\Cvi}{\Cv{i}}

\newcommand{\DeltaCp}{\param{\Delta c_p}}
\newcommand{\TTriple}{\param{T_{\mathrm{tr}}}}
\newcommand{\PTriple}{\param{p_{\mathrm{tr}}}}
\newcommand{\TFreeze}{\param{T_{\mathrm{freeze}}}}

\newcommand{\RefLHv}{\param{L_{v,0}}}
\newcommand{\RefLHs}{\param{L_{s,0}}}
\newcommand{\RefLHf}{\param{L_{f,0}}}
\newcommand{\LatentHeatV}[1]{L_{vap}(#1)}
\newcommand{\LatentHeatS}[1]{L_{sub}(#1)}
\newcommand{\LatentHeatF}[1]{L_{fus}(#1)}
```

# Eddy-Diffusivity Mass-Flux (EDMF) equations

This document is concerned with defining the set of equations solved in the atmospheric turbulence convection model: the EDMF equations. Color-coding is used to indicate:

 - $\paramT{Constant parameters that are fixed in space and time (e.g., those defined in PlanetParameters.jl)}$

 - $\simparamT{Single column (SC) inputs (e.g., variables that are fed into the SC model from the dynamical core (e.g., horizontal velocity))}$

 - $\hyperparamT{Tunable hyper-parameters that will need to be changeable, but will only include single numbers (e.g., Float64)}$

## Domain decomposition

While our model is 1D along $z$ and there is no spatial discretization in the horizontal directions ($x$ and $y$), the horizontal space is broken into $\Nsd$ ($\sim$ 5-10) "bins", or "subdomains" (SDs), denoted by subscript $i$, where $1 \le i \le \Nsd$. One of the subdomains, the "environment", is treated different compared to others, termed "updrafts". This environment subdomain is denoted with a special index $\iEnv{}$ (which we usually set to 1). For dummy variables $\phi$ and $\psi$, we use several domain and SD representations of interest:

```math
\begin{align}
  \SDi{\phi}                                                                                   \quad & \text{horizontal mean of variable $\phi$ over SD $i$}, \\
  \SDi{\phi}' = \phi_i - \SDi{\phi}                                                            \quad & \text{fluctuations of $\phi$ about the SD mean}, \\
  \IntraCVSDi{\phi}{\psi}                                                                      \quad & \text{intra subdomain covariance}, \\
  \DM{\phi} = \sum_i \aSDi{a} \SDi{\phi}                                                       \quad & \text{horizontal mean of $\phi$ over the total domain}, \\
  \SDi{\phi}^* = \SDi{\phi} - \DM{\phi}                                                        \quad & \text{difference between SD & domain means}, \\
  \InterCVSDi{\phi}{\psi}                                                                      \quad & \text{inter subdomain covariance among SD means}, \\
  \phi^* = \phi - \DM{\phi}                                                                    \quad & \text{difference between SD & domain means}, \\
  \TCV{\phi}{\psi} = \sum_{\forall i} a_i \IntraCVSDi{\phi}{\psi} +
  \sum_{\forall i} \sum_{\forall j}  \aSDi{a} \aSDj{a} \SDi{\phi}(\SDi{\psi} - \SDj{\psi})     \quad & \text{total covariance}.
\end{align}
```

Here, $\SDi{\phi}$ and $\SDi{\psi}$ are a dummy variables for the following 7 unknowns:
```math
\begin{align}
  \SDi{w}                  & \quad \text{vertical velocity}, \\
  \SDi{\h}                 & \quad \text{internal energy},  \\
  \SDi{\qt}                & \quad \text{total water specific humidity},  \\
  \SDi{TKE}                & \quad \text{turbulent kinetic energy ($0.5(\IntraCVSDi{u}{u}+\IntraCVSDi{v}{v}+\IntraCVSDi{w}{w})$)},  \\
  \IntraCVSDi{\h}{\h}      & \quad \text{intra subdomain covariance of $\h'$ and $\h'$},  \\
  \IntraCVSDi{\qt}{\qt}    & \quad \text{intra subdomain covariance of $\qt'$ and $\qt'$}, \\
  \IntraCVSDi{\h}{\qt}     & \quad \text{intra subdomain covariance of $\h'$ and $\qt'$}.
\end{align}
```

From the large-scale model perspective, $\DM{\phi}$ represents the resolved grid-scale (GS) mean, and $\TCV{\phi}{\psi}$ represents the SGS fluxes and (co)-variances of scalars that need to be parameterized. Equations in the following sections, \eqref{eq:AreaFracGov}, \eqref{eq:1stMoment} and \eqref{eq:2ndMoment}, are solved on $z_{min} \le z \le z_{max}$ and $t \ge 0$. There are $8 \Nsd$ equations in total.

## Domain averaged equations
The EDMF model can be used in the context of a stand-alone single column, or integrated with a dynamical core. Either way, the EDMF model relies on domain-averaged variables, which may be prescribed or solved for. Taking an area fraction-weighted average of the SD equations yields the domain-averaged equations (which should be consistent with variables in the dynamical core).

The domain-averaged equations for $\DM{\phi} \in [w, \qt, \h, \uH]$ are:

```math
\begin{align}
\PD_t (\rhoRef{} \DM{\phi})
+ \PD_z (\rhoRef{} \DM{w} \DM{\phi})
+ \nabla_h \DOT \left( \rhoRef{} \DM{\phi} \otimes \DM{\phi} \right)
= \\
  \DM{S}_{\text{diff}}
+ \DM{S}_{\text{press}}
+ \DM{S}_{\text{coriolis}}
+ \DM{S}_{\text{subsidence}},
\end{align}
```
where
```math
\begin{align}
\DM{S}_{\text{diff}}       & = \PD_z (\rhoRef{} \aSDe{a} K_{\iEnv{}} \PD_z \DM{\phi}), \label{eq:gm_diffusion} \\
\DM{S}_{\text{press}}      & = - \GRAD_h \DM{p},                                       \label{eq:gm_pressure} \\
\DM{S}_{\text{coriolis}}   & = \CoriolisParam \DM{\phi} \CROSS \mathbf{k},             \label{eq:gm_coriolis} \\
\DM{S}_{\text{subsidence}} & = - \SubsidenceParam \GRAD \phi,                          \label{eq:gm_subsidence} \\
\end{align}
```

## Sub-domain equations: Area fraction

The EDMF equations take the form of advection-diffusion equations. The size of these SDs are tracked by solving an equation governing the area fraction in the $i$th SD ($\aSDi{a}$), given by:

```math
\begin{gather}
  \PD_t (\rhoRef{} \aSDi{a})
  + \PD_z (\rhoRef{} \aSDi{a} \SDi{w})
  + \GRAD_h \DOT
  (\rhoRef{} \aSDi{a} \DM{\uH})
  =
  \SDi{S}^a
  , \quad i = 1,2,..., \Nsd{}, \label{eq:AreaFracGov} \\
  \sum_i \aSDi{a} = 1, \label{eq:AreaFracConserve} \\
  \qquad 0 < \aSDi{a} < 1. \label{eq:AreaFracConstraint}
\end{gather}
```

Here, $\rhoRef{}, \SDi{w}, \uH$ is fluid density, mean vertical velocity along $z$, and domain-mean of the horizontal velocity respectively. The area fraction constraints are necessary to ensure the system of equations is well-posed. All source terms ($\SDi{S}^a$) will be discussed in later sections.

!!! note

    The greater than zero constraint must be satisfied at every step of the solution process, since it is necessary to avoid division by zero in the mean field equations.

```math
\begin{align}
\SDi{S}^a = \SDi{S_{\epsilon\delta}}^a.
\end{align}
```

### Source term definitions
We note that the net exchange is zero $\sum_i \SDi{S_{\epsilon\delta}}^a = 0$. Therefore, we may define the environment source term as the negative sum of all updraft source terms. The entrainment-detrainment source is:

```math
\begin{align}
\SDi{S_{\epsilon\delta}}^a =
\begin{cases}
  \rho a_i \SDi{w} \left( -\delta_i + \sum_{j\ne i} \epsilon_{j} \right) & i \ne \iEnv{} \\
  0 - \sum_{j \ne \iEnv{}} \SDj{S_{\epsilon\delta}}^a & i = \iEnv{} \\
\end{cases},
\end{align}
```
where additional variable definitions are in:

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

 - [Entrainment-Detrainment](@ref) ($\epsilon_{ij}$) and ($\delta_i$).

## Sub-domain equations: 1st moment

The 1st moment sub-domain equations are:

```math
\begin{align}\label{eq:1stMoment}
  \PD_t (\rhoRef{} \aSDi{a} \SDi{\phi})
  + \PD_z (\rhoRef{} \aSDi{a} \SDi{w} \SDi{\phi})
  + \GRAD_h \DOT
  (\rhoRef{} \aSDi{a} \DM{\uH} \SDi{\phi})
  =
  \SDi{S}^\phi
  , \quad i = 1,2,..., \Nsd{}. \\
\end{align}
```

Here, $\SDi{S}^{\phi}$ are source terms, including diffusion, and many other sub-grid-scale (SGS) physics. In general, $\SDi{S}^{\phi}$ and $\SDi{S}^{a}$ may depend on $\SDj{\phi}$ and or $\aSDj{a}$ for any $j$.

### Source terms per equation
The source terms common to all unknowns are:

```math
\begin{align}
\SDi{S}^{\phi} =
  \SDi{S_{\epsilon\delta}}^{\phi}
+ \SDi{S_{\text{turb-transp}}}^{\phi}, \quad \forall \phi
\end{align}
```
Additional source terms exist in other equations:
```math
\begin{align}
\SDi{S}^{w} &=
  \SDi{S_{\epsilon\delta}}^w
+ \SDi{S_{\text{turb-transp}}}^w
+ \SDi{S_{\text{buoy}}}
+ \SDi{S_{\text{nh-press}}}
+ \SDi{S_{\text{coriolis}}}, \\
\SDi{S}^{\h} &=
  \SDi{S_{\epsilon\delta}}^{\h}
+ \SDi{S_{\text{turb-transp}}}^{\h}
+ \SDi{S_{\text{MP-MSS}}}^{\h}
+ \SDi{S_{\text{rad}}}, \\
\SDi{S}^{\qt} &=
  \SDi{S_{\epsilon\delta}}^{\qt}
+ \SDi{S_{\text{turb-transp}}}^{\qt}
+ \SDi{S_{\text{MP-MSS}}}^{\qt}.
\end{align}
```

### Source term definitions

Note: The sum of the total pressure and gravity are recast into the sum of the non-hydrostatic pressure and buoyancy sources.

```math
\begin{align}
\SDi{S_{\epsilon\delta}}^{\phi} &=
\begin{cases}
  \rhoRef{} a_i \SDi{w} \left( -\delta_i \SDi{\phi} + \epsilon_{i} \SDj{\phi} \right) & i \ne \iEnv{} \\
  0 - \sum_{j \ne \iEnv{}} \SDj{S_{\epsilon\delta}}^{\phi} & i=\iEnv{} \\
\end{cases} \\
\SDi{S_{\text{turb-transp}}}^{\phi} & =  -\PD_z (\rhoRef{} a_i \IntraCVSDi{w}{w}) \\
 & = \PD_z (\rhoRef{} a_i K_i^m \PD_z \SDi{\phi}) \\
\SDi{S_{\text{nh-press}}} &= -\rhoRef{} \aSDi{a} \left( \alpha_b \SDi{b}  + \alpha_d \frac{(\SDi{w} - \SDe{w}) || \SDi{w} - \SDe{w} || }{r_d \aSDi{a}^{1/2}} \right) \\
\alpha_b &= 1/3, \quad \alpha_d = 0.375, \quad r_d      = 500 [m] \\
\SDi{S_{\text{buoy}}} &= \rhoRef{} \aSDi{a} \SDi{b} \\
\SDi{S_{\text{coriolis}}} & = f(\SDi{\mathbf{u}} - {\SDi{\mathbf{u}_{\text{geo-wind}}}}) \\
\SDi{S_{\text{rad}}}  &= \left( \PD_t {\SDi{\h}} \right)_{radiation} \\
\SDi{S_{\text{grav}}} &= - \rhoRef{} \grav \\
\SDi{S_{\text{MP-MSS}}}^{\qt} & = \\
\SDi{S_{\text{MP-MSS}}}^{\h} & = \\
\end{align}
```

where additional variable definitions are in:

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

 - [Entrainment-Detrainment](@ref) ($\epsilon_{ij}$) and ($\delta_i$).

 - [Buoyancy](@ref) ($\Buoyancy$).

 - [Eddy diffusivity](@ref) ($K_i$).


## Sub-domain equations: 2nd moment

The 2nd moment sub-domain equations are of the exact same form as the 1st moment equations (equation \eqref{eq:1stMoment}):

```math
\begin{align}\label{eq:2ndMoment}
  \PD_t (\rhoRef{} \aSDi{a} \SDi{\phi})
  + \PD_z (\rhoRef{} \aSDi{a} \SDi{w} \SDi{\phi})
  + \GRAD_h \DOT
  (\rhoRef{} \aSDi{a} \DM{\uH} \SDi{\phi})
  =
  \SDi{S}^\phi
  , \quad i = 1,2,..., \Nsd{}. \\
\end{align}
```
Here, $\SDi{S}^{\phi}$ are source terms, including diffusion, and many other sub-grid-scale (SGS) physics. In general, $\SDi{S}^{\phi}$ and $\SDi{S}^{a}$ may depend on $\SDj{\phi}$ and or $\aSDj{a}$ for any $j$.

### Source terms per equation
The source terms common to all unknowns are:

```math
\begin{align}
\SDi{S}^{\phi\psi} =
  \SDi{S_{\epsilon\delta}}^{\phi\psi}
+ \SDi{S_{\text{x-grad flux}}}^{\phi\psi}
+ \SDi{S_{\text{turb-transp}}}^{\phi\psi}
\quad \forall \phi \psi,
\end{align}
```
Additional source terms exist in other equations:

```math
\begin{align}
\SDi{S}^{TKE} &=
  \SDi{S_{\epsilon\delta}}^{TKE}
+ \SDi{S_{\text{x-grad flux}}}^{TKE}
+ \SDi{S_{\text{turb-transp}}}^{TKE}
+ \SDi{S_{\text{dissip}}}
+ \SDi{S_{\text{press}}},
+ \SDi{S_{\text{buoyancy}}}, \\
\SDi{S}^{\phi\psi} &=
  \SDi{S_{\epsilon\delta}}^{\phi\psi}
+ \SDi{S_{\text{x-grad flux}}}^{\phi\psi}
+ \SDi{S_{\text{turb-transp}}}^{\phi\psi}
+ \SDi{S_{\text{MP-MSS}}}^{\phi\psi}.
\quad \phi\psi \in [\qt\qt, \h\h, \h \qt].
\end{align}
```

### Source term definitions

```math
\begin{align}
\SDi{S_{\epsilon\delta}}^{\phi\psi} &=
\begin{cases}
  \rhoRef{} a_i \SDi{w} \left[ -\delta_i \IntraCVSDi{\phi}{\psi} + \sum_{j\ne i}\epsilon_{ij}
\left(
\IntraCVSDj{\phi}{\psi} + (\SDj{\phi} - \SDi{\phi})(\SDj{\psi} - \SDi{\psi})
\right) \right] & i \ne \iEnv \\
  0 - \sum_{j\ne \iEnv} \SDj{S_{\epsilon\delta}}^{\phi\psi} & i=\iEnv \\
\end{cases} \\
\SDi{S_{\epsilon\delta}}^{TKE} &=
\begin{cases}
  \rhoRef{} a_i \SDi{w} \left[ -\delta_i \SDi{TKE} + \sum_{j\ne i}\epsilon_{ij}
\left(
\SDj{TKE} + \frac{1}{2} (\SDj{w} - \SDi{w})^2
\right) \right] & i \ne \iEnv \\
  0 - \sum_{j\ne \iEnv} \SDj{S_{\epsilon\delta}}^{TKE} & i=\iEnv \\
\end{cases} \\
\SDi{S_{\text{x-grad flux}}}^{\phi\psi}
& =
- \rhoRef{} a_i \IntraCVSDi{w}{\psi} \PD_z \SDi{\phi}
- \rhoRef{} a_i \IntraCVSDi{w}{\phi} \PD_z \SDi{\psi} \\
& =
 \rhoRef{} a_i K_i \PD_z \SDi{\psi} \PD_z \SDi{\phi} \\
\SDi{S_{\text{x-grad flux}}}^{TKE}
& =
\rhoRef{} a_i K_i \left[ \left(\PD_z\DM{u}\right)^2 + \left(\PD_z\DM{v}\right)^2 + \left(\PD_z\DM{w}\right)^2 \right] \\
\SDi{S_{\text{turb-transp}}}^{\phi\psi} & = - \PD_z (\rhoRef{} a_i \overline{w'_i\phi'_i\psi'_i}) \\
& = \PD_z (\rhoRef{} a_i K_i \PD_z \IntraCVSDi{\phi}{\psi}) \\
\SDi{S_{\text{turb-transp}}}^{TKE} & = \PD_z (\rhoRef{} a_i K_i \PD_z \SDi{TKE}) \\
\SDi{S_{\text{dissip}}}
& = -c_e \IntraCVSDi{\phi}{\psi} \frac{\SDi{TKE}^{1/2}}{\SDio{{l_{mix}}}}, \quad \text{Equation 38 in Tan et al.} \\
c_e & = 2 \\
\SDi{S_{\text{press}}}
& = - \aSDi{a} \left[ \IntraCVSDi{u}{(\partial_x p^{\dagger})} +
                      \IntraCVSDi{v}{(\partial_y p^{\dagger})} +
                      \IntraCVSDi{w}{(\partial_z p^{\dagger})}\right]  \\
& = 0, \qquad \text{for now, need to derive correct formulation} \\
\SDi{S_{\text{buoyancy}}}^{TKE} & = \rhoRef{} \aSDi{a} \BuoyancyFlux \\
\SDi{S_{\text{MP-MSSP}}}^{\qt\qt}
& = \\
\SDi{S_{\text{MP-MSSP}}}^{\h\h}
& = \\
\end{align}
```
where additional variable definitions are in:

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

 - [Entrainment-Detrainment](@ref) ($\epsilon_{ij}$) and ($\delta_i$).

 - [Eddy diffusivity](@ref) ($K_i$).

 - [Mixing length](@ref) ($l_{mix}$).

 - [Buoyancy flux](@ref) ($\BuoyancyFlux$).

# EDMF variable definitions

The following definitions are ordered in a dependency fashion; all variables are defined from variables already defined in previous subsections.

## Constants

```math
\begin{align}
c_K & = 0.1 \\
\text{tol}_{\InversionHeight\mathrm{-stable}} & = 0.01 \\
\end{align}
```

## Reference state profiles
Reference state profiles are (suppressing $i$ subscript):

```math
\begin{align}
\log{\pRef} = \log{\hyperparam{\BC{\pRef}}} + \frac{-\grav}{\Rd} \int_{z_{min}}^{z} \frac{1}{\text{denom}} dz \\
\text{denom} = \hyperparam{\BC{\DM{T}}} (1 - \hyperparam{\BC{\DM{\qt}}} + \epsvi (\hyperparam{\BC{\DM{\qt}}} - \DM{\ql} - \DM{\qi})) \\
\rhoRef{} = \frac{\pRef{}}{\hyperparam{\BC{T}} \left[ R_d ( 1 + (\MRatio - 1) \hyperparam{\BC{\DM{\qt}}} - \MRatio (\DM{\ql} + \DM{\qi}) ) \right]} \\
\alphaRef = \frac{1}{\rhoRef} \\
\end{align}
```

## Specific heats

```math
\begin{align}
c_{vm} &= (1 - \SDi{\qt}) \Cv{d} + \SDi{\qv} \Cv{v} + \SDi{\ql} \Cv{l} + \SDi{\qi} \Cv{i} \\
c_{pm} &= (1 - \SDi{\qt}) \Cp{d} + \SDi{\qt} \Cp{v} \\
\end{align}
```

## Latent heat

```math
\begin{align}
\LatentHeatV{T} &= \RefLHv + \Cp{v} - \Cp{l} (T - \TTriple) \\
\end{align}
```

## Mixing ratios
```math
\begin{align}\label{eq:MixingRatios}
r_c & = \frac{\SDi{\qt}+\SDi{\ql}}{1 - \SDi{\qt}} \\
r_v & = \frac{\SDi{\qt}-\SDi{\ql}-\SDi{\qi}}{1 - \SDi{\qt}} \\
\end{align}
```

## Shear production

```math
\begin{align}\label{eq:ShearProduction}
|S|^2 &= (\PD_z \DM{u})^2 + (\PD_z \DM{v})^2 + (\PD_z \SDe{w})^2 \\
\end{align}
```

## Saturation adjustment
When assuming phase equilibrium, a non-linear equation must be solved to determine the temperature that satisfies this equilibrium.
This set of equations can be solved using a standard root solver (e.g., Secant method).

The roots of the following system are satisfied when $\SDi{\qt} > \qvsat(T)$, otherwise $\SDi{T} = \TZero + \frac{\hint(T)}{(1-\qt)\Cv{d} + \qt \Cv{v}} + \qt \RefHintV$.

 - Knowns: $\SDi{\qt}, \SDi{\h}, \rhoRef{}$

 - Unknowns: $\SDi{T}$

```math
\begin{align}
\hint(T) &= c_{vm} (T - \TZero)  + \qv \RefHintV - \qi \RefHintI \\
\pvsat(T) &= \PTriple \left( \frac{T}{\TTriple} \right)^{\frac{\DeltaCp}{\Rv}} \exp{\frac{\RefLHv - \DeltaCp \TZero}{\Rv} \left( \frac{1}{\TTriple} - \frac{1}{T} \right)} \\
\qvsat &= \frac{\pvsat(T)}{\rho \Rv T} \\
\qc &= \max(\qt - \qvsat, 0) \\
\qv &= \qt - \ql - \qi \\
\ql &= \lambda \qc \\
\qi &= (1-\lambda) \qc \\
\lambda(T) &= \mathcal{H}(T-\TFreeze), \quad \mathcal{H} = \text{heaviside function} \\
\end{align}
```
where additional variable definitions are in:

 - [Specific heats](@ref) $c_{pm}$ and $c_{vm}$.

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

This set of equations are only necessary when converting from a ($\ThetaL,\qt$) formulation to our ($\hint,\qt$) formulation. The roots of the following system are satisfied when $\SDi{\qt} > \qvsat(T)$, otherwise $\SDi{\ql} = 0,$ and $\SDi{T} = \SDi{\ThetaL} \left(\frac{\pRef}{\PTilde}\right)^{\frac{\Rd}{\Cp{d}}}$.

 - Knowns: $\SDi{\qt}, \SDi{\ThetaL}, \pRef$

 - Unknowns: $\SDi{T}$

```math
\begin{align}
\SDi{\ThetaL} = \SDi{T} \left(\frac{\PTilde}{\pRef}\right)^{\frac{R_m}{c_{pm}}}
\left(1 - \frac{ \RefLHv \SDi{\ql} + \RefLHs \SDi{\qi}}{c_{pm} \SDi{T}} \right) \\
\end{align}
```
where additional variable definitions are in:

 - [Specific heats](@ref) $c_{pm}$ and $c_{vm}$.

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

## Buoyancy
```math
\begin{align}\label{eq:Buoyancy}
\SDi{b}^{\dagger} = \grav (\SDi{\alpha} - \alphaRef)/\alphaRef \\
\SDi{b} = \SDi{b}^{\dagger} - \sum_j a_j \SDj{b}^{\dagger} \\
\end{align}
```
where additional variable definitions are in:

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

## Potential temperatures
Fix: which virtual potential temperature is used
```math
\begin{align}\label{eq:Theta}
\SDi{\ThetaDry} &= \SDi{T}(\pRef{}/\PTilde{})^{-\Rd/\Cp{d}} \\
\SDi{\ThetaVirt} & = \SDi{\ThetaDry} (1 - r_c + 0.61 r_v) \\
\SDi{\ThetaVirt} &= \SDi{\theta} \left(1 + 0.61 \SDi{q_r} - \SDi{\ql} \right) \\
\SDi{\ThetaRho} &= \frac{\SDi{T} (1 - \SDi{\qt} + \epsvi \qv)}{(\pRef/\PTilde)^{\param{\kappa}}} \\
\end{align}
```
where additional variable definitions are in:

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

 - [Mixing ratios](@ref) ($r_c$, $r_v$).

## Buoyancy gradient

Short buoyancy gradient
```math
\begin{align}\label{eq:BuoyancyGrad}
\BuoyancyGrad & = - \frac{\grav}{\DM{\ThetaVirt}} \PD_z \SDi{\ThetaVirt} \\
\end{align}
```

Long buoyancy gradient
```math
\begin{align}\label{eq:BuoyancyGradLong}
\BuoyancyGrad & = - \PD_z \SDi{\ThetaL}
\left[ (1-f_c) \PD_{\ThetaL} b |_d  + f_c \PD_{\ThetaL} b |_s \right] -
\PD_z \SDi{\qt}      \left[ (1-f_c) \PD_{\qt} b |_d + f_c \PD_{\qt} b |_s \right] \\
f_c &= 0 \qquad \text{good for simple cases, need to confirm for more complex cases} \\
\PD_{\ThetaL} b |_d & = \frac{\grav}{\DM{\ThetaVirt}} \left[ 1 + \left( \frac{\Rv}{\Rd} - 1 \right) \SDi{\qt} \right] \\
\PD_{\ThetaL} b |_s &= \frac{\grav}{\DM{\ThetaVirt}} \left[ 1 + \frac{\Rv}{\Rd} \left(1 + \frac{\LatentHeatV{\SDi{T}}}{\Rv \SDi{T}} \right) \SDi{q_s} - \SDi{\qt} \right] \left( 1 + \frac{{\LatentHeatV{\SDi{T}}}^2}{\param{c_p} \Rv \SDi{T}^2} \SDi{q_s} \right)^{-1} \\
\PD_{\qt} b |_d &= \frac{\grav}{\DM{\ThetaVirt}} \left( \frac{\Rv}{\Rd} - 1 \right) \SDi{\theta} \\
\PD_{\qt} b |_s &= \left( \frac{{\LatentHeatV{\SDi{T}}}}{\param{c_p} \SDi{T}} \PD_{\ThetaL} b |_s - \frac{\grav}{\DM{\ThetaVirt}} \right) \SDi{\theta} \\
\SDi{\theta} &= \SDi{T}(\pRef{}/\PTilde{})^{-\Rd/\Cp{d}} \\
\SDi{q_r} & = q_{rain} = 0 \qquad \text{for now} \\
\SDi{q_s} & = \qvsat \\
\end{align}
```
where additional variable definitions are in:

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

 - [Potential temperatures](@ref) ($\ThetaDry$, $\ThetaVirt$).

 - [Latent heat](@ref) ($\LatentHeatV{T}$).

## Surface fluxes

Variables in this section must be computed simultaneously because it requires the solution of a non-linear equation.

### Monin-Obhukov length
NOTE: All variables (Monin-Obhukov length, friction velocity, temperature scale) in [Surface fluxes](@ref) must be solved simultaneously
```math
\begin{align}\label{eq:MOLen}
\MOLen = \begin{cases}
- \frac{\FrictionVelocity^3 \theta}{\VKConst \grav \SurfaceHeatFlux} & \SurfaceHeatFlux > 0 \\
0 & \text{otherwise} \\
\end{cases} \\
\end{align}
```

### Friction velocity
NOTE: All variables (Monin-Obhukov length, friction velocity, temperature scale) in [Surface fluxes](@ref) must be solved simultaneously

 - Knowns: $u_{\mathrm{ave}} = \sqrt{\DM{u}^2+\DM{v}^2}, \LayerThickness, \SurfaceRoughness{m}$

 - Unknowns: $\FrictionVelocity, \MOLen$, and $\SurfaceMomentumFlux$

```math
\begin{align}\label{eq:FrictionVelocity}
u_{\mathrm{ave}}     & = \frac{\FrictionVelocity}{\VKConst}    \left[ \log\left(\frac{\LayerThickness}{\SurfaceRoughness{m}}\right) - \Psi_m\left(\frac{\LayerThickness}{\MOLen}\right) + \frac{\SurfaceRoughness{m}}{\LayerThickness} \Psi_m\left(\frac{\SurfaceRoughness{m}}{\MOLen}\right) + R_{z0m} \left\{ \psi_m\left(\frac{\SurfaceRoughness{m}}{\MOLen}\right) - 1 \right\} \right] \\
R_{z0m}              & = 1 - \SurfaceRoughness{h}/\LayerThickness \\
\SurfaceMomentumFlux & = -\FrictionVelocity^2                , \label{eq:SurfaceMomentumFlux}  \\
\end{align}
```
where $\Psi_m$ is defined in Appendix A, equations A6 in Nishizawa, S., and Y. Kitamura. "A Surface Flux Scheme Based on the Monin‐Obukhov Similarity for Finite Volume Models." Journal of Advances in Modeling Earth Systems 10.12 (2018): 3159-3175.

### Temperature scale
NOTE: All variables (Monin-Obhukov length, friction velocity, temperature scale) in [Surface fluxes](@ref) must be solved simultaneously

 - Knowns: $\theta_{\mathrm{ave}}, \theta_s, \LayerThickness, \SurfaceRoughness{h}$

 - Unknowns: $\FrictionVelocity, \MOLen$, and $\SurfaceHeatFlux$
```math
\begin{align}\label{eq:TemperatureScale}
\theta_{\mathrm{ave}} - \theta_s & = \frac{Pr \TemperatureScale}{\VKConst} \left[ \log\left(\frac{\LayerThickness}{\SurfaceRoughness{h}}\right) - \Psi_h\left(\frac{\LayerThickness}{\MOLen}\right) + \frac{\SurfaceRoughness{h}}{\LayerThickness} \Psi_m\left(\frac{\SurfaceRoughness{h}}{\MOLen}\right) + R_{z0h} \left\{ \psi_h\left(\frac{\SurfaceRoughness{h}}{\MOLen}\right) - 1 \right\} \right] \\
R_{z0h}                          & = 1 - \SurfaceRoughness{h}/\LayerThickness \\
\SurfaceHeatFlux                 & = -\FrictionVelocity\TemperatureScale , \label{eq:SurfaceHeatFlux}  \\
\end{align}
```
where $\Psi_h$ is defined in Appendix A, equation A6 in Nishizawa, S., and Y. Kitamura. "A Surface Flux Scheme Based on the Monin‐Obukhov Similarity for Finite Volume Models." Journal of Advances in Modeling Earth Systems 10.12 (2018): 3159-3175.

## Prandtl number

```math
\begin{align}\label{eq:PrandtlNumber}
Pr_{neut} &= 0.74 \\
Pr(z) &= \begin{cases}
    Pr_{neut} & \MOLen < 0 \\
    Pr_{neut} \left[ \frac{1 + \omega_2 R_g - \sqrt{-4 R_g + (1+\omega_2 R_g)^2}}{2 R_g} \right] & \text{otherwise} \\
\end{cases} \\
\omega_2 &= \omega_1+1 \\
\omega_1 &= \frac{40}{13} \\
R_g &= \frac{\BuoyancyGrad}{|S|^2} \\
\end{align}
```
where additional variable definitions are in:

 - [Shear production](@ref) ($S$).

 - [Monin-Obhukov length](@ref) ($\MOLen$).

 - [Buoyancy gradient](@ref) ($\BuoyancyGrad$).

## Mixing length

!!! note

    These mixing length have been tested for the environment, not the updrafts

```math
\begin{align}\label{eq:MixingLength}
\SDio{{l_{mix}^m,}} &= \frac{\sum_j l_j e^{-l_j}}{\sum_j e^{-l_j}}, \qquad j = 1,2,3 \\
l_1 &= \frac{\sqrt{c_w\SDe{TKE}}}{\SDe{N}} \\
\SDe{N} &= \frac{\grav \PD_z \SDe{\ThetaVirt}}{\SDe{\ThetaVirt}} , \qquad \text{(buoyancy frequency of environment)} \\
l_2 &= \frac{\VKConst z}{c_K \kappa^* \phi_m(\xi)} \\
\xi &= z/\MOLen \\
\phi_m(\xi) &= \left( 1 + a_l \frac{z}{\MOLen} \right)^{-b_l} \\
(a_l, b_l) &=
\begin{cases}
  (-100, 0.2) & \MOLen < 0 \\
  (2.7, -1) & \text{otherwise} \\
\end{cases} \\
\kappa^* &= \frac{\FrictionVelocity}{\sqrt{\SDe{TKE}}} \\
l_3 &= \sqrt{\frac{c_{\varepsilon}}{c_K}} \sqrt{\SDe{TKE}}
\left[ |S|^2 - \frac{1}{Pr(z)} \BuoyancyGrad \right]^{-1/2} \\
\end{align}
```
where additional variable definitions are in:

 - [Constants](@ref).

 - [Shear production](@ref) ($S$).

 - [Monin-Obhukov length](@ref) ($\MOLen$).

 - [Friction velocity](@ref) ($\FrictionVelocity$).

 - [Buoyancy gradient](@ref) ($\BuoyancyGrad$).

 - [Potential temperatures](@ref) ($\ThetaDry$, $\ThetaVirt$).

 - [Prandtl number](@ref) ($Pr$).

Smoothing function is provided in python file. The Prandtl number was used from Eq. 75 in Dan Li 2019 "Turbulent Prandtl number in the atmospheric BL - where are we now".

## Eddy diffusivity

```math
\begin{align}\label{eq:EddyDiffusivity}
\SDi{K_m} & = \begin{cases}
c_K \SDio{{l_{mix},}} \sqrt{\SDi{TKE}} & i = \iEnv \\
0 & \text{otherwise}
\end{cases} \\
\SDi{K_h} & = \frac{\SDi{K_m}}{Pr} \\
\end{align}
```
where additional variable definitions are in:

 - [Constants](@ref).

 - [Mixing length](@ref) ($l_{mix}$).

 - [Prandtl number](@ref) ($Pr$).

## Buoyancy flux

!!! todo

    Currently, $\BuoyancyFlux$ is hard-coded from the first expression (which was used in SCAMPy), however, this value should be computed from the SurfaceFluxes section.

```math
\begin{align}\label{eq:BuoyancyFlux}
\SurfaceBuoyancyFlux & = \frac{\grav \BC{\alphaRef}}{c_{pm} \BC{\SDi{T}}} (\SensibleSurfaceHeatFlux + (\epsvi - 1) c_{pm} \BC{\SDi{T}} \LatentSurfaceHeatFlux / \LatentHeatV{\BC{\SDi{T}}}) \\
\BuoyancyFlux & = - K_i \BuoyancyGrad \\
\end{align}
```
 - [Eddy diffusivity](@ref) ($K_i$).

 - [Latent heat](@ref) ($\LatentHeatV{T}$).

 - [Buoyancy gradient](@ref) ($\BuoyancyGrad$).

## Entrainment-Detrainment

Entrainment ($\epsilon_{ij}$)
```math
\begin{align}\label{eq:Entrainment}
\epsilon_{ij} &= c_{\epsilon} \frac{\max(\SDi{b}, 0)}{\SDi{w}^2} \\
c_{\epsilon} &= 0.12 \\
\alpha_i &= \frac{\Rd \SDi{T}}{\pRef} (1 - \SDi{\qt} + \epsvi \SDi{\qv}) \\
\SDi{\qv} &= \SDi{\qt} - \SDi{\ql} - \SDi{\qi} \\
\SDi{\qi} &= 0, \quad \text{may change later} \\
\end{align}
```

Detrainment ($\delta_{j}$):
```math
\begin{align}\label{eq:Detrainment}
\delta_{i} &= c_{\delta} \frac{|\min(\SDi{b}, 0)|}{\SDi{w}^2} + \delta_{B} \mathcal{H}(\SDi{\ql}) \\
c_{\delta} &= c_{\delta,0} + \Gamma(\aSDi{a}) \\
\Gamma(\aSDi{a}) &= 0 \\
c_{\delta,0} &= c_{\epsilon} = 0.12 \\
\delta_B &= 0.004 [m^{-1}] \\
\mathcal{H} &= \text{Heaviside function} \\
\end{align}
```
where additional variable definitions are in:

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

 - [Saturation adjustment](@ref) Temperature ($\SDi{T}$) and specific humidity of liquid ($\SDi{\ql}$).

 - [Buoyancy](@ref) ($\Buoyancy$).

## Inversion height

```math
\begin{align}\label{eq:InversionHeight}
\SDio{\InversionHeight} &=
\begin{cases}
  \left[ (\PD_z \theta_{\rho})^{-1} (\BC{\theta_{\rho}} - \theta_{\rho}|_{z_1}) + z_1 \right] & \simparam{\BC{\DM{u}}}^2 + \simparam{\BC{\DM{v}}}^2 <= \text{tol}_{\InversionHeight\mathrm{-stable}} \\
  \left[ (\PD_z Ri_{bulk})^{-1} (\hyperparam{Ri_{bulk, crit}} - Ri_{bulk}|_{z_2}) + z_2 \right] & \text{otherwise} \\
\end{cases} \\
z_1 &= \min_z (\theta_{\rho}(z) > \BC{\theta_{\rho}}) \\
z_2 &= \min_z (Ri_{bulk}(z) > \hyperparam{Ri_{bulk, crit}}) \\
Ri_{bulk} &= \grav z \frac{(\theta_{\rho}/\BC{\theta_{\rho}} - 1)}{\simparam{\DM{u}}^2 + \simparam{\DM{v}}^2} \\
\end{align}
```
where additional variable definitions are in:

 - [Potential temperatures](@ref) ($\theta$).

## Convective velocity

```math
\begin{align}\label{eq:ConvectiveVelocity}
\SDio{\ConvectiveVelocity} &= (\max(\BuoyancyFlux \SDio{\InversionHeight}, 0))^{1/3} \\
\end{align}
```
where additional variable definitions are in:

 - [Inversion height](@ref) ($\SDio{\InversionHeight}$).

 - [Buoyancy flux](@ref) ($\BuoyancyFlux$).

## Non-local mixing length

```math
\begin{align}\label{eq:MixingLengthOld}
\SDio{{l_{mix},}} &= (l_A^{-1} + l_B^{-1})^{-1} \\
l_A &= \VKConst z \left( 1 + a_l \frac{z}{\MOLen} \right)^{b_l} \\
\SDio{{l_B}} &= \SDio{\tau} \SDi{TKE} \\
(a_l, b_l) &=
\begin{cases}
  (-100, 0.2) & \MOLen < 0 \\
  (2.7, -1) & \text{otherwise} \\
\end{cases} \\
\SDio{\tau} &= \SDio{\InversionHeight}/\SDio{\ConvectiveVelocity} \\
\end{align}
```
where additional variable definitions are in:

 - [Inversion height](@ref) ($\SDio{\InversionHeight}$).

 - [Monin-Obhukov length](@ref) ($\MOLen$).

 - [Convective velocity](@ref) ($\SDio{\ConvectiveVelocity}$).

# Boundary Conditions

Here, we specify boundary conditions (BCs) by their type, Dirichlet (D) or Neumann (N), and their value.

## BC functions

```math
\begin{align}
\Gamma_{\phi}(F_1, F_2)
& = \begin{cases}
    4 \frac{F_1 F_2}{\FrictionVelocity^2} (1 - 8.3\zLL/\MOLen)^{-2/3} & \MOLen < 0 \\
    4 \frac{F_1 F_2}{\FrictionVelocity^2} & \text{otherwise}
\end{cases} \\
\Gamma_{TKE}
& = \begin{cases}
    3.75 {\FrictionVelocity}^2 + 0.2 {\ConvectiveVelocity}^2 + {\FrictionVelocity}^2 (-\zLL/\MOLen)^{2/3} & \MOLen < 0 \\
    3.75 {\FrictionVelocity}^2 & \text{otherwise}
\end{cases} \\
\SensibleSurfaceHeatFlux & = \BC{\TCV{w}{\hint}} c_{pm} \rhoRef \\
\LatentSurfaceHeatFlux   & = \BC{\TCV{w}{\qt}}  \LatentHeatV{T} \rhoRef \\
F_{\hint}(\SensibleSurfaceHeatFlux)  & = \frac{\SensibleSurfaceHeatFlux}{c_{pm}}       = \BC{\TCV{w}{\hint}} \rhoRef \\
F_{\qt}(\LatentSurfaceHeatFlux)      & = \frac{\LatentSurfaceHeatFlux}{\LatentHeatV{T}} = \BC{\TCV{w}{\qt}}   \rhoRef \\
\end{align}
```
where additional variable definitions are in:

 - [Reference state profiles](@ref) ($\pRef{}$, $\rhoRef{}$, and $\alphaRef{}$).

 - [Monin-Obhukov length](@ref) ($\MOLen$).

 - [Convective velocity](@ref) ($\SDio{\ConvectiveVelocity}$).

 - [Friction velocity](@ref) ($\FrictionVelocity$).

 - [Latent heat](@ref) ($\LatentHeatV{T}$).

and equation \eqref{eq:TopPercentile} represents the mean of the top $x$-fraction of a standard normal distribution (Neggers et al., 2009).

```math
\begin{align}
\Phi^{-1}(x)  &= \text{inverse cumulative distribution function}, \label{eq:InverseCDF} \\
\mathcal D(x) &= \frac{1}{\sqrt{2\pi x}} \exp{- \frac{1}{2} (\Phi^{-1}(1-x))^2 } , \label{eq:TopPercentile} \\
\end{align}
```

## Area fraction

```math
\begin{align}
c_{frac} = 0.1, \quad
\BCB{\aSDi{a}} =
\begin{cases}
    1-c_{frac}, & i = \iEnv{} \\
  \frac{c_{frac}}{\Nsd}, & i \ne \iEnv{}
\end{cases}, \quad
\BCT{\aSDi{a}} =
\begin{cases}
    1-c_{frac}, & i = \iEnv{} \\
  \frac{c_{frac}}{\Nsd}, & i \ne \iEnv{}
\end{cases}
\end{align}
```

## 1st order moments

Top boundary
```math
\begin{align}
\BCT{\SDi{w}}           &= 0 \\
\PD_z \BCT{\SDi{\qt}}   &= 0 \\
\PD_z \BCT{\SDi{\hint}} &= 0 \\
\end{align}
```
Bottom boundary
```math
\begin{align}
\BCB{\SDi{w}}     &= 0 \\
- \SDi{K_m} \PD_z \BCB{\SDi{\qt}}   &= \TCV{w}{\qt}   + \mathcal D(\aSDi{a}) \sqrt{C_{\qt}  \WindSpeed^2\Gamma_{\phi}(\TCV{w}{\qt}  , \TCV{w}{\qt}   )} \\
- \SDi{K_m} \PD_z \BCB{\SDi{\hint}} &= \TCV{w}{\hint} + \mathcal D(\aSDi{a}) \sqrt{C_{\hint}\WindSpeed^2\Gamma_{\phi}(\TCV{w}{\hint}, \TCV{w}{\hint} )} \\
\end{align}
```
where additional variable/function definitions are in:

 - [BC functions](@ref) $\mathcal D$.

## 2nd order moments

Top boundary
```math
\begin{align}
\BCT{\SDi{TKE}}                         & = 0 \\
\PD_z \BCT{\IntraCVSDi{\qt}{\qt}}       & = 0 \\
\PD_z \BCT{\IntraCVSDi{\hint}{\hint}}   & = 0 \\
\PD_z \BCT{\IntraCVSDi{\hint}{\qt}}     & = 0 \\
\end{align}
```

!!! todo

    Currently, we only account for the _intra_ sub-domain covariance, but we would like to also account for the _inter_ sub-domain covariance for all but the $TKE$.

Bottom boundary
```math
\begin{align}
\BCB{\SDi{TKE}}                   & = \Gamma_{TKE} \\
\BCB{\IntraCVSDi{\qt}{\qt}}       & = \Gamma_{\phi}(\TCV{w}{\qt}  , \TCV{w}{\qt}   ) \\
\BCB{\IntraCVSDi{\hint}{\hint}}   & = \Gamma_{\phi}(\TCV{w}{\qt}  , \TCV{w}{\hint} ) \\
\BCB{\IntraCVSDi{\hint}{\qt}}     & = \Gamma_{\phi}(\TCV{w}{\hint}, \TCV{w}{\hint} ) \\
\end{align}
```
where additional variable/function definitions are in:

 - [BC functions](@ref) $\Gamma_{TKE}$, $\Gamma_{\phi}$, $F_{\hint}$, $\SensibleSurfaceHeatFlux$, $F_{\qt}$, $\LatentSurfaceHeatFlux$.

## Case-specific configurations

| **Case** | **Variable**            | **Value**                        | **Reference**    |
|:---------|-------------------------|----------------------------------|------------------|
| Bomex    | $\BC{p_s}$              | 1000 [hPa]                       |                  |
| Bomex    | $\BC{\DM{\qt}}$         | 5 [g/kg]                         |                  |
| Bomex    | $\BC{\DM{\ThetaL}}$     | 300 [K]                          |                  |
| Bomex    | $\BC{\TCV{w}{\qt}}$     | $5.2 \times 10^{-5} [m s^{-1}]$  |                  |
| Bomex    | $\BC{\TCV{w}{\ThetaL}}$ | $8 \times 10^{-3} [K m s^{-1}]$  |                  |
| Soares   | $\BC{\TCV{w}{\qt}}$     | $2.5 \times 10^{-5} [m s^{-1}]$  |                  |
| Soares   | $\BC{\TCV{w}{\ThetaL}}$ | $6 \times 10^{-2} [K m s^{-1}]$  |                  |
|          |                         |                                  |                  |

