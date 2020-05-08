# Microphysics Module

The `Microphysics` module describes warm rain bulk parameterization of
cloud microphysical processes. The module describes the warm rain (no ice
and snow) formation and is based on the ideas of Kessler 1995.

Parameterized processes include:
  - rain sedimentation with mass weighted average terminal velocity,
  - condensation/evaporation of cloud water,
  - autoconversion,
  - accretion,
  - evaporation of rain water.

The cloud microphysics variables are expressed as specific humidities:
  - q_tot - total water specific humidity,
  - q_vap - water vapor specific humidity,
  - q_liq - liquid water specific humidity,
  - q_rai - rain water specific humidity.

Parameters used in the parameterization are defined in
`MicrophysicsParameters` module. They consist of:

|    symbol            |         definition                                        | units                    | default value         |
|----------------------|-----------------------------------------------------------|--------------------------|-----------------------|
|``n_{0_{MP}}``        | rain drop size distribution parameter                     | ``\frac{1}{m^4}``        | ``16 \cdot 10^6``     |
|``\tau_{cond_evap}``  | cloud water condensation/evaporation timescale            | ``s``                    | ``10``                |
|``\tau_{acnv}``       | cloud to rain water autoconversion timescale              | ``s``                    | ``10^3``              |
|``q_{liq\_threshold}``| cloud to rain water autoconversion threshold              | -                        | ``5 \cdot 10^{-4}``   |
|``E_{col}``           | collision efficiency between rain drops and cloud droplets| -                        | ``0.8``               |
|``C_{drag}``          | rain drop drag coefficient                                | -                        | ``0.55``              |
|``a_{vent}, b_{vent}``| rain drop ventilation factor coefficients                 | -                        | ``1.5 \;``,``\; 0.53``|
|``K_{therm}``         | thermal conductivity of air                               | ``\frac{J}{m \; s \; K}``| ``2.4 \cdot 10^{-2}`` |
|``\nu_{air}``         | kinematic viscosity of air                                | ``\frac{m^2}{s}``        | ``1.6 \cdot 10^{-5}`` |
|``D_{vapor}``         | diffusivity of water vapor                                | ``\frac{m^2}{s}``        | ``2.26 \cdot 10^{-5}``|

## Rain drop size distribution

The rain-drop size distribution is assumed to follow Marshall-Palmer
distribution (Marshall Palmer 1948 eq. 1):
```math
\begin{equation}
n(r) = n_{0_{MP}} exp\left(- \lambda_{MP} \, r \right)
\end{equation}
```
where:
 - ``r`` is the drop radius,
 - ``n_{0_{MP}}`` and ``\lambda_{MP}`` are the Marshall-Palmer distribution
 parameters (twice the values used in the Marshall Palmer 1948, because
 we use drop radius and not diameter).

## Terminal velocity

The terminal velocity of an individual rain drop is defined by the balance
between the gravitational acceleration (taking into account the density
difference between water and air) and the drag force:

```math
\begin{equation}
v_{drop} = \left(\frac{8}{3 \, C_{drag}} \left( \frac{\rho_{water}}{\rho} -1 \right) \right)^{1/2} (g \, r)^{1/2} = v_c(\rho) \, (g \, r)^{1/2}
\label{eq:vdrop}
\end{equation}
```
where:
 - ``g`` is the gravitational acceleration,
 - ``C_{drag}`` is the drag coefficient,
 - ``\rho_{water}`` is the density of water,
 - ``\rho`` is the density of air.

The mass weighted terminal velocity ``v_t`` is defined following Ogura
and Takahashi 1971
```math
\begin{equation}
v_t = \frac{F_{rain}}{RWC}
\label{eq:vt}
\end{equation}
```
where:
 - ``F_{rain} = \int_0^\infty n(r) \, m(r) \, v_{drop}(r) \, dr`` is the
 vertical flux of rain drops,
 - ``RWC = \int_0^\infty n(r) \, m(r) \, dr = \rho \, q_{rai}`` is the
 rain water content.

Integrating over the assumed Marshall-Palmer distribution results in
```math
\begin{equation}
RWC = \frac{8 \pi \, n_{0_{MP}} \, \rho_{water}}{\lambda_{MP}^4}
\label{eq:lambda}
\end{equation}
```
```math
\begin{equation}
F_{rain} = \Gamma \left(\frac{9}{2} \right) \frac{4}{3} n_{0_{MP}} \, \pi \, \rho_{water} v_c(\rho) \, g^{1/2} (\lambda_{MP})^{-9/2}
\label{eq:frain}
\end{equation}
```
Substituting eq.(\ref{eq:lambda}) and eq.(\ref{eq:frain}) into
eq.(\ref{eq:vt}) results in:
```math
\begin{equation}
v_t = \Gamma \left( \frac{9}{2} \right) \, \frac{v_c(\rho)}{6} \, \left( \frac{g}{\lambda_{MP}}\right)^{1/2}
\end{equation}
```
where ``\lambda_{MP}`` is computed as
```math
\lambda_{MP} = \left( \frac{8 \pi \rho_{water} n_{0_{MP}}}{\rho q_{rai}} \right)^{1/4}
```
The default value of ``C_{drag}`` is chosen such that the ``v_t`` is
close to the empirical terminal velocity formulation in Smolarkiewicz
and Grabowski 1996. Assuming a constant drag coefficient is
an approximation as it should be size and flow dependent, see
[drag_coefficient](https://www.grc.nasa.gov/www/K-12/airplane/dragsphere.html).

```@example rain_terminal_velocity
using ClimateMachine.Microphysics
using Plots
using CLIMAParameters
using CLIMAParameters.Atmos.Microphysics
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# eq. 5d in Smolarkiewicz and Grabowski 1996
# https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
function terminal_velocity_empirical(q_rai::DT, q_tot::DT, ρ::DT, ρ_air_ground::DT) where {DT<:Real}
    rr  = q_rai / (DT(1) - q_tot)
    vel = DT(14.34) * ρ_air_ground^DT(0.5) * ρ^-DT(0.3654) * rr^DT(0.1346)
    return vel
end

q_rain_range = range(1e-8, stop=5e-3, length=100)
ρ_air, q_tot, ρ_air_ground = 1.2, 20 * 1e-3, 1.22

plot(q_rain_range * 1e3,  [terminal_velocity(param_set, q_rai, ρ_air) for q_rai in q_rain_range], xlabel="q_rain [g/kg]", ylabel="velocity [m/s]", title="Average terminal velocity of rain", label="ClimateMachine")
plot!(q_rain_range * 1e3, [terminal_velocity_empirical(q_rai, q_tot, ρ_air, ρ_air_ground) for q_rai in q_rain_range], label="Empirical")
savefig("rain_terminal_velocity.svg") # hide
nothing # hide
```
![](rain_terminal_velocity.svg)

## Cloud condensation/evaporation

Condensation and evaporation of cloud water is parameterized as a relaxation
to equilibrium value at the current time step.
```math
\begin{equation}
  \left. \frac{d \, q_{liq}}{dt} \right|_{cond, evap} = \frac{q^{eq}_{liq} - q_{liq}}{\tau_{cond\_evap}}
\end{equation}
```
where:
 - ``q^{eq}_{liq}`` - liquid water specific humidity in equilibrium,
 - ``q_{liq}`` - liquid water specific humidity,
 - ``\tau_{cond\_evap}`` - relaxation timescale (parameter in
 `MicrophysicsParameters` module).

## Autoconversion

Autoconversion defines the rate of conversion form cloud to rain water
due to collisions between cloud droplets. It is parameterized following
Kessler 1995:
```math
\begin{equation}
  \left. \frac{d \, q_{rai}}{dt} \right|_{acnv} = \frac{max(0, q_{liq} - q_{liq\_threshold})}{\tau_{acnv}}
\end{equation}
```
where:
 - ``q_{liq}`` - liquid water specific humidity,
 - ``\tau_{acnv}`` - timescale (parameter in `MicrophysicsParameters`
 module),
 - ``q_{liq\_threshold}`` - autoconversion (parameter in
 `MicrophysicsParameters` module).

The default values of ``\tau_{acnv}`` and ``q_{liq\_threshold}`` are based
on Smolarkiewicz and Grabowski 1996.

## Accretion

Accretion defines the rate of conversion from cloud to rain water resulting
from collisions between cloud droplets and rain drops. It is parameterized
following Kessler 1995:
```math
\begin{equation}
\left. \frac{d \, q_{rai}}{dt} \right|_{accr} = \int_0^\infty n(r) \, \pi r^2 \, v_{drop} E_{col} q_{liq} dr
\end{equation}
```
where:
 - ``E_{col}`` is the collision efficiency,
 - ``v_{drop}`` is defined in eq.(\ref{eq:vdrop}).

Integrating over the distribution and using the RWC to eliminate the ``\lambda_{MP}`` results in:
```math
\begin{equation}
\left. \frac{d \, q_{rai}}{dt} \right|_{accr}  = \Gamma \left(\frac{7}{2} \right) \pi^{1/8} 8^{-7/8} E_{col} v_c(\rho) \, \left(\frac{\rho}{\rho_{water}}\right)^{7/8} n_{0_{MP}}^{1/8} g^{1/2} q_{liq} q_{rai}^{7/8} = A(\rho) \, n_{0_{MP}}^{1/8} g^{1/2} q_{liq} q_{rai}^{7/8}
\end{equation}
```
The default value of collision efficiency ``E_{coll}`` is set to
0.8 so that the resulting accretion rate is close to the empirical
accretion rate in Smolarkiewicz and Grabowski 1996. Assuming a
constant ``E_{col}`` is an approximation, see for example [collision
efficiency](https://journals.ametsoc.org/doi/10.1175/1520-0469%282001%29058%3C0742%3ACEODIA%3E2.0.CO%3B2).

```@example accretion
using ClimateMachine.Microphysics
using Plots
using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# eq. 5b in Smolarkiewicz and Grabowski 1996
# https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
function accretion_empirical(q_rai::DT, q_liq::DT, q_tot::DT) where {DT<:Real}
    rr  = q_rai / (DT(1) - q_tot)
    rl  = q_liq / (DT(1) - q_tot)
    return DT(2.2) * rl * rr^DT(7/8)
end

# some example values
q_rain_range = range(1e-8, stop=5e-3, length=100)
ρ_air, q_liq, q_tot = 1.2, 5e-4, 20e-3

plot(q_rain_range * 1e3,  [conv_q_liq_to_q_rai_accr(param_set, q_liq, q_rai, ρ_air) for q_rai in q_rain_range], xlabel="q_rain [g/kg]", ylabel="accretion rate [1/s]", title="Accretion", label="ClimateMachine")
plot!(q_rain_range * 1e3, [accretion_empirical(q_rai, q_liq, q_tot) for q_rai in q_rain_range], label="empirical")
savefig("accretion_rate.svg") # hide
nothing # hide
```
![](accretion_rate.svg)


## Rain evaporation

Based on Maxwell 1971 the equation of growth of individual water drop is:
```math
\begin{equation}

r \frac{dr}{dt} = \frac{1}{\rho_{water}}
                  \left(\frac{q_{vap}}{q_{vap}^{sat}} - 1 \right)
                  \left(
                    \frac{L}{KT} \left(\frac{L}{R_v T} - 1 \right) +
                    \frac{R_v T}{p_{vap}^{sat} D}
                  \right)^{-1}
                = \frac{1}{\rho_{water}} S(q_{vap}, q_{vap}^{sat}) \, G(T)
\end{equation}
```
where:
 - ``q_{vap}^{sat}`` is the saturation vapor specific humidity,
 - ``L`` is the latent heat of vaporization,
 - ``K_{thermo}`` is the thermal conductivity of air,
 - ``R_v`` is the gas constant of water vapor,
 - ``D_{vapor}`` is the diffusivity of water vapor,
 - ``S(q_{vap}, q_{vap}^{sat}) = \frac{q_{vap}}{q_{vap}^{sat}} - 1 ``
 is commonly labeled as supersaturation,
 - ``G(T) = \left(\frac{L}{KT} \left(\frac{L}{R_v T} - 1 \right) + \frac{R_v T}{p_{vap}^{sat} D} \right)^{-1}``
 combines the effects of thermal conductivity and water diffusivity.

The rate of ``q_{rai}`` evaporation is:
```math
\begin{equation}
\left. \frac{d \, q_{rai}}{dt} \right|_{evap}  =  \int_0^\infty \frac{1}{\rho} \, 4 \pi \, r \, S(q_{vap}, q_{vap}^{sat}) \, G(T) \, F(r) \, n(r) \, dr
\end{equation}
```
where:
 - ``F(r)`` is the rain drop ventilation factor.

Following Seifert and Beheng 2006 eq. 24 the ventilation factor is
defined as:
```math
\begin{equation}
F(r) = a_{vent} + b_{vent}  N_{Sc}^{1/3} N_{Re}(r)^{1/2}
\end{equation}
```
where:
 - ``a_{vent}``, ``b_{vent}`` are coefficients,
 - ``N_{Sc}`` is the Schmidt number,
 - ``N_{Re}`` is the Reynolds number of a falling rain drop.
The Schmidt number is assumed constant:
```math
N_{Sc} = \frac{\nu_{air}}{D_{vapor}}
```
where:
 - ``\nu_{air}`` is the kinematic viscosity of air.
The Reynolds number of a rain drop is defined as:
```math
N_{Re} = \frac{2 \, r \, v_{drop}(r, \rho)}{\nu_{air}} = \frac{2 v_c(\rho) \, g^{1/2} \, r^{3/2}}{\nu_{air}}
```
The final integral is:
```math
\begin{equation}
\left. \frac{d \, q_{rai}}{dt} \right|_{evap}  =  4 \pi S(q_{vap}, q_{vap}^{sat}) \frac{n_{0_{MP}} G(T)}{\rho}
                                                  \int_0^\infty \left( a_{vent} r + b_{vent} N_{Sc}^{1/3} (2 v_c(\rho))^{1/2} \frac{g^{1/4}}{\nu_{air}^{1/2}} r^{7/4} \right) exp(-\lambda_{MP} r) dr
\end{equation}
```
Integrating and eliminating ``\lambda_{MP}`` using eq.(\ref{eq:lambda})
results in:
```math
\begin{equation}
\left. \frac{d \, q_{rai}}{dt} \right|_{evap}  = S(q_{vap}, q_{vap}^{sat}) \frac{G(T) n_{0_{MP}}^{1/2}}{\rho} \left( A q_{rai}^{1/2} + B \frac{g^{1/4}}{n_{0_{MP}}^{3/16} \nu_{air}^{1/2}} q_{rai}^{11/16} \right)
\end{equation}
```
where:
 - ``A = (2 \pi)^{1/2} a_{vent} \left( \frac{\rho}{\rho_{water}} \right)^{1/2}``
 - ``B = \Gamma\left(\frac{11}{4}\right) 2^{7/16} \pi^{5/16} b_{vent} N_{Sc}^{1/3} v_c(\rho)^{1/2} \left( \frac{\rho}{\rho_{water}} \right)^{11/16}``

The values of ``a_{vent}`` and ``b_{vent}`` are chosen so that at ``q_{tot}
= 15 g/kg`` and ``T=288K`` the resulting rain evaporation rate is close
to the empirical rain evaporation rate from Smolarkiewicz and Grabowski 1996.

```@example rain_evaporation
using ClimateMachine.Microphysics
using ClimateMachine.MoistThermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: R_d, planet_radius, grav, MSLP, molmass_ratio
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using Plots

# eq. 5c in Smolarkiewicz and Grabowski 1996
# https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
function rain_evap_empirical(q_rai::DT, q::PhasePartition, T::DT, p::DT, ρ::DT) where {DT<:Real}

    q_sat  = q_vap_saturation(param_set, T, ρ, q)
    q_vap  = q.tot - q.liq
    rr     = q_rai / (DT(1) - q.tot)
    rv_sat = q_sat / (DT(1) - q.tot)
    S      = q_vap/q_sat - DT(1)

    ag, bg = 5.4 * 1e2, 2.55 * 1e5
    G = DT(1) / (ag + bg / p / rv_sat) / ρ

    av, bv = 1.6, 124.9
    F = av * (ρ/DT(1e3))^DT(0.525)  * rr^DT(0.525) + bv * (ρ/DT(1e3))^DT(0.7296) * rr^DT(0.7296)

    return DT(1) / (DT(1) - q.tot) * S * F * G
end

# example values
T, p = 273.15 + 15, 90000.
ϵ = 1. / molmass_ratio(param_set)
p_sat = saturation_vapor_pressure(param_set, T, Liquid())
q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.))
q_rain_range = range(1e-8, stop=5e-3, length=100)
q_tot = 15e-3
q_vap = 0.15 * q_sat
q_ice = 0.
q_liq = q_tot - q_vap - q_ice
q = PhasePartition(q_tot, q_liq, q_ice)
R = gas_constant_air(param_set, q)
ρ = p / R / T

plot(q_rain_range * 1e3,  [conv_q_rai_to_q_vap(param_set, q_rai, q, T, p, ρ) for q_rai in q_rain_range], xlabel="q_rain [g/kg]", ylabel="rain evaporation rate [1/s]", title="Rain evaporation", label="ClimateMachine")
plot!(q_rain_range * 1e3, [rain_evap_empirical(q_rai, q, T, p, ρ) for q_rai in q_rain_range], label="empirical")
savefig("rain_evaporation_rate.svg") # hide
nothing # hide
```
![](rain_evaporation_rate.svg)

## References

@article{Grabowski_and_Smolarkiewicz_1996,
author = {Grabowski, Wojciech W. and Smolarkiewicz, Piotr K.},
title = {Two-Time-Level Semi-Lagrangian Modeling of Precipitating Clouds},
journal = {Monthly Weather Review},
volume = {124},
number = {3},
pages = {487-497},
year = {1996},
doi = {10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2}}

@article{Kessler_1995,
author = {Kessler, E.},
title = {On the continuity and distribution of water substance in atmospheric circulations},
journal = {Atmospheric Research},
volume = {38},
number = {1},
pages = {109 - 145},
year = {1995},
doi = {10.1016/0169-8095(94)00090-Z}}

@book{Mason_1971,
author = {Mason, B. J.},
title = {The Physics of Clouds},
publisher = {Oxford Univ. Press},
year = {1971}}

@article{Marshall_and_Palmer_1948,
author = {Marshall, J. S. and Palmer, W. Mc K.},
title = {The distribution of raindrops with size},
journal = {Journal of Meteorology},
volume = {5},
number = {4},
pages = {165-166},
year = {1948},
doi = {10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2}}

@article{Ogura_and_Takahashi_1971,
author = {Oqura, Yoshimitsu and Takahashi, Tsutomu},
title = {Numerical simulation of the life cycle of a thunderstorm cell},
journal = {Monthly Weather Review},
volume = {99},
number = {12},
pages = {895-911},
year = {1971},
doi = {10.1175/1520-0493(1971)099<0895:NSOTLC>2.3.CO;2}}

@article{Seifert_and_Beheng_2006,
author={Seifert, A. and Beheng, K. D.},
title={A two-moment cloud microphysics parameterization for mixed-phase clouds. Part 1: Model description},
journal={Meteorology and Atmospheric Physics},
year={2006},
volume={92},
number={1},
pages={45--66},
doi={10.1007/s00703-005-0112-4}}
