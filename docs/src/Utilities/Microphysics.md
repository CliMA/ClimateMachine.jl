# Microphysics Module

The `Microphysics` module provides 1-moment bulk parameterization of cloud microphysical processes. The module describes the warm rain (no ice and snow) formation and is based on the ideas of Kessler 1995. The cloud microphysics variables are expressed as specific humidities:
  - q_tot - total water specific humidity
  - q_vap - water vapor specific humidity
  - q_liq - liquid water specific humidity
  - q_rai - rain water specific humidity
Constant parameters used in the parameterization are defined in `MicrophysicsParameters` module.

Parameterized processes include
  - rain sedimentation with mass weighted average terminal velocity
  - condensation/evaporation of cloud water
  - autoconversion
  - accretion
  - evaporation of rain water

## terminal velocity

The mass weighted average rain terminal velocity is calculated assuming Marshall-Palmer distribution of rain drops.

The assumed rain-drop size distribution (Marshall Palmer 1948 eq. 1) as a function of rain drop diameter:
```math
\begin{equation}
n(D) = n_{0_{MP}} exp\left(- \lambda_{MP} D \right)
\end{equation}
```
where
 - ``D`` is the drop diameter
 - ``n_{0_{MP}}`` and ``\lambda_{MP}`` are the Marshall-Palmer distribution parameters.

In the remaining part of the Microphysics module derivation this size distribution is expressed as a function of drop radius ``r``
```math
\begin{equation}
n(D) = 2 n_{0_{MP}} exp\left(- 2 \lambda_{MP} r \right)
\end{equation}
```

The terminal velocity of an individual rain drop is defined by the balance between the gravity (taking into account the density difference between water and air) and the drag force:
```math
\begin{equation}
v_{drop} = \left(\frac{8}{3} \frac{g}{C_{drag}} \left( \frac{\rho_{water}}{\rho} -1 \right) \right)^{1/2} r^{1/2} = v_c r^{1/2}
\end{equation}
```
where:
 - ``g`` is the gravitational acceleration
 - ``C_{drag}`` is the drag coefficient
 - ``\rho_{water}`` is the density of water
 - ``\rho`` is the density of air.

The mass weighted terminal velocity ``v_t`` is defined following Ogura and Takahashi 1971
```math
\begin{equation}
v_t = \frac{F_{rain}}{RWC}
\end{equation}
```
where:
 - ``F_{rain} = \int_0^\infty n(r) \; m(r) \; v_{drop}(r) dr`` is the vertical flux of rain drops
 - ``RWC = \int_0^\infty n(r) \; m(r) dr = \rho \; q_{rai}`` is the rain water content.

Integrating over the assumed Marshall-Palmer distribution results in
```math
\begin{equation}
RWC = \frac{\pi \; n_{0_{MP}} \; \rho_{water}}{\lambda_{MP}^4}
\end{equation}
```
```math
F_{rain} = \Gamma \left(\frac{9}{2} \right) \frac{8}{3} n_{0_{MP}} \; \pi \; \rho_{water} v_c  (2 \lambda_{MP})^{-9/2}
```
Eliminating ``\lambda_{MP}`` between the two equations and dividing by ``RWC`` results in
```math
\begin{equation}
v_t = \Gamma \left( \frac{9}{2} \right) \; \frac{v_c}{6 \sqrt{2}} \; (n_{0_{MP}} \; \pi)^{-1/8} \left( \frac{\rho}{\rho_{water}} \right)^{1/8} q_{rai}^{1/8} = v_{t_{c}} \left( \frac{\rho}{\rho_{water}} \right)^{1/8} q_{rai}^{1/8}
\end{equation}
```
The default values of the two parameters in the equation for ``v_t`` are ``n_{0_{MP}} = 8e6 \; 1/m^4`` and ``C_{drag} = 0.55``. The ``n_{0_{MP}}`` value is taken from the Marshal Palmer 1948. The ``C_{drag}`` is chosen such that the ``v_t`` values match the empirical terminal velocity formulation in Smolarkiewicz and Grabowski 1996. Assuming a constant drag coefficient is an approximation as it should be size and flow dependent, see [drag_coefficient](https://www.grc.nasa.gov/www/K-12/airplane/dragsphere.html).


```@example rain_terminal_velocity
using CLIMA.Microphysics
using Plots, LaTeXStrings

# eq. 5d in Smolarkiewicz and Grabowski 1996
# https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
function terminal_velocity_empirical(q_rai::DT, q_tot::DT, ρ::DT, ρ_air_ground::DT) where {DT<:Real}
    rr  = q_rai / (DT(1) - q_tot)
    vel = DT(14.34) * ρ_air_ground^DT(0.5) * ρ^-DT(0.3654) * rr^DT(0.1346)
    return vel
end

q_rain_range = range(0, stop=5e-3, length=100)
ρ_air, q_tot, ρ_air_ground = 1.2, 10 * 1e-3, 1.22

plot(q_rain_range * 1e3,  [terminal_velocity(q_rai, ρ_air) for q_rai in q_rain_range], xlabel="q_rain [g/kg]", ylabel="velocity [m/s]", title="Average terminal velocity of rain", label="CLIMA")
plot!(q_rain_range * 1e3, [terminal_velocity_empirical(q_rai, q_tot, ρ_air, ρ_air_ground) for q_rai in q_rain_range], label="empirical")
savefig("rain_terminal_velocity.svg") # hide
nothing # hide
```
![](rain_terminal_velocity.svg)

## cloud condensation/evaporation

Condensation and evaporation of cloud water is parametrized as a relaxation to equilibrium. The equilibrium state is calculated using saturation adjustment defined in the `MoistThermodynamics` module.
```math
\begin{equation}
  \left. \frac{d \; q_{liq}}{dt} \right|_{cond, evap} = \frac{q^{eq}_{liq} - q_{liq}}{\tau_{cond\_evap}}
\end{equation}
```
where:
 - ``q^{eq}_{liq}`` - liquid water specific humidity in equilibrium
 - ``q_{liq}`` - liquid water specific humidity.
 - ``\tau_{cond\_evap}`` - relaxation timescale (parameter in `MicrophysicsParameters` module).
## autoconversion

Autoconversion defines a rate of conversion form cloud to rain water resulting from collisions between cloud droplets. It is parametrized following Kessler 1995.
```math
\begin{equation}
  \left. \frac{d \; q_{rai}}{dt} \right|_{acnv} = \frac{max(0, q_{liq} - q_{liq\_threshold})}{\tau_{acnv}}
\end{equation}
```
where:
 - ``q_{liq}`` - liquid water specific humidity.
 - ``\tau_{acnv}`` - timescale (parameter in `Microphysics` module)
 - ``q_{liq\_threshold}`` - autoconversion (parameter in `MicrophysicsParameters` module).
## accretion

Accretion defines the rate of conversion from cloud to rain water resulting from collisions between cloud droplets and rain drops. It is parameterized following Kessler 1995 and Ogura and Takahashi_1971.

## rain evaporation




```@meta
CurrentModule = CLIMA.Microphysics
```

## Functions

```@docs
terminal_velocity
conv_q_vap_to_q_liq
conv_q_liq_to_q_rai_acnv
conv_q_liq_to_q_rai_accr
conv_q_rai_to_q_vap
```

## References

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

@article{Kessler_1995,
author = {Kessler, E.},
title = {On the continuity and distribution of water substance in atmospheric circulations},
journal = {Atmospheric Research},
volume = {38},
number = {1},
pages = {109 - 145},
year = {1995},
doi = {10.1016/0169-8095(94)00090-Z}}

@article{Grabowski_and_Smolarkiewicz_1996,
author = {Grabowski, Wojciech W. and Smolarkiewicz, Piotr K.},
title = {Two-Time-Level Semi-Lagrangian Modeling of Precipitating Clouds},
journal = {Monthly Weather Review},
volume = {124},
number = {3},
pages = {487-497},
year = {1996},
doi = {10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2}}
