# Microphysics Module

The `Microphysics` module provides 1-moment bulk parameterization of cloud
microphysical processes. The module describes the warm rain (no ice and snow)
formation and is based on the Kessler_1995 parameterization.
The cloud microphysics variables are expressed as specific humidities:
  - q_tot - total water specific humidity
  - q_vap - water vapor specific humidity
  - q_liq - liquid water specific humidity
  - q_rai - rain water specific humidity
Parameters are defined in `MicrophysicsParameters` module.

Parameterized processes include
  - rain sedimentation with mass weighted mean terminal velocity
  - condensation/evaporation of cloud water as a relaxation to equilibrium
  - autoconversion
  - accretion
  - rain evaporation

## rain terminal velocity

The mass weighted average rain terminal velocity is calculated assuming
Marshall-Palmer distribution of rain drops.

The assumed rain-drop size distribution (Marshal_Palmer_1948 eq 1)
as a function of rain drop radius:
\begin{equation}
n(r) = MP_n_0 * exp(\frac{- \lambda}{2} r)
\end{equation}
where
 - $r$ is the drop radius
 - $MP_n_0$ and $\lambda$ are the distribution parameters.

The terminal velocity of an individual rain drop is defined by the balance
between the gravity (taking into account the density difference between
water and air) and the drag force:

\begin{equation}
v_{drop} = \left(\frac{8}{3} \frac{g}{C_{drag}} (\frac{\rho_{water}}{\rho} -1) \right)^{1/2} r^{1/2}
\end{equation}

where:
 - $g$ is the gravitational acceleration
 - $C_{drag}$ is the drag coefficient
 - $\rho_{water}$ is the density of water
 - $\rho$ is the density of air

The mass weighted terminal velocity is derived following
Ogura_and_Takahashi_1971 but without using empirical relations.


```@meta
CurrentModule = CLIMA.Microphysics
```

## Functions API

```@docs
terminal_velocity
conv_q_vap_to_q_liq
conv_q_liq_to_q_rai_acnv
conv_q_liq_to_q_rai_accr
conv_q_rai_to_q_vap
```
