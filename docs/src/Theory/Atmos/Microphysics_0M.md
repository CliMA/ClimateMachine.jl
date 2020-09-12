# Microphysics_0M

The `Microphysics_0M.jl` module defines a 0-moment bulk parameterization of
  the moisture sink due to precipitation.
It offers a simplified way of removing the excess water
  without assuming anything about the size distributions of cloud
  or precipitation particles.

The ``q_{tot}`` (total water specific humidity) sink due to precipitation
  is obtained by relaxation with a constant timescale
  to a state with condensate exceeding a threshold value removed.
The threshold for removing excess ``q_{tot}`` is defined either by the
  condensate specific humidity or supersaturation.
The thresholds and the relaxation timescale are defined in
  `CLIMAParameters.jl`.

!!! note

    To remove precipitation instantly, the relaxation timescale should be
    equal to the timestep length.

## Moisture sink due to precipitation

If based on maximum condensate specific humidity, the sink is defined as:
``` math
\begin{equation}
  \left. \mathcal{S}_{q_{tot}} \right|_{precip} =-
    \frac{max(0, q_{liq} + q_{ice} - q_{c0})}{\tau_{precip}}
\end{equation}
```
where:
  - ``q_{liq}``, ``q_{ice}`` are cloud liquid water and cloud ice specific humidities,
  - ``q_{c0}`` is the condensate specific humidity threshold above which water is removed,
  - ``\tau_{precip}`` is the relaxation timescale.

If based on saturation excess, the sink is defined as:
```math
\begin{equation}
  \left. \mathcal{S}_{q_{tot}} \right|_{precip} =-
    \frac{max(0, q_{liq} + q_{ice} - S_{0} \, q_{vap}^{sat})}{\tau_{precip}}
\end{equation}
```
where:
  - ``q_{liq}``, ``q_{ice}`` are cloud liquid water and cloud ice specific humidities,
  - ``S_{0}`` is the supersaturation threshold above which water is removed,
  - ``q_{vap}^{sat}`` is the saturation specific humidity,
  - ``\tau_{precip}`` is the relaxation timescale.

## Coupling to the state variables

Following the conservation equations for
[moisture](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Moisture)
and [mass](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Mass),
the ``\mathcal{S}_{q_{tot}}`` sink has to be multiplied by ``\rho`` before
  adding it as one of the sink terms to both moisture and mass state variables.
For the conservation equation for
[total energy](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Energy),
  no additional source/sink terms $M$ are considered, and the
  the sink due to removing ``q_{tot}`` is computed as:
```math
\begin{equation}
\left. \sum_{j\in\{v,l,i\}}(I_j + \Phi)  \rho C(q_j \rightarrow q_p) \right|_{precip} =
  \left[\lambda I_l + (1 - \lambda) I_i + \Phi \right]
  \rho \, \left.\mathcal{S}_{q_{tot}} \right|_{precip}
\end{equation}
```
where:
 - ``\lambda`` is the liquid fraction
 - ``I_l = c_{vl} (T - T_0)`` is the internal energy of liquid water
 - ``I_i = c_{vi} (T - T_0) - I_{i0}`` is the internal energy of ice
 - ``T`` is the temperature,
 - ``T_0`` is the thermodynamic reference temperature (which is unrelated to the reference temperature used in hydrostatic reference states used in the momentum equations),
 - ``I_{i0}`` is the specific internal energy of ice at ``T_0``
 - ``c_{vl}`` and ``c_{vi}`` are the isochoric specific heats
     of liquid water, and ice.
 - ``\Phi`` is the effective gravitational potential.

This assumes that the ``\mathcal{S}_{q_{tot}}`` sink is partitioned between the
  cloud liquid water and cloud ice sinks
  ``\mathcal{S}_{q_{liq}}`` and ``\mathcal{S}_{q_{ice}}`` based on the
  cloud liquid water and cloud ice fractions.
