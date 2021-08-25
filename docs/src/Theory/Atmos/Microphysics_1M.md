# Microphysics

We are using the `Microphysics_1M.jl` module
  from the [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) package.
See the [documentation](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/)
  for further comments on the scheme derivation.

## Coupling to the state variables

### Warm Rain

The source of rain ``\mathcal{S}_{q_{rai}}`` is a sum of the
  [autoconversion](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Rain-autoconversion),
  [accretion](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Accretion), and
  [rain evaporation](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Rain-evaporation-and-snow-sublimation)
  processes.
The sink of total water is equal to the source of rain:
  ``\mathcal{S}_{q_{tot}}`` = -``\mathcal{S}_{q_{rai}}``.
The sink of cloud liquid water is the sum of rain autoconversion
  and rain accretion processes.
Following the conservation equations for
[mass](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Mass),
[moisture](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Moisture), and
[precipitation](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Precipitating-Species)
the ``\mathcal{S}_{q_{tot}}`` sink has to be multiplied by ``\rho`` before
  adding it as one of the sink terms to both ``\rho`` and ``\rho q_{tot}``
  state variables.
The ``\mathcal{S}_{q_{liq}}``, ``\mathcal{S}_{q_{rai}}`` sources have to be multiplied by ``\rho``
  before adding them as one of the source terms to ``\rho q_{liq}`` and
  ``\rho q_{rai}``state variables.
For the conservation equation for
[total energy](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Energy),
  the sink due to removing ``q_{tot}`` is computed as:
```math
\begin{equation}
\left. \mathcal{S}_{\rho e} \right|_{precip} =
  \left. \sum_{j\in\{v,l,i\}}(I_j + \Phi)  \rho C(q_j \rightarrow q_p) \right|_{precip} =
  (I_l + \Phi) \rho \, \mathcal{S}_{q_{tot}}
\end{equation}
```
where:
 - ``I_l = c_{vl} (T - T_0)`` is the internal energy of liquid water,
 - ``T`` is the temperature,
 - ``T_0`` is the thermodynamic reference temperature (which is unrelated to the reference temperature used in hydrostatic reference states used in the momentum equations),
 - ``c_{vl}`` is the isochoric specific heat of liquid water,
 - ``\Phi`` is the effective gravitational potential.

### Rain and Snow

The source of rain ``\mathcal{S}_{q_{rai}}`` is a sum of the
  [autoconversion](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Rain-autoconversion),
  [accretion](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Accretion),
  [rain evaporation](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Rain-evaporation-and-snow-sublimation), and
  [snow melt](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Snow-melt)
  processes.
Similarily, the source of snow ``\mathcal{S}_{q_{sno}}`` is a sum of the
  [autoconversion](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Snow-autoconversion),
  [accretion](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Accretion),
  [snow deposition/sublimation](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Rain-evaporation-and-snow-sublimation), and
  [snow melt](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Snow-melt)
  processes.
The sink of total water is equal to the sum of the rain and snow sources:
  ``\mathcal{S}_{q_{tot}}`` = -``\mathcal{S}_{q_{rai}}`` - ``\mathcal{S}_{q_{sno}}``.
The sink of cloud liquid water ``\mathcal{S}_{q_{liq}}`` is the sum of
  rain autoconversion,
  rain accretion with cloud liquid water, and
  snow accretion with cloud liquid water processes.
The sink of cloud ice ``\mathcal{S}_{q_{ice}}`` is the sum of
  snow autoconversion,
  snow accretion with cloud ice, and
  rain accretion with cloud ice processes.
Following the conservation equations for
[mass](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Mass),
[moisture](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Moisture), and
[precipitation](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Precipitating-Species)
the ``\mathcal{S}_{q_{tot}}`` sink has to be multiplied by ``\rho`` before
  adding it as one of the sink terms to both ``\rho`` and ``\rho q_{tot}``
  state variables.
The ``\mathcal{S}_{q_{liq}}``, ``\mathcal{S}_{q_{ice}}``,
  ``\mathcal{S}_{q_{rai}}``, and ``\mathcal{S}_{q_{sno}}`` sources
  have to be multiplied by ``\rho`` before adding them as one of the
  source terms to ``\rho q_{liq}``, ``\rho q_{ice}``, ``\rho q_{rai}``, and
  ``\rho q_{sno}``state variables.
For the conservation equation for
[total energy](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/AtmosModel/#Energy)
  the source term due to microphysics processes is caused by
  either removing cloud condensate outside of the working fluid
  or by changing phase and releasing latent heat outside of the working fluid.
Below, contributions from different microphysics processes are listed:\\
The contribution from
  cloud liquid water to rain autoconversion,
  cloud liquid water accretion by rain,
  and rain evaporation
  is computed as:
```math
\begin{equation}
\left. \mathcal{S}_{\rho e} \right|_{precip} =
  - (I_l + \Phi) \rho \, \mathcal{S}_{q_{rai}}
\end{equation}
```
where:
 - ``I_l = c_{vl} (T - T_0)`` is the internal energy of liquid water,
 - ``T`` is the temperature,
 - ``T_0`` is the thermodynamic reference temperature (which is unrelated to the reference temperature used in hydrostatic reference states used in the momentum equations),
 - ``c_{vl}`` is the isochoric specific heat of liquid water,
 - ``\Phi`` is the effective gravitational potential.
 - ``\mathcal{S}_{q_{rai}}`` is the source of rain from the above three processes.
The contribution from
  cloud ice to snow autoconversion,
  cloud ice accretion by snow,
  cloud liquid accretion by snow in temperatures below freezing,
  and snow deposition/sublimation
  is computed as:
```math
\begin{equation}
\left. \mathcal{S}_{\rho e} \right|_{precip} =
  - (I_i + \Phi) \rho \, \mathcal{S}_{q_{sno}}
\end{equation}
```
where:
 - ``I_i = c_{vi} (T - T_0) - I_{i0}`` is the internal energy of ice,
 - ``c_{vi}`` is the isochoric specific heat of ice,
 - ``I_{i0}`` is the difference in specific internal energy between ice and liquid at ``T_0``,
 - ``\mathcal{S}_{q_{sno}}`` is the source of snow from the above four processes.
The contribution from
  accretion of cloud liquid water by snow in temperatures above freezing
  is computed as:
```math
\begin{equation}
\left. \mathcal{S}_{\rho e} \right|_{precip} =
  ((1 + \alpha) I_l - \alpha I_i + \Phi) \rho \, \mathcal{S}_{q_{liq}}
\end{equation}
```
where:
 - ``\alpha = \frac{c_{vl}}{L_f}(T - T_{freeze})``
 - ``\mathcal{S}_{q_{liq}}`` is the source of cloud liquid water from
  accretion of cloud liquid water by snow in temperatures above freezing.
The contribution from cloud ice accretion by rain (the result is snow)
  is computed as:
```math
\begin{equation}
\left. \mathcal{S}_{\rho e} \right|_{precip} =
   (I_i + \Phi) \rho \, \mathcal{S}_{q_{ice}} - \rho L_f \mathcal{S}_{q_{rai}}
\end{equation}
```
 where:
 - ``\mathcal{S}_{q_{ice}}`` and ``\mathcal{S}_{q_{rai}}``
  are the sinks of cloud ice and rain due to accretion..
Finally, the contribution from accretion between rain and snow
  as well as snow melting into rain is computed as:
```math
\begin{equation}
\left. \mathcal{S}_{\rho e} \right|_{precip} =
   \rho L_f \mathcal{S}_{q_{sno}}
\end{equation}
```
 where:
 - ``\mathcal{S}_{q_{sno}}`` is the source of snow in those two processes.
