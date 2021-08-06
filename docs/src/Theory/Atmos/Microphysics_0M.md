# Microphysics_0M

We are using the `Microphysics_0M.jl` module
  from the [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) package.
See the [documentation](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_0M/)
  for further comments on the scheme derivation.

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
\left. \mathcal{S}_{\rho e} \right|_{precip} =
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
