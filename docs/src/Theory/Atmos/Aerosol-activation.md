# Aerosol Activation Parametrization

The `Aerosol-activation.jl` module defines a parameterization of
  the activation of multiple lognormal modes with a uniform internal mixture 
  composed of both soluble and insoluble materials.
It uses the combination of previous studies and the Köhler theory to model the formation of cloud droplets.

Aerosol activation is when cloud particles form on the aerosol particles existing in the air.

Assumptions:
  - There are no kinetic limitations on the aerosol activation process.
  - The particles are activated if their critical supersaturation is less than the maximum supersaturation in the numerical integration.
  - Do not consider the effects of surfactants on surface tension.
  - The solute is sufficiently soluble so its concentration does not increase as the droplet grows.

Other notes about the parametrization:
  - Using a lognormal aerosol size distribution to derive the parametrization that is naturally bounded at high updraft velocity.
  - Worked with 2 broad aerosol modes rather than many more narrow ones. Each mode has an interal mixture of material and they all compete for water.

The following are equations used for the parametrization.

1: Hydgroscopicity Parameter
```math
B \equiv \frac{v \phi \epsilon M_{w} \rho_a}{M_a \rho_{w}}
```
where
  - ``B`` is the hygroscopicity parameter.
  - ``v`` is the number of ions salt disassociates into when in water.
  - ``\phi`` is the osmotic coefficient.
  - ``\epsilon`` is the mass fraction of soluble material.
  - ``M_{w}`` is the molecular weight of water.
  - ``\rho_{a}`` is the density of the aerosol material.
  - ``M_{a}`` is the molecular weight of the aerosol material.
  - ``p_{w}`` is the density of water.

2: Mean Hygroscopicity Paramter
```math
\bar{B_i} \equiv \frac{M_{w} \sum_{j = 1}^{J} r_{ij} v_{ij} \phi_{ij} \epsilon_{ij} /M_{aij}} {\rho_{w} \sum_{j = 1}^{J} r_{ij}/p_{aij}}
```
where
  - ``\bar{B_i}`` is the mean hygroscopicity parameter of the aerosol material in the aerosol mode ``i``.
  - ``v_{ij}`` is the number of ions that salt disassociates into when in water of aerosol component ``j`` and mode ``i``.
  - ``\phi_{ij}`` is the osmotic coefficient of aerosol component ``j`` and mode ``i``.
  - ``\epsilon_{ij}`` is the mass fraction of the soluble material of aerosol component ``j`` and mode ``i``.
  - ``M_w`` is the molecular weight of water.
  - ``\rho_{aij}`` is the density of the aerosol's component ``j`` and mode ``i``.
  - ``M_{aij}`` is the molecular weight of aerosol component ``j`` and mode ``i``.
  - ``\rho_{w}`` is the density of water.
  - ``j`` is the component number. Min value is 1 and max value is ``J``.
  - ``r_{ij}`` is the mass mixing ratio of component ``j`` and mode ``i``.
  - ``i`` is the mode number.

3: Condensation Rate
```math
\frac{dw}{dt} = 4 \pi \rho_{w} \sum_{i = 1}^{I} \int_{0}^{S} r_i^2 \frac{dr_i}{dt} \frac{dn_i(S')}{da} dS'
```
where
  - ``i`` is the aerosol type. The min is 1 and the max is ``I``.
  - ``r_{i}`` is the radius of the droplet forming on aerosol type ``i``.
  - ``S`` and ``S'`` represent the supersaturation.
  - ``n_i`` is the number concentration of aerosol mode i.
  - ``w`` is the condensation.
  - ``\rho_w`` is the density of water.
  - ``a`` is the geometric mean dry radius.

4: Maximum/Critical Supersaturation
```math
S_{max} = \frac{1}{{\sum_{i=1}^{I} \frac{1}{S_{mi}^{2}} [f_i(\frac{\zeta}{\eta_{i}})^{\frac{3}{2}} + g_{i}(\frac{S_{mi}^{2}}{\eta_{i} + 3\zeta})^{\frac{3}{4}}}]^{\frac{1}{2}}}

```
where
  - ``S_{mi} = \frac{2}{\sqrt{\bar{B_{i}}}}(\frac{A}{3a_{mi}})^{3/2}``
  - ``\zeta \equiv \frac{2A}{3}(\frac{\alpha V}{G-})^{\frac{1}{2}}``
  - ``f_i \equiv 0.5 \mathrm{exp} (2.5 \mathrm{ln}^{2} \sigma_{i})``
  - ``\eta_{i} \equiv \frac{(\alpha V/G)^{\frac{3}{2}}}{2 \pi \rho_{w} \gamma N_{i}}``
  - ``g_i \equiv 1 + 0.25 \mathrm{ln} \sigma_i``
  - ``A`` is the surface tension effects in the Köhler equilibrium equation.
  - ``V`` is the updraft velocity.
  - ``G`` is the diffusion of heat and moisture to the particles.
  - ``\alpha`` and ``\gamma`` are coefficients in the supersaturation balance equation.

5: Dry Radius
```math
a_{ci} = a_{mi}(\frac{S_{mi}}{S_{max}})^{\frac{2}{3}}       
```
where
  - ``i = 1,...,I``
  - ``a_{mi}`` is the geometric mean dry radius.
  -  ``S_{mi} = \frac{2}{\sqrt{\bar{B_{i}}}}(\frac{A}{3a_{mi}})^{3/2}``
  - ``S_{max}`` is the equation 4.

6: Total Number of Activated Aerosols
```math
N = \sum_{i = 1}^{I} N_{i}\frac{1}{2}[1 - \mathrm{erf}(u_{i})]
```
where
  - ``N_{i}`` is the total number concentration.
  - ``u_{i} \equiv \frac{\mathrm{ln}(a_{ci}/a_{mi})}{\sqrt{2} \mathrm{ln} \sigma_{i}} = \frac{2 \mathrm{ln}(S_{mi}/S_{max})}{3\sqrt{2} \mathrm{ln} \sigma_{i}}``
  - ``\sigma_{i}`` is the geometric standard deviation for aerosol mode ``i``.

7: Total Mass of Activated Aerosols
```math
M = \sum_{i = 1}^{I} M_{i}\frac{1}{2}[1 - \mathrm{erf}(u_{i} - \frac{3\sqrt{2}}{2} \mathrm{ln}\sigma_{i})]
```
where
  - ``u_{i} \equiv \frac{\mathrm{ln}(a_{ci}/a_{mi})}{\sqrt{2} ln \sigma_{i}} = \frac{2 \mathrm{ln}(S_{mi}/S_{max})}{3\sqrt{2} ln \sigma_{i}}``
  - ``\sigma_{i}`` is the geometric standard deviation for aerosol mode ``i``.
  - ``M_{i}`` is the mass of the aerosol mode ``i``.


Variables:
  - ``osmotic_coefficient`` 
  - ``temperature``
  - ``aerosol_molecular_mass``
  - ``aerosol_particle_density``
  - ``aerosol_density``
  - ``water_density``
  - ``water_molar_density``
  - ``water_molecular_mass``
  - ``updraft_velocity``
  - ``R``
  - ``avogadros_number``
  - ``dissociation``
  - ``mass_fraction``
  - ``particle_radius``
  - ``particle_radius_stdev``
  - ``activation_time``
  - ``alpha``
  - ``diffusion``
  -  ``gamma``