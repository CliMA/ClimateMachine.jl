
Isabella Dula and Shevali Kadakia
    
This file has the complete set of tests to verify and validate the parameterization of
the model given in Abdul-Razzak and Ghan (2000). 
The structure of the pipeline is as follows:
--Test classifications:
    --Verfication (VER): ensures that function output has consistent output, 
    no matter inputted values (i.e. verifies that the functions are 
    doing what we want them to)
    --Validation (VAL): checks functions against model data in 
    Abdul-Razzak and Ghan (2000) (i.e., validates the functions outputs 
    against published results)
--Dimension (DIM):
    --Tests are done with multi-dimensional inputs: 
    --0: Only one mode and one component (e.g., coarse sea salt) 
    --1: Considering multiple modes over one component (e.g., a
    ccumulation mode and coarse mode sea salt)
    --2: Considering multiple modes with multiple components (e.g., 
    accumulation and coarse mode for sea salt and dust)
--Modes and Components Considered
        --This testing pipeline uses aerosol data from 
        Porter and Clarke (1997) to provide real-world inputs into the functions
        --Modes: Accumulation (ACC) and coarse (COA)
        --Components: Sea Salt (SS) and Dust (DUS)

Assumptions:
  - There are no kinetic limitations on the aerosol activation process.
  - The particles are activated if their critical supersaturation is less than the maximum supersaturation in the numerical integration.
  - Do not consider the effects of surfactants on surface tension.
  - The solute is sufficiently soluble so its concentration does not increase as the droplet grows.

Other notes about the parametrization:
  - Using a lognormal aerosol size distribution to derive the parametrization that is naturally bounded at high updraft velocity.
  - Worked with 2 broad aerosol modes rather than many more narrow ones. Each mode has an interal mixture of material and they all compete for water.

The following are equations used for the parametrization.

# MAIN FUNCTIONS

1: Mean Hygroscopicity Paramter
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

2: Maximum Supersaturation
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

3: Total Number of Activated Aerosols
```math
N = \sum_{i = 1}^{I} N_{i}\frac{1}{2}[1 - \mathrm{erf}(u_{i})]
```
where
  - ``N_{i}`` is the total number concentration.
  - ``u_{i} \equiv \frac{\mathrm{ln}(a_{ci}/a_{mi})}{\sqrt{2} \mathrm{ln} \sigma_{i}} = \frac{2 \mathrm{ln}(S_{mi}/S_{max})}{3\sqrt{2} \mathrm{ln} \sigma_{i}}``
  - ``\sigma_{i}`` is the geometric standard deviation for aerosol mode ``i``.

# HELPER FUNCTIONS

1: Critical Supersaturation Calculation
```math
S_mi = \frac{2}{\bar{B_i}}^{0.5} * \frac{3}{part_rds}^{3/2}
```
where
  - ``S_mi`` is the supersaturation of the aerosol mode ``i``.
  - ``\bar{B_i}`` is the mean hygroscopicity parameter of the aerosol material in the aerosol mode ``i``.
  - ``part_rds`` is the mean radius of the particles.

2: Coefficient of Curvature Calculation
```math
A = \frac{2 * act_time .* wtr_mm}{wtr_ρ .* R .* temp}
```
where
  - ``act_time`` is time that the aerosol gets activated. 
  - ``water_molecular_mass`` is the molecular mass of water. 
  - ``water_density`` is the density of water. 
  - ``R`` is the universal gas constant. 
  - ``temperature`` is the temperature.

