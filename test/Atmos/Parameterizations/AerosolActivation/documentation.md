# Aerosol Activation 
### Isabella Dula and Shevali Kadakia

The 'AerosolActivation.jl' module describes the parametrization of the activation of aerosol particles into cloud nucleii. Accompanying the module is the 'AerosolModel.jl' module, which packages aerosol models with constructors that allow for efficient and user-friendly organzation. The modules are an implementation of the parametrization described in Abdul-Razzak and Ghan (2000).

Aerosol activation describes the process by which liquid cloud particles condensate around suspended aerosol particles in the atmosphere. The final output of this parametrization is the total number of activated aerosol particles from a given aerosol population.  

## Model Assumptions
  - There are no kinetic limitations on the aerosol activation process.
  - The particles are activated if their critical supersaturation is less than the maximum supersaturation in the numerical integration.
  - The effects of surfactants on surface tension are not considered.
  - The solute is sufficiently soluble so its concentration does not increase as the droplet grows.
  - Aerosol particles in the same mode are *internally mixed*. 
  - An aerosol mode follows a log-normal size distribution. 
  - The aerosol activation only occurs when the water vapor reaches saturation conditions. 
 

## Aerosol Model 
The 'AerosolModel.jl' module creates constructors that organize the aerosol population so that they can be passed into the 'AerosolActivation.jl' module. Aerosols exist in different modes. Each mode comprises the group of aerosol species that share the same log-normal size distrubtion. Note that the same aerosol species has a different radius depending on its lifetime in the atmosphere and how it was formed. Thus, the same species can exist in different modes for each of its size distributions in the aerosol population. 

The following table gives the relevant physical and chemical properties to describe the species in a given aerosol mode. The particle dry radius, geometric standard deviation, and particle density describe the entire mode. Other species-dependent chemical properties are inputted for each aerosol species and weighted in the ``mean_hygroscopicity`` function according to their mass ratio. 

|    variable name     |         definition               | units      | 
|---------------|-----------------------------------------|-----------------
|``a_m``  | particle dry radius               | ``m``          |
|``\sigma``  | geometric standard deviation       | ``-``          |
|``N``  | particle density       | ``m^{-3}``          |
|``r``  | mass ratio of each chemical species      | ``-``          |
|``\epsilon``  | fraction of each chemical species that is water-soluble       | ``-``          |
|``\phi``  | osmotic coefficient      | ``-``          |
|``M_a``  | molar mass      | ``kg mol^{-1}``          |
|``\nu``  | number of ions dissociated into water    | ``-``          |
|``\rho_a``  | aerosol density      | ``kgm^{-3}``          |
|``J``  | number of components in mode       | ``-``          |

Note that each mode is labeled  ``i = 1, 2, ..., I`` and each component is labeled ``j = 1, 2, ..., J``, where ``I`` and ``J`` are the total modes and components in the system, respectively. 

## Aerosol Activation Parametrization
### Parameters
This parametrized process includes: 
  - The mean hygroscopicity of each aerosol mode
  - The maximum supersaturation
  - The total number of activated particles

Parameters used in the parameterization are defined in ``CLIMAParameters.jl`` package and the ``Microphysics.jl`` package, given in the following table:
|    variable name          |definition       | units             | 
|----------------------------|-----------------------------------------------------------|--------------------------|
|``R``      | universal gas constant  | J K^{-1} mol^{-1}  |
|``g``      | gravitational acceleration  | m s^{-2}  |    
|``T``      | local temperature  | T |  
|``P``      | local pressure  | Pa  |  
|``P_{sat}``      | saturation pressure  | Pa  |  
|``\M_W``      | molar mass of water  | kg mol^{-1}  |  
|``\rho_W``      | liquid water density  | kg m^{-3}  |
|``\M_{air}``      | molar mass of air  | kg mol^{-1|  
|``C_p`` | specific heat of air  | J kg^{-1} K{-1}  |    
|``L``      | latent heat of evaporation  | J kg^{-1}  |         
|``\tau``      | surface tension of water  | N m^{-1}  |
|``G``      | diffusion of heat and moisture to particles   | -  | 
|``V``      | updraft velocity  | m s^{-1}  |     

### Mean Hygroscopicity
The mean hygroscopicity describes an aerosol mode population's tendency to absorb moisture from the surrounding air. The mean hygroscopicity is computed for each mode according to equation (4) in Abdul-Razzak and Ghan (2000) and is defined as: 

```math
\begin{equation}
\bar{B_i} \equiv \frac{M_{w} \sum_{j = 1}^{J} r_{ij} \nu_{ij} \phi_{ij} \epsilon_{ij} /M_{aij}} {\rho_{w} \sum_{j = 1}^{J} r_{ij}/p_{aij}}
\end{equation}
```
where:
  - ``\nu_{ij}`` is the number of ions that salt disassociates into when in water of aerosol component ``j`` and mode ``i``.
  - ``\phi_{ij}`` is the osmotic coefficient of aerosol component ``j`` and mode ``i``.
  - ``\epsilon_{ij}`` is the mass fraction of the soluble material of aerosol component ``j`` and mode ``i``.
  - ``M_w`` is the molecular weight of water.
  - ``\rho_{aij}`` is the density of the aerosol component ``j`` and mode ``i``.
  - ``M_{aij}`` is the molar mass of aerosol component ``j`` and mode ``i``.
  - ``\rho_{w}`` is the density of water.
  - ``r_{ij}`` is the mass ratio of component ``j`` and mode ``i``.


### Maximum Supersaturation
The maximum supersaturation is given in equation (6) of Abdul-Razzak and Ghan (2000): 
```math
\begin{equation}
S_{max} = \frac{1}{{\sum_{i=1}^{I} \frac{1}{S_{mi}^{2}} [f_i(\frac{\zeta}{\eta_{i}})^{\frac{3}{2}} + g_{i}(\frac{S_{mi}^{2}}{\eta_{i} + 3\zeta})^{\frac{3}{4}}}]^{\frac{1}{2}}}
\end{equation}
```
where
  - ``S_{mi}`` is the critical supersaturation for mode *I*
  - ``\zeta`` is a coefficient the maximum supersaturation equation. 
  - ``\eta_{i}`` is a coefficient the maximum supersaturation equation. 
  - ``f_i`` is a coeffient of the mode variance in the maximum supersaturation equation.  
  - ``g_i `` is a coeffient of the mode variance in the maximum supersaturation equation. 
  - ``A`` is the coefficient for the surface tension effects in the Köhler equilibrium equation.
  - ``V`` is the updraft velocity.
  - ``G`` is the diffusion of heat and moisture to the particles.
  - ``\alpha`` is a coefficient in the supersaturation balance equation.
  - ``\gamma`` is a coefficient in the supersaturation balance equation.

f_i is given in equation (7) of Abdul-Razzak and Ghan (2000): 
```math
\begin(equation)
\f_i  \equiv 0.5 \mathrm{exp} (2.5 \mathrm{ln}^{2} \sigma_{i})
\end{equation}
```
where:
  - ``\sigma_i`` is the mode geometric standard deviation. 

g_i is given in equation (8) of Abdul-Razzak and Ghan (2000): 
```math
\begin(equation)
\g_i  \equiv 1 + 0.25 \mathrm{ln} \sigma_i
\end{equation}
```
where:
  - ``\sigma_i`` is the mode geometric standard deviation. 

The critical superation is given in equation (9) of Abdul-Razzak and Ghan (2000): 

```math
\begin{equation}
S_{mi} = \frac{2}{\sqrt{\bar{B_{i}}}}(\frac{A}{3a_{mi}})^{3/2}``
\end{equation}
```
where: 
  - ``\bar{B_i}`` is the mean hygroscopicity 
  - ``A`` is the coefficient for the surface tension effects in the Köhler equilibrium equation.
  - ``a_{mi}`` is the mean radius for mode ``i``


\zeta is given in equation (10) of Abdul-Razzak and Ghan: 

```math
\begin(equation)
\zeta \equiv \frac{2A}{3}(\frac{\alpha V}{G-})^{\frac{1}{2}}
\end(equation)
```
where: 
  - ``A`` is the coefficient for the surface tension effects in the Köhler equilibrium equation
  - ``\alpha`` is a coefficient in the supersaturation balance equation. 
  - ``V`` is the updraft velocity. 
  - ``G`` is the diffusion of heat and moisture to the particles. 

\eta is given in equation (11) of Abdul-Razzak and Ghan: 

```math
\begin(equation)
\eta_i \equiv \frac{(\alpha V/G)^{\frac{3}{2}}}{2 \pi \rho_{w} \gamma N_{i}}
\end(equation)
```
where: 
  - ``\alpha`` is a coefficient in the supersaturation balance equation. 
  - ``V`` is the updraft velocity. 
  - ``G`` is the diffusion of heat and moisture to the particles. 
  - ``\rho_w`` is the density of liquid water. 
  - ``\gamma`` is a coefficent in the superaturation balance equation. 
  - ``N_i`` is the mode particle density. 

The coefficient for the surface tension effects in the Köhler equilibrium equation is given in equation (5) of Abdul-Razzak and Ghan (1998):

```math
\begin{equation}
A \equiv \frac{2 \tau M_w}{\rho_w R T}
\end{equation}
```
where: 
  - ``\tau`` is the surface tension of water. 
  - ``M_w`` is the molar mass of water. 
  - ``\rho_w`` is the density of liquid water. 
  - ``R`` is the gas constant. 
  - ``T`` is the temperature. 

\alpha is given in equation (11) of Abdul-Razzak and Ghan (1998):
```math
\begin{equation}
\alpha = frac{g M_w L}{C_p R T^2} - frac{g M_{air}}{R T}
\end{equation}
```
where:
  - ``g`` is gravitational acceleration. 
  - ``M_w`` is the molar mass of water. 
  - ``L`` is the latent heat of water evaporation. 
  - ``C_p`` is the specific heat of air. 
  - ``R`` is the gas constant. 
  - ``T`` is the temperature. 
  - ``M_{air}`` is the molar mass of air. 

\gamma is given in equation (12) of Abdul-Razzak and Ghan (1998):
```math
\begin{equation}
\gamma = frac{R T}{P_s M_w} + frac{M_w L^2}{C_p P M_{air} T}
\end{equation}
```
where:
  - ``M_w`` is the molar mass of water. 
  - ``L`` is the latent heat of water evaporation. 
  - ``C_p`` is the specific heat of air. 
  - ``R`` is the gas constant. 
  - ``T`` is the temperature. 
  - ``M_{air}`` is the molar mass of air. 
  - ``P_s`` is the saturation pressure. 
  - ``P`` is the pressure. 
  

### Total Particles Activated 
The total particles activated represents the number of aerosol particles from the aerosol population that become nucleation points for the formation of cloud particles. This gives the total number of cloud particles formed from a given aerosol population. The total particles activated is given in equation (13) of Abdul-Razzak and Ghan (2000): 

```math
N = \sum_{i = 1}^{I} N_{i}\frac{1}{2}[1 - \mathrm{erf}(u_{i})]
```
where
  - ``N_{i}`` is the total number concentration.
  - ``u_{i}`` is a function. 

u_i is given in equation (15) of Abdul-Razzak and Ghan (2000):

```math
\begin{equation}
u_i \equiv \frac{2 \mathrm{ln}(S_{mi}/S_{max})}{3\sqrt{2} \mathrm{ln} \sigma_{i}}
\end{equation}
```
where: 
  - ``S_{mi}`` is the mode critical supersaturation.
  - ``S_{max}`` is the maximum supersaturation. 
  - ``\sigma_{i}`` is the geometric standard deviation for aerosol mode ``i``.


## Parametrization Implementation Example
This section gives an example of the implementation of this parametrization and how the total number of activated particles depends various input parameters. This implememtation example has an aerosol model with one mode comprised entirely of ammonium sulfate, matching the example given in Abdul-Razzak and Ghan (2000). The aerosol parameters inputted are: 
|    variable name     |         default value               | units      | 
|---------------|-----------------------------------------|-----------------
|``a_m``  | 5e-8               | ``m``          |
|``\sigma``  | 2.0       | ``-``          |
|``N``  | 100       | ``m^{-3}``          |
|``r``  | 1.0      | ``-``          |
|``\epsilon``  | 1.0       | ``-``          |
|``\phi``  | 1.0      | ``-``          |
|``M_a``  | 132.0      | ``kg mol^{-1}``          |
|``\nu``  | 3    | ``-``          |
|``\rho_a``  | 1770.0      | ``kgm^{-3}``          |
|``J``  | 1       | ``-``          |

The system parameters inputted are: 
|    variable name          |default value       | units             | 
|----------------------------|-----------------------------------------------------------|--------------------------|
|``R``      | 8.314  | J K^{-1} mol^{-1}  |
|``g``      | 9.8  | m s^{-2}  |    
|``T``      | 294.0  | T |  
|``P``      | 100000  | Pa  |  
|``P_{sat}``      | 1000000  | Pa  |  
|``\M_W``      | 0.01802  | kg mol^{-1}  |  
|``\rho_W``      | 1000  | kg m^{-3}  |
|``\M_{air}``      | 0.02896  | kg mol^{-1|  
|``C_p`` | 1005  | J kg^{-1} K{-1}  |    
|``L``      | 2.260  | J kg^{-1}  |         
|``\tau``      | 0.072  | N m^{-1}  |
|``G``      | TODO (calc from microphysics.jl)   | -  | 
|``V``      | 0.5  | m s^{-1}  |     

### Initial particle density on fraction of particles activated
TODO

### Updraft velocity on fraction of particles activated
TODO

### Mode radius on fraction of particles activated
TODO

### Soluble mass fraction on fraction of particles activated
TODO

<!-- The structure of the pipeline is as follows:
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




3: Total Number of Activated Aerosols


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
<<<<<<< HEAD
=======

>>>>>>> ca53418b9729cacde4ab66aeb21c1c8fced1751c -->
