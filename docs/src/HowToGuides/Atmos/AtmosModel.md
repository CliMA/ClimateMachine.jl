# [AtmosModel](@id AtmosModel-docs) 

This page provides a summary of a specific type of balance law within the ClimateMachine source code,
the `AtmosModel`. This documentation aims to introduce a user to the properties of the
`AtmosModel`, including the balance law equations and default model configurations. Both
LES and GCM configurations are included.

## Conservation Equations
The conservation equations specific to this implementation of `AtmosModel` are included below.


### Mass
```math
\frac{\partial \rho}{\partial t} + \nabla\cdot (\rho\vec{u}) = \rho \mathcal{\hat S}_{q_t}.
```

### Momentum
```math
\frac{(\partial \rho\vec{u})}{\partial t} + \nabla\cdot \left[ \rho\vec{u} \otimes \vec{u} + (p - p_r) \vec{I}_3\right] =
- (\rho - \rho_r) \nabla\Phi - 2\vec{\Omega} \times \rho\vec{u} \\
- \nabla\cdot (\rho \vec{\tau}) - \nabla\cdot\left( \vec{d}_{q_t} \otimes \rho\vec{u} \right) + \nabla\cdot \left( q_c w_c \vec{\hat k} \otimes \rho \vec{u} \right) + \rho \vec{F}_{\vec{u}}
```

### Energy
```math
 \frac{\partial(\rho e^\mathrm{tot})}{\partial t} + \nabla\cdot \left( (\rho e^\mathrm{tot} + p)\vec{u} \right)
 = -\nabla\cdot (\rho \vec{F}_R) - \nabla\cdot \bigl[\rho (\vec{J} + \vec{D})\bigr] + \rho Q  \\
  +\nabla\cdot \left(\rho W_c \vec{\hat k} \right)  - \nabla\cdot (\vec{u} \cdot \rho\vec{\tau)} %+ \rho \vec{u} \cdot \vec{F}_{\vec{u}} \\
   - \sum_{j\in\{v,l,i\}}(I_j + \Phi)  \rho C(q_j \rightarrow q_p) - M
```

### Moisture
```math
\frac{\partial (\rho q_t)}{\partial t} + \nabla\cdot (\rho q_t \vec{u})
= \rho \mathcal{S}_{q_t} - \nabla\cdot (\rho \vec{d}_{q_t}) + \nabla\cdot \bigl(\rho q_c w_c \vec{\hat k}  \bigr)
\equiv \rho \mathcal{\hat S}_{q_t}
```

### Precipitating Species
```math
\frac{\partial (\rho q_{p,i})}{\partial t} + \nabla\cdot \left[\rho q_{p,i} (\vec{u} - w_{p,i} \vec{\hat k}) \right] =\\
\rho \left[C(q_t \rightarrow q_{p,i}) + C(q_{p,k} \rightarrow q_{p,i}) \right] -\nabla\cdot (\rho \vec{d}_{q_{p, i}})
```

### Tracer Species
```math
\frac{(\partial \rho \chi_i)}{\partial t} + \nabla\cdot \left(\rho \chi_i \vec{u} \right) = \rho \mathcal{S}_{\chi_i} - \nabla\cdot (\rho \vec{d}_{\chi_i}) + \nabla\cdot (\rho \chi_{i} w_{\chi, i} \vec{\hat k})   
```

## Equation Abstractions

```math
\frac{\partial \vec{Y}}{\partial t} = - \nabla \cdot (\vec{F}_{nondiff} + \vec{F}_{diff} + \vec{F}_{rad} + \vec{F}_{precip}) + \vec{S}
```

### State Variables
```math
\vec{Y}=\left( \begin{array}{c}
\rho \\
\rho\vec{u} \\
\rho e^{\mathrm{tot}}\\
\rho q_k\\
\rho q_{p,i}\\
\rho \chi_j
\end{array}
\right).
```

### Fluxes

#### Nondiffusive Fluxes

```math
 \mathrm{F}_{nondiff}=\left( \begin{array}{c}
 \rho \vec{u} \\
 \rho \vec{u} \otimes \vec{u} + (p - p_r) \vec{I}_3 \\
 \rho e^{\mathrm{tot}} \vec{u} + p \vec{u}\\
 \rho q_k \vec{u}\\
 \rho q_{p,i} \vec{u} \\
 \rho \chi_j \vec{u}
\end{array}
\right).
```

#### Diffusive Fluxes

```math
\mathrm{F}_{diff}=\left( \begin{array}{c}
\rho\vec{d}_{q_t} \\
\rho\vec{\tau} + \rho\vec{d}_{q_t} \otimes \vec{u}\\
\vec{u} \cdot \rho\vec{\tau} + \rho (\vec{J} + \vec{D}) \\
\rho\vec{d}_{q_k}\\
\rho \vec{d}_{q_{p, i}}\\
\rho \vec{d}_{\chi_j}
\end{array}
\right).
```

#### Radiation Fluxes
```math
\mathrm{F}_{rad} =
\left( \begin{array}{c}
\vec{0} \\
\vec{0} \\
\rho \vec{F}_R \\
\vec{0} \\
\vec{0} \\
\vec{0}
\end{array}
\right)
```

### Fluxes of precipitating species
```math
\mathrm{F}_{fall} =
- \left( \begin{array}{c}
\rho q_{c} w_{c} \vec{\hat k}  \\
q_c w_c \vec{\hat k} \otimes \rho \vec{u}  \\
\rho W_c \vec{\hat k} \\
\rho q_{k} w_{k} \vec{\hat k}  \\
\rho q_{p,i} w_{p, i} \vec{\hat k} \\
\rho \chi_{i} w_{\chi, i} \vec{\hat k}
\end{array} \right)
```

#### [Sources]*(*@id atmos-sources)
```@docs
ClimateMachine.Atmos.source!
```

```math
\mathrm{S}(\vec{Y}, \nabla\vec{Y})=
 \left( \begin{array}{c}
 -\rho C(q_t \rightarrow q_p) \\
  -(\rho - \rho_r) \nabla\Phi - 2 \vec{\Omega} \times \rho\vec{u}  + \rho \vec{F}_{\vec{u}} \\
 \rho Q - \sum_{j\in\{v,l,i\}} (I_j + \Phi)  \rho C(q_j \rightarrow q_p) - M \\% + \rho \vec{u} \cdot \vec{F}_{\vec{u}}  \\
\rho C(q_p \rightarrow q_k) + \rho \sum_j \rho C(q_j \rightarrow q_k) \\
    \rho \sum_k C(q_k \rightarrow q_{p, i}) - \rho \sum_j C(q_{p, i} \rightarrow q_{p, j})\\
\rho \mathcal{S}_{\chi_i}
\end{array}
\right)
```

## Configurations
The struct `AtmosModel` defines a specific subtype of a balance law (i.e. conservation equations) specific to
atmospheric modelling. A complete description of a `model` is provided by the fields listed below. In this
implementation of the `AtmosModel` we concern ourselves with the conservative form of the compressible equations
of moist fluid motion given a set of initial, boundary and forcing(source) conditions.

### [LES Configuration](@id LESConfig) (with defaults)
Default field values for the LES `AtmosModel` definition are included below. Users are directed to the 
model subcomponent pages to view the possible options for each subcomponent.
```
    ::Type{AtmosLESConfigType},
    param_set::AbstractParameterSet;
    orientation::O = FlatOrientation(),
    ref_state::RS = HydrostaticState(
        LinearTemperatureProfile(
            FT(200),
            FT(280),
            FT(grav(param_set)) / FT(cp_d(param_set)),
        ),
        FT(0),
    ),
    turbulence::T = SmagorinskyLilly{FT}(0.21),
    hyperdiffusion::HD = NoHyperDiffusion(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (Gravity(), Coriolis(), GeostrophicForcing{FT}(7.62e-5, 0, 0)),
    tracers::TR = NoTracers(),
    boundarycondition::BC = AtmosBC(),
    init_state_conservative::IS = nothing,
    data_config::DC = nothing,
```

!!! note

    Most AtmosModel subcomponents are common to both LES / GCM configurations.
    Equation sets are written in vector-invariant form and solved in Cartesian coordinates.
    The component `orientation` determines whether the problem is solved in a `box (LES)` or a `sphere (GCM)`)


### [GCM Configuration](@id GCMConfig)(with defaults)
Default field values for the GCM `AtmosModel` definition are included below. Users are directed to the 
model subcomponent pages to view the possible options for each subcomponent. 
```
    ::Type{AtmosGCMConfigType},
    param_set::AbstractParameterSet;
    orientation::O = SphericalOrientation(),
    ref_state::RS = HydrostaticState(
        LinearTemperatureProfile(
            FT(200),
            FT(280),
            FT(grav(param_set)) / FT(cp_d(param_set)),
        ),
        FT(0),
    ),
    turbulence::T = SmagorinskyLilly{FT}(C_smag(param_set)),
    hyperdiffusion::HD = NoHyperDiffusion(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (Gravity(), Coriolis()),
    tracers::TR = NoTracers(),
    boundarycondition::BC = AtmosBC(),
    init_state_conservative::IS = nothing,
    data_config::DC = nothing,
```
