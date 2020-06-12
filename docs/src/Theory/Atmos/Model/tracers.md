# [Tracers](@id Tracers-docs)

!!! note

    Usage: Enable tracers using a keyword argument in the AtmosModel
    specification\
    - `tracers = NoTracer()`\
    - `tracers = NTracers{N, FT}(δ_χ)` where N is the number of tracers
    required.\
    FT is the float-type and $\delta_{\chi}$ is an SVector of diffusivity
    scaling coefficients

!!! note
    Hyperdiffusion is currently not supported with tracers. Laplacian
    diffusion coefficients may still be specified. (See above)


In `tracers.jl`, we define the equation sets governing tracer
dynamics. Specifically, we address the the equations of tracer motion in
conservation form,

```julia
export NoTracers, NTracers
```

### [Equations](@id tracer-eqns)
```math
\frac{\partial \rho\chi}{\partial t} +  \nabla \cdot ( \rho\chi u) = \nabla \cdot (\rho\delta_{D\chi}\mathrm{D_{T}}\nabla\chi) + \rho \mathrm{S}
```
where  $$\chi$$ represents the tracer species, $$\mathrm{S}$$ represents
the tracer source terms and $$\delta_{D\chi} \mathrm{D_{T}}$$ represents
the scaled turbulent eddy diffusivity for each tracer.  Currently a default
scaling of `1` is supported.  The equation as written above corresponds to
a single scalar tracer, but can be extended to include multiple independent
tracer species.

We first define an abstract tracer type, and define the default function
signatures. Two options are currently supported. [`NoTracers`](@ref
no-tracers), and [`NTracers`](@ref multiple-tracers).

## [Abstract Tracer Type](@id abstract-tracer-type)

Default stub functions for a generic tracer type are defined here.

```julia
abstract type TracerModel <: BalanceLaw end

vars_state_conservative(::TracerModel, FT) = @vars()
vars_state_gradient(::TracerModel, FT) = @vars()
vars_state_gradient_flux(::TracerModel, FT) = @vars()
vars_state_auxiliary(::TracerModel, FT) = @vars()

function atmos_init_aux!(
    ::TracerModel,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    nothing
end
function atmos_nodal_update_auxiliary_state!(
    ::TracerModel,
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
function flux_tracers!(
    ::TracerModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
function compute_gradient_flux!(
    ::TracerModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
function flux_second_order!(
    ::TracerModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    nothing
end
function compute_gradient_argument!(
    ::TracerModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
```

## [NoTracers](@id no-tracers)
The default tracer type in both the LES and GCM configurations is the no
tracer model. (This means no state variables for tracers are being carried
around). For the purposes of this model, moist variables are considered
separately in `moisture.jl`.

```@docs
ClimateMachine.Atmos.NoTracers
```

## [NTracers](@id multiple-tracers)
Allows users to specify an integer corresponding to the number of
tracers required.  Note that tracer naming is not currently supported,
i.e. the user must track each tracer variable based on its numerical
index. Sources can be added to each tracer based on the corresponding
numerical vector index. Initial profiles must be specified using the
`init_state_conservative!` hook at the experiment level.

```@docs
ClimateMachine.Atmos.NTracers{N,FT}
```
