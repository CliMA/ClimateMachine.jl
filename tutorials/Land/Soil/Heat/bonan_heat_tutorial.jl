# # Solving the heat equation in soil

# This tutorial shows how to use CliMA code to solve the heat
# equation in soil.
# For background on the heat equation in general,
# and how to solve it using CliMA code, please see the
# [`heat_equation.jl`](../../Heat/heat_equation.md)
# tutorial.

# The version of the heat equation we are solving here assumes no
# sources or sinks and no diffusion of liquid water. It takes the form

# ``
# \frac{∂ ρe_{int}}{∂ t} =  ∇ ⋅ κ(θ_l, θ_i; ν, ...) ∇T
# ``

# Here

# ``t`` is the time (s),

# ``z`` is the location in the vertical (m),

# ``ρe_{int}`` is the volumetric internal energy of the soil (J/m^3),

# ``T`` is the temperature of the soil (K),

# ``κ`` is the thermal conductivity (W/m/K),

# ``ϑ_l`` is the augmented volumetric liquid water fraction,

# ``θ_i`` is the volumetric ice fraction, and

# ``ν, ...`` denotes parameters relating to soil type, such as porosity.


# We will solve this equation in an effectively 1-d domain with ``z ∈ [-1,0]``,
# and with the following boundary and initial conditions:

# ``T(t=0, z) = 277.15^\circ K``

# ``T(t, z = 0) = 288.15^\circ K ``

# `` -κ ∇T(t, z = -1) = 0 ẑ``


# The temperature ``T`` and
# volumetric internal energy ``ρe_{int}`` are related as

# ``
# ρe_{int} = ρc_s (θ_l, θ_i; ν, ...) (T - T_0) - θ_i ρ_i LH_{f0}
# ``

# where

# ``ρc_s`` is the volumetric heat capacity of the soil (J/m^3/K),

# ``T_0`` is the freezing temperature of water,

# ``ρ_i`` is the density of ice (kg/m^3), and

# ``LH_{f0}`` is the latent heat of fusion at ``T_0``.

# In this tutorial, we will use a [`PrescribedWaterModel`](@ref
# ClimateMachine.Land.PrescribedWaterModel). This option allows
# the user to specify a function for the spatial and temporal
# behavior of `θ_i` and `θ_l`; it does not solve Richard's equation
# for the evolution of moisture. Please see the tutorials
# in the `Soil/Coupled/` folder or the `Soil/Water/`
# folder for information on solving
# Richard's equation, either coupled or uncoupled from the heat equation, respectively.



# # Import necessary modules

# External (non - CliMA) modules
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using Plots
using DelimitedFiles

# CliMA Parameters
using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0


# ClimateMachine modules
using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.SoilHeatParameterizations
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state
import ClimateMachine.DGMethods: calculate_dt
using ClimateMachine.ArtifactWrappers

# # Preliminary set-up

# Get the parameter set, which holds constants used across CliMA models.
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();
# Initialize and pick a floating point precision.
ClimateMachine.init()
FT = Float32;
# Load a function that will create an interpolation of the
# simulation output, to be used in plotting:
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(
    clima_dir,
    "tutorials",
    "Land",
    "Soil",
    "interpolation_helper.jl",
));
# # Determine soil parameters

# Below are the soil component fractions for various soil
# texture classes,  from Cosby et al. (1984) [1, 5].
# Note that these fractions are volumetric fractions, relative
# to other soil solids, i.e. not including pore space. These are denoted `ν_ss_i`; the CliMA
# Land Documentation uses the symbol `ν_i` to denote the volumetric fraction
# of a soil component `i` relative to the soil, including pore space.

ν_ss_silt_array =
    FT.(
        [5.0, 12.0, 32.0, 70.0, 39.0, 15.0, 56.0, 34.0, 6.0, 47.0, 20.0] ./
        100.0,
    )
ν_ss_quartz_array =
    FT.(
        [92.0, 82.0, 58.0, 17.0, 43.0, 58.0, 10.0, 32.0, 52.0, 6.0, 22.0] ./
        100.0,
    )
ν_ss_clay_array =
    FT.(
        [3.0, 6.0, 10.0, 13.0, 18.0, 27.0, 34.0, 34.0, 42.0, 47.0, 58.0] ./
        100.0,
    )
porosity_array =
    FT.([
        0.395,
        0.410,
        0.435,
        0.485,
        0.451,
        0.420,
        0.477,
        0.476,
        0.426,
        0.492,
        0.482,
    ]);


# The soil types that correspond to array elements above are, in order,
# sand, loamy sand, sandy loam, silty loam, loam, sandy clay loam,
# silty clay loam, clay loam, sandy clay, silty clay, and clay.

# Here we choose the soil type to be sandy.
# The soil column is uniform in space and time.
soil_type_index = 1
ν_ss_minerals =
    ν_ss_clay_array[soil_type_index] + ν_ss_silt_array[soil_type_index]
ν_ss_quartz = ν_ss_quartz_array[soil_type_index]
porosity = porosity_array[soil_type_index];


# This tutorial additionally compares the output of a ClimateMachine simulation with that
# of Supplemental Program 2, Chapter 5, of Bonan (2019) [1].
#  We found this useful as it
# allows us compare results from our code against a published version.


# The simulation code of [1] employs a formalism for the thermal
# conductivity `κ` based on Johansen, 1975, [2]. It assumes
# no organic matter, and only requires the volumetric
# fraction of soil solids for quartz and other minerals.
# ClimateMachine employs the formalism of Balland and Arp (2005)[3],
# which requires the
# fraction of soil solids for quartz, gravel,
# organic matter, and other minerals. Dai (2019)[4] found
# model [3] to better match
# measured soil properties across a range of soil types.

# To compare the output of the two simulations, we set the organic
# matter content and gravel content to zero in the CliMA model.
# The remaining soil components (quartz and other minerals) match between
# the two. We also run the simulation for relatively wet soil
# (water content at 80% of porosity). Under these conditions,
# the two formulations for `κ`, though taking different functional forms,
# are relatively consistent.
# The differences between models are important
# for soil with organic material and for soil that is relatively dry.
ν_ss_om = FT(0.0)
ν_ss_gravel = FT(0.0);


# We next calculate a few intermediate quantities needed for the
# determination of the thermal conductivity [3]. These include
# the conductivity of the solid material, the conductivity
# of saturated soil, and the conductivity of frozen saturated soil.

κ_quartz = FT(7.7) # W/m/K
κ_minerals = FT(2.5) # W/m/K
κ_om = FT(0.25) # W/m/K
κ_liq = FT(0.57) # W/m/K
κ_ice = FT(2.29); # W/m/K

# The particle density of soil solids in moisture-free soil
# is taken as a constant, across soil types, as in [1].
# This is a good estimate for organic material free soil. The user is referred to
# [3] for a more general expression.
ρp = FT(2700) # kg/m^3
κ_solid = k_solid(ν_ss_om, ν_ss_quartz, κ_quartz, κ_minerals, κ_om)
κ_sat_frozen = ksat_frozen(κ_solid, porosity, κ_ice)
κ_sat_unfrozen = ksat_unfrozen(κ_solid, porosity, κ_liq);

# The thermal conductivity of dry soil is also required, but this is
# calculated internally using the expression of [3].

# The volumetric specific heat of dry soil is chosen so as to match Bonan's simulation.
# The user could instead compute this using a volumetric fraction weighted average
# across soil components.
ρc_ds = FT((1 - porosity) * 1.926e06) # J/m^3/K

# Finally, we store the soil-specific parameters and functions
# in a place where they will be accessible to the model
# during integration.
soil_param_functions = SoilParamFunctions{FT}(
    porosity = porosity,
    ν_ss_gravel = ν_ss_gravel,
    ν_ss_om = ν_ss_om,
    ν_ss_quartz = ν_ss_quartz,
    ρc_ds = ρc_ds,
    ρp = ρp,
    κ_solid = κ_solid,
    κ_sat_unfrozen = κ_sat_unfrozen,
    κ_sat_frozen = κ_sat_frozen,
);


# # Initial and Boundary conditions
# We will be using a [`PrescribedWaterModel`](@ref
# ClimateMachine.Land.PrescribedWaterModel), where the user supplies the augmented
# liquid fraction and ice fraction as functions of space and time. Since we are not
# implementing phase changes, it makes sense to either have entirely liquid or
# frozen water. This tutorial shows liquid water.

# Because the two models for thermal conductivity agree well for wetter soil, we'll
# choose that here. However, the user could also explore how they differ by choosing
# drier soil.

# Please note that if the user uses a mix of liquid and frozen water, that they must
# ensure that the total water content does not exceed porosity.
prescribed_augmented_liquid_fraction = FT(porosity * 0.8)
prescribed_volumetric_ice_fraction = FT(0.0);


# Choose boundary and initial conditions for heat that will not lead to freezing of water:
heat_surface_state = (aux, t) -> eltype(aux)(288.15)
heat_bottom_flux = (aux, t) -> eltype(aux)(0.0)
T_init = (aux) -> eltype(aux)(275.15);


# We also need to define a function `init_soil!`, which
# initializes all of the prognostic variables (here, we
# only have `ρe_int`, the volumetric internal energy).
# The initialization is based on user-specified
# initial conditions. Note that the user provides initial
# conditions for heat based on the temperature - `init_soil!` also
# converts between `T` and `ρe_int`.

function init_soil!(land, state, aux, localgeo, time)
    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, time)
    θ_l = volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
    ρc_ds = land.soil.param_functions.ρc_ds
    ρc_s = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, land.param_set)

    state.soil.heat.ρe_int = volumetric_internal_energy(
        θ_i,
        ρc_s,
        land.soil.heat.initialT(aux),
        land.param_set,
    )
end;


# # Create the model structure
soil_water_model = PrescribedWaterModel(
    (aux, t) -> prescribed_augmented_liquid_fraction,
    (aux, t) -> prescribed_volumetric_ice_fraction,
);

# The boundary value problem in this case, with two spatial derivatives,
# requires a boundary condition at the top of the domain and the bottom.
# This gives four possible combinations of boundary conditions - top Dirichlet
# and bottom either Neumann or Dirichlet, or top Neumann and bottom either
# Dirichlet or Neumann. The user should set the unused fields to `nothing`
# to indicate that they do not want to supply a boundary condition of that type.
# For example, below we indicate that we are applying (and supplying!) a Dirichlet
# condition at the top of the domain, and a Neumann condition at the bottom.

soil_heat_model = SoilHeatModel(
    FT;
    initialT = T_init,
    dirichlet_bc = Dirichlet(
        surface_state = heat_surface_state,
        bottom_state = nothing,
    ),
    neumann_bc = Neumann(
        surface_flux = nothing,
        bottom_flux = heat_bottom_flux,
    ),
);

# The full soil model requires a heat model and a water model, as well as the
# soil parameter functions:
m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);

# The equations being solved in this tutorial have no sources or sinks:
sources = ();

# Finally, we create the `LandModel`. In more complex land models, this would
# include the canopy, carbon state of the soil, etc.
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil!,
);


# # Specify the numerical details
# These include the resolution, domain boundaries, integration time,
# Courant number, and ODE solver.

N_poly = 1
nelem_vert = 100

zmax = FT(0)
zmin = FT(-1)

driver_config = ClimateMachine.SingleStackConfiguration(
    "LandModel",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m;
    zmin = zmin,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);


# In this tutorial, we determine a timestep based on a Courant number (
# also called a Fourier number in the context of the heat equation).
# In short, we can use the parameters of the model (`κ` and `ρc_s`),
# along with with the size of
# elements of the grid used for discretizing the PDE, to estimate
# a natural timescale for heat transfer across a grid cell.
# Because we are using an explicit ODE solver, the timestep should
# be a fraction of this in order to resolve the dynamics.

# This allows us to automate, to a certain extent, choosing a value for
# the timestep, even as we switch between soil types.

function calculate_dt(dg, model::LandModel, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = DGMethods.courant(diffusive_courant, dg, model, Q, Δt, t, direction)
    return Courant_number / CFL
end

function diffusive_courant(
    m::LandModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    soil = m.soil
    ϑ_l, θ_i = get_water_content(soil.water, aux, state, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, soil.param_functions.porosity)
    κ_dry = k_dry(m.param_set, soil.param_functions)
    S_r = relative_saturation(θ_l, θ_i, soil.param_functions.porosity)
    kersten = kersten_number(θ_i, S_r, soil.param_functions)
    κ_sat = saturated_thermal_conductivity(
        θ_l,
        θ_i,
        soil.param_functions.κ_sat_unfrozen,
        soil.param_functions.κ_sat_frozen,
    )
    κ = thermal_conductivity(κ_dry, kersten, κ_sat)
    ρc_ds = soil.param_functions.ρc_ds
    ρc_s = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, m.param_set)
    return Δt * κ / (Δx * Δx * ρc_ds)
end


t0 = FT(0)
timeend = FT(60 * 60 * 3)
Courant_number = FT(0.5) # much bigger than this leads to domain errors

solver_config = ClimateMachine.SolverConfiguration(
    t0,
    timeend,
    driver_config;
    Courant_number = Courant_number,
    CFL_direction = VerticalDirection(),
);


# # Run the integration
ClimateMachine.invoke!(solver_config);


# # Plot results and comparison data from [1]
# We can pull out the `z` and `T` values on the DG grid, but these will be
# multi-valued at boundaries. We'll create an additional cartesian grid
# , create an interpolation of the
# DG output, and evaluate on this second grid.
# `T` and `z` are stored in aux:
aux = solver_config.dg.state_auxiliary;


# Specify interpolation grid:
zres = FT(0.02)
boundaries = [
    FT(0) FT(0) zmin
    FT(1) FT(1) zmax
]
resolution = (FT(2), FT(2), zres)
thegrid = solver_config.dg.grid
intrp_brck = create_interpolation_grid(boundaries, resolution, thegrid);


# Smooth output, and look at T vs z:
iaux = interpolate_variables([(aux)], intrp_brck)
iaux = iaux[1]
z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
iz = Array(iaux[:, z_ind, :][:])
z = Array(aux[:, z_ind, :][:])
T_ind = varsindex(vars_state(m, Auxiliary(), FT), :soil, :heat, :T)
iT = Array(iaux[:, T_ind, :][:])

plot(
    iT,
    iz,
    label = "ClimateMachine",
    ylabel = "z (m)",
    xlabel = "T (K)",
    title = "Heat transfer in sand",
)
plot!(T_init.(z), z, label = "Initial condition")
filename = "bonan_heat_data.csv"
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
bonan_dataset = ArtifactWrapper(
    joinpath(clima_dir, "tutorials", "Land", "Soil", "Artifacts.toml"),
    "bonan_soil_heat",
    ArtifactFile[ArtifactFile(
        url = "https://caltech.box.com/shared/static/99vm8q8tlyoulext6c35lnd3355tx6bu.csv",
        filename = filename,
    ),],
)
bonan_dataset_path = get_data_folder(bonan_dataset)
data = joinpath(bonan_dataset_path, filename)
ds_bonan = readdlm(data, ',')
bonan_T = reverse(ds_bonan[:, 2])
bonan_z = reverse(ds_bonan[:, 1])
bonan_T_continuous = Spline1D(bonan_z, bonan_T)
bonan_at_clima_z = bonan_T_continuous.(z)
plot!(bonan_at_clima_z, z, label = "Bonan simulation")
plot!(legend = :bottomleft)
savefig("thermal_conductivity_comparison.png")
# ![](thermal_conductivity_comparison.png)

# The plot shows that the temperature at the top of the
# soil is gradually increasing. This is because the surface
# temperature is held fixed at a value larger than
# the initial temperature. If we ran this for longer,
# we would see that the bottom of the domain would also
# increase in temperature because there is no heat
# leaving the bottom (due to zero heat flux specified in
# the boundary condition).

# # References
# [1] Bonan, G. Climate Change and Terrestrial Ecosystem Modeling (2019),
# Cambridge University Press

# [2] Johansen, O. 1975. Thermal conductivity of soils. Ph.D. thesis,
# Trondheim, Norway. Cold Regions Research and Engineering Laboratory
# Draft Translation 637, 1977, ADA 044002.

# [3] Balland, V., and P. A. Arp (2005), Modeling soil thermal
# conductivities over a wide range of conditions, J. Env. Eng. Sci., 4, 549–558.

# [4] Dai, Y., N. W. amd Hua Yuan, S. Zhang, W. Shangguan, S. Liu, X. Lu,
# and Y. Xin (2019a), Evaluation of soil thermal conductivity schemes for
# use in land surface modeling, J. Adv. Model. Earth Sys., 11, 3454–3473.

# [5] Cosby, B. J., Hornberger, G. M., Clapp, R. B., and Ginn, T. R. (1984).
# A statistical exploration of the relationships of soil moisture
# characteristics to the physical properties of soils. Water Resources
# Research, 20, 682–690.
