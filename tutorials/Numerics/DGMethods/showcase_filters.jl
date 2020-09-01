# # Filters
# In this tutorial we show the result of applying filters
# available in the CliMA codebase in a 1 dimensional box advection setup.
# See [Filters API](https://clima.github.io/ClimateMachine.jl/latest/APIs/Numerics/Meshes/Mesh/#Filters-1) for filters interface details.

using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "tutorials", "Numerics", "DGMethods", "Box1D.jl"))

output_dir = @__DIR__;
mkpath(output_dir);

# The unfiltered result of the box advection test for order 4 polynomial with
# central flux is
run_box1D(4, 0.0, 1.0, 1.0, joinpath(output_dir, "box_1D_4_no_filter.svg"))
# ![](box_1D_4_no_filter.svg)

# The unfiltered result of the box advection test for order 4 polynomial with
# Rusanov flux (aka upwinding for advection) is
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_no_filter_upwind.svg"),
    numerical_flux_first_order = RusanovNumericalFlux(),
)
# ![](box_1D_4_no_filter_upwind.svg)


# Below we show results for the same box advection test
# but using different filters.
#
# As seen in the results, when the TMAR filter is used mass is not necessarily
# conserved (mass increases are possible).

# `TMARFilter()` with central numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_tmar.svg");
    tmar_filter = true,
)
# ![](box_1D_4_tmar.svg)

# Running the TMAR filter with Rusanov the mass conservation since some of the
# are reduced, but mass is still not conserved.
# `TMARFilter()` with Rusanov numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_tmar_upwind.svg");
    tmar_filter = true,
    numerical_flux_first_order = RusanovNumericalFlux(),
)
# ![](box_1D_4_tmar_upwind.svg)

# `CutoffFilter(grid, Nc=1)` with central numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_cutoff_1.svg");
    cutoff_filter = true,
    cutoff_param = 1,
)
# ![](box_1D_4_cutoff_1.svg)

# `CutoffFilter(grid, Nc=3)` with central numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_cutoff_3.svg");
    cutoff_filter = true,
    cutoff_param = 3,
)
# ![](box_1D_4_cutoff_3.svg)

# `ExponentialFilter(grid, Nc=1, s=4)` with central numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_exp_1_4.svg");
    exp_filter = true,
    exp_param_1 = 1,
    exp_param_2 = 4,
)
# ![](box_1D_4_exp_1_4.svg)

# `ExponentialFilter(grid, Nc=1, s=8)` with central numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_exp_1_8.svg");
    exp_filter = true,
    exp_param_1 = 1,
    exp_param_2 = 8,
)
# ![](box_1D_4_exp_1_8.svg)

# `ExponentialFilter(grid, Nc=1, s=32)` with central numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_exp_1_32.svg");
    exp_filter = true,
    exp_param_1 = 1,
    exp_param_2 = 32,
)
# ![](box_1D_4_exp_1_32.svg)

# `BoydVandevenFilter(grid, Nc=1, s=4)` with central numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_boyd_1_4.svg");
    boyd_filter = true,
    boyd_param_1 = 1,
    boyd_param_2 = 4,
)
# ![](box_1D_4_boyd_1_4.svg)

# `BoydVandevenFilter(grid, Nc=1, s=8)` with central numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_boyd_1_8.svg");
    boyd_filter = true,
    boyd_param_1 = 1,
    boyd_param_2 = 8,
)
# ![](box_1D_4_boyd_1_8.svg)

# `BoydVandevenFilter(grid, Nc=1, s=32)` with central numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_boyd_1_32.svg");
    boyd_filter = true,
    boyd_param_1 = 1,
    boyd_param_2 = 32,
)
# ![](box_1D_4_boyd_1_32.svg)

# `ExponentialFilter(grid, Nc=1, s=8)` and `TMARFilter()` with central numerical
# flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_tmar_exp_1_8.svg");
    exp_filter = true,
    tmar_filter = true,
    exp_param_1 = 1,
    exp_param_2 = 8,
)
# ![](box_1D_4_tmar_exp_1_8.svg)

# `BoydVandevenFilter(grid, Nc=1, s=8)` and `TMARFilter()` with central
# numerical flux:
run_box1D(
    4,
    0.0,
    1.0,
    1.0,
    joinpath(output_dir, "box_1D_4_tmar_boyd_1_8.svg");
    boyd_filter = true,
    tmar_filter = true,
    boyd_param_1 = 1,
    boyd_param_2 = 8,
)
# ![](box_1D_4_tmar_boyd_1_8.svg)
