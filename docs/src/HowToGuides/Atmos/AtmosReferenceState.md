# Atmospheric temperature profiles

Here, we plot the atmospheric reference state profiles for a few different polynomial orders and number of elements.

```@example
using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
using Plots
include(joinpath(clima_dir, "docs", "plothelpers.jl"));
include(joinpath(clima_dir, "test", "Atmos", "Model", "get_atmos_ref_states.jl"));

function export_ref_state_plot(nelem_vert, N_poly)
    solver_config = get_atmos_ref_states(nelem_vert, N_poly, 0.5)
    z = get_z(solver_config.dg.grid)
    all_data = dict_of_nodal_states(solver_config, ["z"])
    T = all_data["ref_state.T"]
    ρ = all_data["ref_state.ρ"]
    p = all_data["ref_state.p"]
    ρe = all_data["ref_state.ρe"]
    p1 = plot(T, z./10^3, xlabel="Temperature [K]");
    p2 = plot(ρ, z./10^3, xlabel="Density [kg/m^3]");
    p3 = plot(p./10^3, z./10^3, xlabel="Pressure [kPa]");
    p4 = plot(ρe./10^3, z./10^3, xlabel="Total energy [kJ]");
    plot(p1, p2, p3, p4, layout=(1,4), ylabel="z [km]")
    savefig("N_poly_$(N_poly).png")
end

export_ref_state_plot(80, 1)
export_ref_state_plot(40, 2)
export_ref_state_plot(20, 4)
```
## Polynomial order 1, 80 elements
![](N_poly_1.png)

## Polynomial order 2, 40 elements
![](N_poly_2.png)

## Polynomial order 4, 20 elements
![](N_poly_4.png)
