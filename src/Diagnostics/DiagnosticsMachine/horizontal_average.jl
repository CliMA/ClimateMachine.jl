"""
    HorizontalAverage

Targeted at AtmosLES configurations, produces a vector of length `z`
of averages for the horizontal planes in the domain, where `z` is the
interpolated grid height or the DG grid`s `x3id` over the nodes and
elements (duplicates are not removed).
"""
abstract type HorizontalAverage <: DiagnosticVar end
function dv_HorizontalAverage end

# TODO: replace these with a `dv_array_dims` that takes `nvars`
# and returns the dims for the array? Or create the array? Use
# `Array`? `similar`?

function dv_dg_points_length(
    ::ClimateMachineConfigType,
    ::Type{HorizontalAverage},
)
    :(Nqk)
end
function dv_dg_points_index(
    ::ClimateMachineConfigType,
    ::Type{HorizontalAverage},
)
    :(k)
end

function dv_dg_elems_length(
    ::ClimateMachineConfigType,
    ::Type{HorizontalAverage},
)
    :(nvertelem)
end
function dv_dg_elems_index(
    ::ClimateMachineConfigType,
    ::Type{HorizontalAverage},
)
    :(ev)
end

function dv_dg_dimnames(::ClimateMachineConfigType, ::Type{HorizontalAverage})
    ("z",)
end
function dv_dg_dimranges(::ClimateMachineConfigType, ::Type{HorizontalAverage})
    z = quote
        ijk_range = 1:Nqh:npoints
        e_range = 1:nvertelem
        reshape(grid.vgeo[ijk_range, grid.x3id, e_range], :)
    end
    (z,)
end

function dv_i_dimnames(::AtmosLESConfigType, ::Type{HorizontalAverage})
    ("z",)
end
function dv_i_dimnames(::AtmosGCMConfigType, ::Type{HorizontalAverage})
    ("level",)
end

function dv_op(::ClimateMachineConfigType, ::Type{HorizontalAverage}, lhs, rhs)
    :($lhs += MH * $rhs)
end
function dv_reduce(
    ::ClimateMachineConfigType,
    ::Type{HorizontalAverage},
    array_name,
)
    quote
        MPI.Reduce!($array_name, +, 0, mpicomm)
        if mpirank == 0
            for v in 1:size($array_name, 2)
                $(array_name)[:, v, :] ./= DiagnosticsMachine.Collected.ΣMH_z
            end
        end
    end
end

macro horizontal_average(impl, config_type, name, scale = nothing)
    iex = quote
        $(generate_dv_interface(:HorizontalAverage, config_type, name))
        $(generate_dv_function(:HorizontalAverage, config_type, name, impl))
        $(generate_dv_scale(:HorizontalAverage, config_type, name, scale))
    end
    esc(MacroTools.prewalk(unblock, iex))
end

"""
    @horizontal_average(
        impl,
        config_type,
        name,
        units,
        long_name,
        standard_name,
        scale = nothing,
    )

Define `name`, a horizontal average diagnostic variable for `config_type`
with the specified attributes and the given implementation. In order to
produce a density-weighted average, `scale` must be specified as the name
of the horizontal average diagnostic variable for density.

# Example

```julia
@horizontal_average(
    AtmosLESConfigType,
    w_ht_sgs,
    "kg kg^-1 m s^-1",
    "vertical sgs flux of total specific enthalpy",
    "",
    rho,
) do (atmos::AtmosModel, states::States, curr_time, cache)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    D_t = get!(cache, :D_t) do
        _, D_t, _ = turbulence_tensors(
            atmos,
            states.prognostic,
            states.gradient_flux,
            states.auxiliary,
            curr_time,
        )
        D_t
    end
    d_h_tot = -D_t .* states.gradient_flux.∇h_tot
    d_h_tot[end] * states.prognostic.ρ
end
```
"""
macro horizontal_average(
    impl,
    config_type,
    name,
    units,
    long_name,
    standard_name,
    scale = nothing,
)
    iex = quote
        $(generate_dv_interface(
            :HorizontalAverage,
            config_type,
            name,
            units,
            long_name,
            standard_name,
        ))
        $(generate_dv_function(:HorizontalAverage, config_type, name, impl))
        $(generate_dv_scale(:HorizontalAverage, config_type, name, scale))
    end
    esc(MacroTools.prewalk(unblock, iex))
end
