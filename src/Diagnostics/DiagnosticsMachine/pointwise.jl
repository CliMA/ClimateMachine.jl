"""
    PointwiseDiagnostic

A diagnostic with the same dimensions as the original grid (DG or
interpolated). Mostly used to directly copy out prognostic or
auxiliary state variables.
"""
abstract type PointwiseDiagnostic <: DiagnosticVar end
function dv_PointwiseDiagnostic end

function dv_dg_points_length(
    ::ClimateMachineConfigType,
    ::Type{PointwiseDiagnostic},
)
    :(npoints)
end
function dv_dg_points_index(
    ::ClimateMachineConfigType,
    ::Type{PointwiseDiagnostic},
)
    :(ijk)
end

function dv_dg_elems_length(
    ::ClimateMachineConfigType,
    ::Type{PointwiseDiagnostic},
)
    :(nrealelem)
end
function dv_dg_elems_index(
    ::ClimateMachineConfigType,
    ::Type{PointwiseDiagnostic},
)
    :(e)
end

function dv_dg_dimnames(::ClimateMachineConfigType, ::Type{PointwiseDiagnostic})
    ("nodes", "elements")
end
function dv_dg_dimranges(
    ::ClimateMachineConfigType,
    ::Type{PointwiseDiagnostic},
)
    (:(1:npoints), :(1:nrealelem))
end

function dv_i_dimnames(::ClimateMachineConfigType, ::Type{PointwiseDiagnostic})
    :(tuple(collect(keys(dims))...))
end

function dv_op(
    ::ClimateMachineConfigType,
    ::Type{PointwiseDiagnostic},
    lhs,
    rhs,
)
    :($lhs = $rhs)
end

# Reduction for point-wise diagnostics would be a gather, but that will probably
# blow up memory. TODO.
function dv_reduce(
    ::ClimateMachineConfigType,
    ::Type{PointwiseDiagnostic},
    array_name,
)
    quote end
end

macro pointwise_diagnostic(impl, config_type, name, project = false)
    iex = quote
        $(generate_dv_interface(:PointwiseDiagnostic, config_type, name))
        $(generate_dv_function(:PointwiseDiagnostic, config_type, name, impl))
        $(generate_dv_project(:PointwiseDiagnostic, config_type, name, project))
    end
    esc(MacroTools.prewalk(unblock, iex))
end

"""
    @pointwise_diagnostic(
        impl,
        config_type,
        name,
        units,
        long_name,
        standard_name,
        project = false,
    )

Define `name` a point-wise diagnostic variable for `config_type`,
with the specified attributes and the given implementation. If
`project` is `true`, the variable will be projected along unit
vectors (for cubed shell topologies) after interpolation.

# Example

```julia
@pointwise_diagnostic(
    AtmosGCMConfigType,
    thv,
    "K",
    "virtual potential temperature",
    "virtual_potential_temperature",
) do (
    moisture::Union{EquilMoist, NonEquilMoist},
    atmos::AtmosModel,
    states::States,
    curr_time,
    cache,
)
    ts = get!(cache, :ts) do
        recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    end
    virtual_pottemp(ts)
end
```
"""
macro pointwise_diagnostic(
    impl,
    config_type,
    name,
    units,
    long_name,
    standard_name,
    project = false,
)
    iex = quote
        $(generate_dv_interface(
            :PointwiseDiagnostic,
            config_type,
            name,
            units,
            long_name,
            standard_name,
        ))
        $(generate_dv_function(:PointwiseDiagnostic, config_type, name, impl))
        $(generate_dv_project(:PointwiseDiagnostic, config_type, name, project))
    end
    esc(MacroTools.prewalk(unblock, iex))
end
