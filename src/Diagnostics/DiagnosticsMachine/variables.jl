"""
    DiagnosticVar

The base type for all diagnostic variables.

Various kinds of diagnostic variables (such as `HorizontalAverage`)
are defined as abstract sub-types of this type and implement the group
of functions listed at the end for use by the diagnostics group code
generator.

A particular diagnostic variable is defined as a concrete sub-type of
one of the kinds of diagnostic variables. The type itself is generated as
are the group of functions below.

A diagnostic variable is always associated with a `ClimateMachine`
configuration type, so as to allow the same name to be used by different
configurations.
"""
abstract type DiagnosticVar end

"""
    dv_dg_points_length(::CT, ::Type{DVT}) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns an expression that evaluates to the length of the points
dimension of a DG array of the diagnostic variable type.
"""
function dv_dg_points_length end

"""
    dv_dg_points_index(::CT, ::Type{DVT}) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns an expression that, within a `@traverse_dg_grid`, evaluates
to an index into the points dimension of a DG array of the diagnostic
variable type.
"""
function dv_dg_points_index end

"""
    dv_dg_elems_length(::CT, ::Type{DVT}) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns an expression that evaluates to the length of the elements
dimension of a DG array of the diagnostic variable type.
"""
function dv_dg_elems_length end

"""
    dv_dg_elems_index(::CT, ::Type{DVT}) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns an expression that, within a `@traverse_dg_grid`, evaluates to
an index into the elements dimension of a DG array of the diagnostic
variable type.
"""
function dv_dg_elems_index end

"""
    dv_dg_dimnames(::CT, ::Type{DVT}) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns a tuple of the names of the dimensions of the diagnostic
variable type for when interpolation is _not_ used.
"""
function dv_dg_dimnames end

"""
    dv_dg_dimranges(::CT, ::Type{DVT}) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns a tuple of expressions that evaluate to the range of the
dimensions for the diagnostic variable type for when interpolation is
_not_ used.
"""
function dv_dg_dimranges end

"""
    dv_i_dimnames(::CT, ::Type{DVT}) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns a tuple of the names of the dimensions of the diagnostic
variable type or of an expression, for when interpolation is used.
"""
function dv_i_dimnames end

"""
    dv_op(::CT, ::Type{DVT}, lhs, rhs) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns an expression that, within a `@traverse_dg_grid`, evaluates to
an assignment of `rhs` to `lhs`, where `rhs` is the implementation of
the diagnostic variable and `lhs` is the appropriate location in the
array containing the computed diagnostic variable values.
"""
function dv_op end

"""
    dv_reduce(::CT, ::Type{DVT}, array_name) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

"""
function dv_reduce end

"""
    dv_name(::CT, ::DVT) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns the name of the diagnostic variable as a `String`. Generated.
"""
function dv_name end

"""
    dv_attrib(::CT, ::DVT) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns a `Dict` of diagnostic variable attributes, primarily for NetCDF.
Generated.
"""
function dv_attrib end

# Default method for variable attributes.
dv_attrib(::ClimateMachineConfigType, ::DiagnosticVar) = Dict()

"""
    dv_args(::CT, ::DVT) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Returns a tuple of `(arg_name, arg_type, slurp, default)` (as returned
by `MacroTools.splitarg` for the arguments specified by the
implementation of the diagnostic variable.
"""
function dv_args end

"""
    dv_scale(::CT, ::DVT) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

If scaling was specified for the diagnostic variable, return the
diagnostic variable with which the scaling should be done, otherwise
return `nothing`.
"""
function dv_scale end

# Default method for diagnostic variable scaling.
dv_scale(::ClimateMachineConfigType, ::DiagnosticVar) = nothing

"""
    dv_project(::CT, ::DVT) where {
        CT <: ClimateMachineConfigType,
        DVT <: DiagnosticVar,
    }

Return `true` if the specified diagnostic variable should be
projected after interpolation.
"""
function dv_project end

# Default method for diagnostic variable scaling.
dv_project(::ClimateMachineConfigType, ::DiagnosticVar) = false

####

# Generate a standardized type name from:
# - the configuration type,
# - the diagnostic variable kind, and
# - the diagnostic variable name.
function dv_type_name(dvtype, config_type, name)
    let uppers_in(s) =
            foldl((f, c) -> isuppercase(c) ? f * c : f, String(s), init = "")
        return uppers_in(config_type) *
               "_" *
               uppers_in(dvtype) *
               "_" *
               String(name)
    end
end

"""
    generate_dv_interface(
        dvtype,
        config_type,
        name,
        units,
        long_name,
        standard_name,
    )

Generate the type for a diagnostic variable, add an instance of the
type into `AllDiagnosticVars`, and generate `dv_name` and `dv_attrib`.
"""
function generate_dv_interface(
    dvtype,
    config_type,
    name,
    units = "",
    long_name = "",
    standard_name = "",
)
    dvtypname = Symbol(dv_type_name(dvtype, config_type, name))
    attrib_ex = quote end
    if any(a -> a != "", [units, long_name, standard_name])
        attrib_ex = quote
            dv_attrib(::$config_type, ::$dvtypname) = OrderedDict(
                "units" => $units,
                "long_name" => $long_name,
                "standard_name" => $standard_name,
            )
        end
    end
    quote
        struct $dvtypname <: $dvtype end
        DiagnosticsMachine.AllDiagnosticVars[$config_type][$(String(name))] =
            $dvtypname()
        dv_name(::$config_type, ::$dvtypname) = $(String(name))
        $(attrib_ex)
    end
end

"""
    generate_dv_function(dvtype, config_type, name, impl)

Generate `dv_args` for a diagnostic variable as well as the
implementation function: `dv_<dvtype>`, adding the configuration
type and the diagnostic variable type as the first two parameters
for dispatch.

The implementation _must_ be defined as:
```
f(
    [<component-name>::<component-type>,]
    bl::<balance-law-type>,
    states::States,
    curr_time::Float64,
    cache::Dict{Symbol, Any},
)
```
Where `<component-name>` is the name of the property within the balance
law type, `<component-type>` and `<balance-law-type>` are used for
dispatch, and `cache` may be used to store intermediate computations.
"""
function generate_dv_function(dvtype, config_type, name, impl)
    dvfun = Symbol("dv_", dvtype)
    dvtypname = Symbol(dv_type_name(dvtype, config_type, name))
    @capture(impl, ((args__,),) -> (body_)) ||
        @capture(impl, (args_) -> (body_)) ||
        error("Bad implementation for $(esc(names[1]))")
    split_fun_args = map(splitarg, args)
    fun_args = map(a -> :($(a[1])::$(a[2])), split_fun_args)
    quote
        function dv_args(::$config_type, ::$dvtypname)
            $split_fun_args
        end
        function $dvfun(::$config_type, ::$dvtypname, $(fun_args...))
            $body
        end
    end
end

"""
    generate_dv_scale(dvtype, config_type, name, scale)

Generate `dv_scale` for a diagnostic variable, returning the diagnostic
variable with which it is to be scaled.
"""
function generate_dv_scale(dvtype, config_type, name, scale)
    dvtypname = Symbol(dv_type_name(dvtype, config_type, name))
    sdv = nothing
    if !isnothing(scale)
        cfg_type_name = getfield(ConfigTypes, config_type)
        sdv = DiagnosticsMachine.AllDiagnosticVars[cfg_type_name][String(scale)]
    end
    quote
        dv_scale(::$config_type, ::$dvtypname) = $sdv
    end
end

"""
    generate_dv_project(dvtype, config_type, name, scale)

Generate `dv_project` for a diagnostic variable, returning `true` if
the diagnostic variable is to be projected.
"""
function generate_dv_project(dvtype, config_type, name, project)
    dvtypname = Symbol(dv_type_name(dvtype, config_type, name))
    quote
        dv_project(::$config_type, ::$dvtypname) = $project
    end
end

"""
    States

Composite of the various states, used as a parameter to diagnostic
variable implementations.
"""
struct States{PS, GFS, AS}
    prognostic::PS
    gradient_flux::GFS
    auxiliary::AS
end

# The various kinds of diagnostic variables as well as interfaces to
# create variables of these kinds.
include("pointwise.jl")
include("horizontal_average.jl")
