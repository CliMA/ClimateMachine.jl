module SingleStackUtils

export get_vars_from_nodal_stack, get_vars_from_element_stack

using OrderedCollections
using StaticArrays

using ..DGmethods
using ..DGmethods.Grids
using ..MPIStateArrays
using ..VariableTemplates

"""
    get_vars_from_nodal_stack(
        grid::DiscontinuousSpectralElementGrid{T, dim, N},
        Q::MPIStateArray,
        vars;
        vrange::UnitRange = 1:size(Q, 3),
        i::Int = 1,
        j::Int = 1,
        exclude::Vector{String} = String[],
    ) where {T, dim, N}

Return a dictionary whose keys are the `flattenednames()` of the variables
specified in `vars` (as returned by e.g. `vars_state_conservative()`), and
whose values are arrays of the values for that variable along the vertical
dimension in `Q`. Only a single element is expected in the horizontal as
this is intended for the single stack configuration and `i` and `j` identify
the horizontal nodal coordinates.

Variables listed in `exclude` are skipped.
"""
function get_vars_from_nodal_stack(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    vars;
    vrange::UnitRange = 1:size(Q, 3),
    i::Int = 1,
    j::Int = 1,
    exclude::Vector{String} = String[],
) where {T, dim, N}

    # extract grid information and bring `Q` to the host if needed
    FT = eltype(Q)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    nvertelem = grid.topology.stacksize
    host_Q = Array ∈ typeof(Q).parameters ? Q.realdata : Array(Q.realdata)

    # set up the dictionary to be returned
    var_names = flattenednames(vars)
    stack_vals = OrderedDict()
    num_vars = varsize(vars)
    vars_wanted = Int[]
    for vi in 1:num_vars
        if !(var_names[vi] in exclude)
            stack_vals[var_names[vi]] = FT[]
            push!(vars_wanted, vi)
        end
    end

    # extract values from `host_Q`
    for ev in vrange
        for k in 1:Nqk
            ijk = i + Nq * ((j - 1) + Nq * (k - 1))
            for v in vars_wanted
                push!(stack_vals[var_names[v]], host_Q[ijk, v, ev])
            end
        end
    end

    return stack_vals
end

"""
    get_vars_from_element_stack(
        grid::DiscontinuousSpectralElementGrid{T, dim, N},
        Q::MPIStateArray,
        vars;
        vrange::UnitRange = 1:size(Q, 3),
        exclude::Vector{String} = [],
    ) where {T, dim, N}
        exclude = [],
    )

Return an array of [`get_vars_from_nodal_stack()`](@ref)s whose dimensions
are the number of nodal points per element in the horizontal plane.

Variables listed in `exclude` are skipped.
"""
function get_vars_from_element_stack(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    vars;
    vrange::UnitRange = 1:size(Q, 3),
    exclude::Vector{String} = [],
) where {T, dim, N}
    Nq = N + 1
    return [
        get_vars_from_nodal_stack(
            grid,
            Q,
            vars,
            vrange = vrange,
            i = i,
            j = j,
            exclude = exclude,
        ) for i in 1:Nq, j in 1:Nq
    ]
end

end # module
