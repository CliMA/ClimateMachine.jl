module SingleStackUtils

export get_vars_from_nodal_stack,
    get_vars_from_element_stack,
    get_horizontal_variance,
    get_horizontal_mean,
    reduce_nodal_stack,
    reduce_element_stack,
    horizontally_average!,
    dict_of_nodal_states,
    NodalStack,
    single_stack_diagnostics

using OrderedCollections
using UnPack
using StaticArrays
import KernelAbstractions: CPU

using ..BalanceLaws
using ..DGMethods
using ..DGMethods.Grids
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
        interp = false,
    ) where {T, dim, N}

Return a dictionary whose keys are the `flattenednames()` of the variables
specified in `vars` (as returned by e.g. `vars_state`), and
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
    interp = false,
) where {T, dim, N}

    # extract grid information and bring `Q` to the host if needed
    FT = eltype(Q)
    Nq = N .+ 1
    # Code assumes the same polynomial order in all horizontal directions
    @inbounds begin
        Nq1 = Nq[1]
        Nq2 = Nq[2]
        Nqk = dim == 2 ? 1 : Nq[dim]
    end
    Np = dofs_per_element(grid)
    state_data = array_device(Q) isa CPU ? Q.realdata : Array(Q.realdata)

    # set up the dictionary to be returned
    var_names = flattenednames(vars)
    stack_vals = OrderedDict()
    num_vars = varsize(vars)
    vars_wanted = Int[]
    @inbounds for vi in 1:num_vars
        if !(var_names[vi] in exclude)
            stack_vals[var_names[vi]] = FT[]
            push!(vars_wanted, vi)
        end
    end
    elemtobndy = convert(Array, grid.elemtobndy)
    vmap⁻ = convert(Array, grid.vmap⁻)
    vmap⁺ = convert(Array, grid.vmap⁺)
    vgeo = convert(Array, grid.vgeo)
    # extract values from `state_data`
    @inbounds for ev in vrange, k in 1:Nqk, v in vars_wanted
        if interp && k == 1 && elemtobndy[5, ev] == 0
            # Get face degree of freedom number
            n = i + Nq1 * ((j - 1))
            # get the element numbers
            ev⁻ = ev
            # Get neighboring id data
            id⁻, id⁺ = vmap⁻[n, 5, ev⁻], vmap⁺[n, 5, ev⁻]
            ev⁺ = ((id⁺ - 1) ÷ Np) + 1
            # get the volume degree of freedom numbers
            vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1

            J⁻, J⁺ = vgeo[vid⁻, Grids._M, ev⁻], vgeo[vid⁺, Grids._M, ev⁺]
            state_local = J⁻ * state_data[vid⁻, v, ev⁻]
            state_local += J⁺ * state_data[vid⁺, v, ev⁺]
            state_local /= (J⁻ + J⁺)
            push!(stack_vals[var_names[v]], state_local)
        elseif interp && k == Nqk && elemtobndy[6, ev] == 0
            # Get face degree of freedom number
            n = i + Nq1 * ((j - 1))
            # get the element numbers
            ev⁻ = ev
            # Get neighboring id data
            id⁻, id⁺ = vmap⁻[n, 6, ev⁻], vmap⁺[n, 6, ev⁻]
            # periodic and need to handle this point (otherwise handled above)
            if id⁺ == id⁻
                vid⁻ = ((id⁻ - 1) % Np) + 1

                state_local = state_data[vid⁻, v, ev⁻]
                push!(stack_vals[var_names[v]], state_local)
            end
        else
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            state_local = state_data[ijk, v, ev]
            push!(stack_vals[var_names[v]], state_local)
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
        exclude::Vector{String} = String[],
        interp = false,
    ) where {T, dim, N}

Return an array of [`get_vars_from_nodal_stack()`](@ref)s whose dimensions
are the number of nodal points per element in the horizontal plane.

Variables listed in `exclude` are skipped.
"""
function get_vars_from_element_stack(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    vars;
    vrange::UnitRange = 1:size(Q, 3),
    exclude::Vector{String} = String[],
    interp = false,
) where {T, dim, N}

    Nq = N .+ 1
    @inbounds Nq1 = Nq[1]
    @inbounds Nq2 = Nq[2]

    return [
        get_vars_from_nodal_stack(
            grid,
            Q,
            vars,
            vrange = vrange,
            i = i,
            j = j,
            exclude = exclude,
            interp = interp,
        ) for i in 1:Nq1, j in 1:Nq2
    ]
end

"""
    get_horizontal_mean(
        grid::DiscontinuousSpectralElementGrid{T, dim, N},
        Q::MPIStateArray,
        vars;
        vrange::UnitRange = 1:size(Q, 3),
        exclude::Vector{String} = String[],
        interp = false,
    ) where {T, dim, N}

Return a dictionary whose keys are the `flattenednames()` of the variables
specified in `vars` (as returned by e.g. `vars_state`), and
whose values are arrays of the horizontal averages for that variable along
the vertical dimension in `Q`. Only a single element is expected in the
horizontal as this is intended for the single stack configuration.

Variables listed in `exclude` are skipped.
"""
function get_horizontal_mean(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    vars;
    vrange::UnitRange = 1:size(Q, 3),
    exclude::Vector{String} = String[],
    interp = false,
) where {T, dim, N}

    Nq = N .+ 1
    @inbounds Nq1 = Nq[1]
    @inbounds Nq2 = Nq[2]

    vars_avg = OrderedDict()
    vars_sq = OrderedDict()
    for i in 1:Nq1
        for j in 1:Nq2
            vars_nodal = get_vars_from_nodal_stack(
                grid,
                Q,
                vars,
                vrange = vrange,
                i = i,
                j = j,
                exclude = exclude,
                interp = interp,
            )
            vars_avg = merge(+, vars_avg, vars_nodal)
        end
    end
    map!(x -> x ./ Nq1 / Nq1, values(vars_avg))
    return vars_avg
end

"""
    get_horizontal_variance(
        grid::DiscontinuousSpectralElementGrid{T, dim, N},
        Q::MPIStateArray,
        vars;
        vrange::UnitRange = 1:size(Q, 3),
        exclude::Vector{String} = String[],
        interp = false,
    ) where {T, dim, N}

Return a dictionary whose keys are the `flattenednames()` of the variables
specified in `vars` (as returned by e.g. `vars_state`), and
whose values are arrays of the horizontal variance for that variable along
the vertical dimension in `Q`. Only a single element is expected in the
horizontal as this is intended for the single stack configuration.

Variables listed in `exclude` are skipped.
"""
function get_horizontal_variance(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    vars;
    vrange::UnitRange = 1:size(Q, 3),
    exclude::Vector{String} = String[],
    interp = false,
) where {T, dim, N}

    Nq = N .+ 1
    @inbounds Nq1 = Nq[1]
    @inbounds Nq2 = Nq[2]

    vars_avg = OrderedDict()
    vars_sq = OrderedDict()
    for i in 1:Nq1
        for j in 1:Nq2
            vars_nodal = get_vars_from_nodal_stack(
                grid,
                Q,
                vars,
                vrange = vrange,
                i = i,
                j = j,
                exclude = exclude,
                interp = interp,
            )
            vars_nodal_sq = OrderedDict(vars_nodal)
            map!(x -> x .^ 2, values(vars_nodal_sq))
            vars_avg = merge(+, vars_avg, vars_nodal)
            vars_sq = merge(+, vars_sq, vars_nodal_sq)
        end
    end
    map!(x -> (x ./ Nq1 / Nq1) .^ 2, values(vars_avg))
    map!(x -> x ./ Nq1 / Nq1, values(vars_sq))
    vars_var = merge(-, vars_sq, vars_avg)
    return vars_var
end

"""
    reduce_nodal_stack(
        op::Function,
        grid::DiscontinuousSpectralElementGrid{T, dim, N},
        Q::MPIStateArray,
        vars::NamedTuple,
        var::String;
        vrange::UnitRange = 1:size(Q, 3),
    ) where {T, dim, N}

Reduce `var` from `vars` within `Q` over all nodal points in the specified
`vrange` of elements with `op`. Return a tuple `(result, z)` where `result` is
the final value returned by `op` and `z` is the index within `vrange` where the
`result` was determined.
"""
function reduce_nodal_stack(
    op::Function,
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    vars::Type,
    var::String;
    vrange::UnitRange = 1:size(Q, 3),
    i::Int = 1,
    j::Int = 1,
) where {T, dim, N}

    Nq = N .+ 1
    @inbounds begin
        Nq1 = Nq[1]
        Nq2 = Nq[2]
        Nqk = dim == 2 ? 1 : Nq[dim]
    end

    var_names = flattenednames(vars)
    var_ind = findfirst(s -> s == var, var_names)
    if var_ind === nothing
        return
    end

    state_data = array_device(Q) isa CPU ? Q.realdata : Array(Q.realdata)
    z = vrange.start
    result = state_data[i + Nq1 * (j - 1), var_ind, z]
    for ev in vrange
        for k in 1:Nqk
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            new_result = op(result, state_data[ijk, var_ind, ev])
            if !isequal(new_result, result)
                result = new_result
                z = ev
            end
        end
    end

    return (result, z)
end

"""
    reduce_element_stack(
        op::Function,
        grid::DiscontinuousSpectralElementGrid{T, dim, N},
        Q::MPIStateArray,
        vars::NamedTuple,
        var::String;
        vrange::UnitRange = 1:size(Q, 3),
    ) where {T, dim, N}

Reduce `var` from `vars` within `Q` over all nodal points in the specified
`vrange` of elements with `op`. Return a tuple `(result, z)` where `result` is
the final value returned by `op` and `z` is the index within `vrange` where the
`result` was determined.
"""
function reduce_element_stack(
    op::Function,
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    vars::Type,
    var::String;
    vrange::UnitRange = 1:size(Q, 3),
) where {T, dim, N}

    Nq = N .+ 1
    @inbounds Nq1 = Nq[1]
    @inbounds Nq2 = Nq[2]

    return [
        reduce_nodal_stack(
            op,
            grid,
            Q,
            vars,
            var,
            vrange = vrange,
            i = i,
            j = j,
        ) for i in 1:Nq1, j in 1:Nq2
    ]
end

"""
    horizontally_average!(
        grid::DiscontinuousSpectralElementGrid{T, dim, N},
        Q::MPIStateArray,
        i_vars,
    ) where {T, dim, N}

Horizontally average variables, from variable
indexes `i_vars`, in `MPIStateArray` `Q`.

!!! note
    These are not proper horizontal averages-- the main
    purpose of this method is to ensure that there are
    no horizontal fluxes for a single stack configuration.
"""
function horizontally_average!(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    i_vars,
) where {T, dim, N}

    Nq = N .+ 1
    @inbounds begin
        Nq1 = Nq[1]
        Nq2 = Nq[2]
        Nqk = dim == 2 ? 1 : Nq[dim]
    end

    ArrType = typeof(Q.data)
    state_data = array_device(Q) isa CPU ? Q.realdata : Array(Q.realdata)

    for ev in 1:size(state_data, 3), k in 1:Nqk, i_v in i_vars
        Q_sum = 0
        for i in 1:Nq1, j in 1:Nq2
            Q_sum += state_data[i + Nq1 * ((j - 1) + Nq2 * (k - 1)), i_v, ev]
        end
        Q_ave = Q_sum / (Nq1 * Nq2)
        for i in 1:Nq1, j in 1:Nq2
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            state_data[ijk, i_v, ev] = Q_ave
        end
    end
    Q.realdata .= ArrType(state_data)
end

get_data(solver_config, ::Prognostic) = solver_config.Q
get_data(solver_config, ::Auxiliary) = solver_config.dg.state_auxiliary
get_data(solver_config, ::GradientFlux) = solver_config.dg.state_gradient_flux

"""
    dict_of_nodal_states(
        solver_config,
        state_types = (Prognostic(), Auxiliary());
        aux_excludes = [],
        interp = false,
        )

A dictionary of single stack prognostic and auxiliary
variables at the `i=1`,`j=1` node given
 - `solver_config` a `SolverConfiguration`
 - `aux_excludes` a vector of strings containing the
    variables to exclude from the auxiliary state.
"""
function dict_of_nodal_states(
    solver_config,
    state_types = (Prognostic(), Auxiliary());
    aux_excludes = String[],
    interp = false,
)
    FT = eltype(solver_config.Q)
    all_state_vars = []
    for st in state_types
        state_vars = get_vars_from_nodal_stack(
            solver_config.dg.grid,
            get_data(solver_config, st),
            vars_state(solver_config.dg.balance_law, st, FT),
            exclude = st isa Auxiliary ? aux_excludes : String[],
            interp = interp,
        )
        push!(all_state_vars, state_vars...)
    end
    return OrderedDict(all_state_vars...)
end

# A container for holding various
# global or point-wise states:
struct States{P, A, D, HD}
    prog::P
    aux::A
    diffusive::D
    hyperdiffusive::HD
end

"""
    NodalStack(
            bl::BalanceLaw,
            grid::DiscontinuousSpectralElementGrid,
            prognostic,
            auxiliary,
            diffusive,
            hyperdiffusive;
            i = 1,
            j = 1,
            interp = true,
        )

A struct whose `iterate(::NodalStack)` traverses
the nodal stack and returns a NamedTuple of
point-wise fields (`Vars`).

# Example
```julia
for state_local in NodalStack(
        bl,
        grid,
        prognostic, # global field along nodal stack
        auxiliary,
        diffusive,
        hyperdiffusive
    )
prog = state_local.prog # point-wise field along nodal stack
end
```

## TODO: Make `prognostic`, `auxiliary`, `diffusive`, `hyperdiffusive` optional

# Arguments
 - `bl` the balance law
 - `grid` the discontinuous spectral element grid
 - `prognostic` the global prognostic state
 - `auxiliary` the global auxiliary state
 - `diffusive` the global diffusive state (gradient-flux)
 - `hyperdiffusive` the global hyperdiffusive state
 - `i,j` the `i,j`'th nodal stack (in the horizontal directions)
 - `interp` a bool indicating whether to
    interpolate the duplicate Gauss-Lebotto
    points at the element faces.

!!! warn
    Before iterating, the data is transferred from the
    device (GPU) to the host (CPU), as this is intended
    for debugging / diagnostics usage.
"""
struct NodalStack{N, BL, G, S, VR, TI, TJ, CI, IN}
    bl::BL
    grid::G
    states::S
    vrange::VR
    i::TI
    j::TJ
    cart_ind::CI
    interp::IN
    function NodalStack(
        bl::BalanceLaw,
        grid::DiscontinuousSpectralElementGrid;
        prognostic,
        auxiliary,
        diffusive,
        hyperdiffusive,
        i = 1,
        j = 1,
        interp = true,
    )
        states = States(prognostic, auxiliary, diffusive, hyperdiffusive)
        vrange = 1:size(prognostic, 3)
        grid_info = basic_grid_info(grid)
        @unpack Nqk = grid_info
        if polynomialorders(grid)[end] == 0
            interp = false
        end
        # Store cartesian indices, so we can map the iter_state
        # to the cartesian space `Q[i, var, j]`
        if interp
            cart_ind = CartesianIndices(((Nqk - 1), size(prognostic, 3)))
        else
            cart_ind = CartesianIndices((Nqk, size(prognostic, 3)))
        end
        args = (bl, grid, states, vrange, i, j, cart_ind, interp)
        BL, G, S, VR, TI, TJ, CI, IN = typeof.(args)
        if interp
            len = size(prognostic, 3) * (Nqk - 1) + 1
        else
            len = size(prognostic, 3) * Nqk
        end
        new{len, BL, G, S, VR, TI, TJ, CI, IN}(args...)
    end
end

Base.length(gs::NodalStack{N}) where {N} = N

# Helper function
get_state(v, state, vid⁻, vid⁺, ev⁻, ev⁺, J⁻, J⁺) =
    (J⁻ * state[vid⁻, v, ev⁻] + J⁺ * state[vid⁺, v, ev⁺]) / (J⁻ + J⁺)

to_cpu(state) =
    array_device(state) isa CPU ? state.realdata : Array(state.realdata)

function interp_top(state, args, n_vars, bl, st)
    vs = Vars{vars_state(bl, st, eltype(state))}
    if n_vars ≠ 0
        return vs(map(v -> get_state(v, state, args...), 1:n_vars))
    else
        return nothing
    end
end

function interp_bot(state, vid⁻, ev⁻, n_vars, bl, st)
    vs = Vars{vars_state(bl, st, eltype(state))}
    if n_vars ≠ 0
        return vs(map(v -> state[vid⁻, v, ev⁻], 1:n_vars))
    else
        return nothing
    end
end

function no_interp(state, ijk, ev, n_vars, bl, st)
    vs = Vars{vars_state(bl, st, eltype(state))}
    if n_vars ≠ 0
        return vs(map(v -> state[ijk, v, ev], 1:n_vars))
    else
        return nothing
    end
end

function Base.iterate(gs::NodalStack, iter_state = 1)
    iter_state > length(gs) && return nothing

    # extract grid information
    grid = gs.grid
    FT = eltype(grid)
    grid_info = basic_grid_info(grid)
    @unpack N, Nq, Np, Nqk = grid_info
    @inbounds Nq1, Nq2 = Nq[1], Nq[2]
    states = gs.states
    bl = gs.bl
    interp = gs.interp

    # bring `Q` to the host if needed

    prognostic = to_cpu(states.prog)
    auxiliary = to_cpu(states.aux)
    diffusive = to_cpu(states.diffusive)
    hyperdiffusive = to_cpu(states.hyperdiffusive)

    n_vars_prog = size(prognostic, 2)
    n_vars_aux = size(auxiliary, 2)
    n_vars_diff = size(diffusive, 2)
    n_vars_hd = size(hyperdiffusive, 2)

    elemtobndy = convert(Array, grid.elemtobndy)
    vmap⁻ = convert(Array, grid.vmap⁻)
    vmap⁺ = convert(Array, grid.vmap⁺)
    vgeo = convert(Array, grid.vgeo)
    i, j = gs.i, gs.j
    if iter_state == length(gs)
        ijk_cart = (Nqk, size(states.prog, 3))
    else
        ijk_cart = Tuple(gs.cart_ind[iter_state])
    end
    ev = ijk_cart[2]
    k = ijk_cart[1]
    iter_state⁺ = iter_state + 1
    if interp && k == 1 && elemtobndy[5, ev] == 0
        # Get face degree of freedom number
        n = i + Nq1 * ((j - 1))
        # get the element numbers
        ev⁻ = ev
        # Get neighboring id data
        id⁻, id⁺ = vmap⁻[n, 5, ev⁻], vmap⁺[n, 5, ev⁻]
        ev⁺ = ((id⁺ - 1) ÷ Np) + 1
        # get the volume degree of freedom numbers
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1
        J⁻, J⁺ = vgeo[vid⁻, Grids._M, ev⁻], vgeo[vid⁺, Grids._M, ev⁺]

        args = (vid⁻, vid⁺, ev⁻, ev⁺, J⁻, J⁺)
#! format: off
        prog = interp_top(prognostic, args, n_vars_prog, bl, Prognostic())
        aux = interp_top(auxiliary, args, n_vars_aux, bl, Auxiliary())
        ∇flux = interp_top(diffusive, args, n_vars_diff, bl, GradientFlux())
        hyperdiff = interp_top(hyperdiffusive, args, n_vars_hd, bl, Hyperdiffusive())
#! format: on

        return ((; prog, aux, ∇flux, hyperdiff), iter_state⁺)

    elseif interp && k == Nqk && elemtobndy[6, ev] == 0
        # Get face degree of freedom number
        n = i + Nq1 * ((j - 1))
        # get the element numbers
        ev⁻ = ev
        # Get neighboring id data
        id⁻, id⁺ = vmap⁻[n, 6, ev⁻], vmap⁺[n, 6, ev⁻]
        # periodic and need to handle this point (otherwise handled above)
        if id⁺ == id⁻
            vid⁻ = ((id⁻ - 1) % Np) + 1

#! format: off
            prog = interp_bot(prognostic, vid⁻, ev⁻, n_vars_prog, bl, Prognostic())
            aux = interp_bot(auxiliary, vid⁻, ev⁻, n_vars_aux, bl, Auxiliary())
            ∇flux = interp_bot(diffusive, vid⁻, ev⁻, n_vars_diff, bl, GradientFlux())
            hyperdiff = interp_bot( hyperdiffusive, vid⁻, ev⁻, n_vars_hd, bl, Hyperdiffusive())
#! format: on

            return ((; prog, aux, ∇flux, hyperdiff), iter_state⁺)
        else
            error("uncaught case in iterate(::NodalStack)")
        end
    else
        ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

#! format: off
        prog = no_interp(prognostic, ijk, ev, n_vars_prog, bl, Prognostic())
        aux = no_interp(auxiliary, ijk, ev, n_vars_aux, bl, Auxiliary())
        ∇flux = no_interp(diffusive, ijk, ev, n_vars_diff, bl, GradientFlux())
        hyperdiff = no_interp(hyperdiffusive, ijk, ev, n_vars_hd, bl, Hyperdiffusive())
#! format: on

        return ((; prog, aux, ∇flux, hyperdiff), iter_state⁺)
    end
end

include("single_stack_diagnostics.jl")

end # module
