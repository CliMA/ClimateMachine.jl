using ..Orientations

"""
    single_stack_diagnostics(
        grid::DiscontinuousSpectralElementGrid,
        bl::BalanceLaw,
        t::Real,
        direction;
        kwargs...,
    )

# Arguments
 - `grid` the grid
 - `bl` the balance law
 - `t` time
 - `direction` direction
 - `kwargs` keyword arguments, passed to [`NodalStack`](@ref).

An array of nested NamedTuples, containing results of
 - `z` - altitude
 - `prog` - the prognostic state
 - `aux` - the auxiliary state
 - `∇flux` - the gradient-flux (diffusive) state
 - `hyperdiff` - the hyperdiffusive state

and all the nested NamedTuples, merged together,
from the `precompute` methods.
"""
function single_stack_diagnostics(
    grid::DiscontinuousSpectralElementGrid,
    bl::BalanceLaw,
    t::Real,
    direction;
    kwargs...,
)
    return [
        begin
            @unpack prog, aux, ∇flux, hyperdiff = local_states
            diffusive = ∇flux
            state = prog
            hyperdiffusive = hyperdiff

            _args_fx1 = (; state, aux, t, direction)
            _args_fx2 = (; state, aux, t, diffusive, hyperdiffusive)
            _args_src = (; state, aux, t, direction, diffusive)

            cache_fx1 = precompute(bl, _args_fx1, Flux{FirstOrder}())
            cache_fx2 = precompute(bl, _args_fx2, Flux{SecondOrder}())
            cache_src = precompute(bl, _args_src, Source())

            # cache_fx1, cache_fx2, and cache_src have overlapping
            # data, only need one copy, so merge:
            cache = merge(cache_fx1, cache_fx2, cache_src)

            z = altitude(bl, aux)

            # TODO: Use flatten_named_tuple = true, and flatten arrays
            flatten_named_tuple = false
            if flatten_named_tuple
                nt = (;
                    z = altitude(bl, aux),
                    prog = flattened_named_tuple(prog), # Vars -> flattened NamedTuples
                    aux = flattened_named_tuple(aux), # Vars -> flattened NamedTuples
                    ∇flux = flattened_named_tuple(∇flux), # Vars -> flattened NamedTuples
                    hyperdiff = flattened_named_tuple(hyperdiff), # Vars -> flattened NamedTuples
                    cache = cache,
                )
                # Flatten top level:
                flattened_named_tuple(nt)
            else
                nt = (;
                    z = altitude(bl, aux),
                    prog = prog,
                    aux = aux,
                    ∇flux = ∇flux,
                    hyperdiff = hyperdiff,
                    cache = cache,
                )
                nt
            end
            nt
        end for local_states in NodalStack(bl, grid; kwargs...)
    ]
end
