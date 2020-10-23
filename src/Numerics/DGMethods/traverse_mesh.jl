##### Iterators for traversing the mesh

abstract type TraversalPattern end

export Pointwise, traverse_mesh

"""
    Pointwise <: TraversalPattern

Traverse the mesh in order-agnostic pattern.
The order that the mesh is traversed should
have no impact on the traversal kernel.
"""
struct Pointwise <: TraversalPattern end

#####
##### traverse_mesh
#####

"""
    traverse_mesh(f!::F!, tp::TraversalPattern, grid, states...; execute = true)

Traverses the mesh, and calls function `f!`
in a mesh traversal pattern `tp` for the given
states. `execute = true` will call `wait` on the
result (a KernelAbstraction's `Event`).
"""
function traverse_mesh end

# traverse_mesh(f!::F!, tp::Pointwise, m, Q, dg; execute = true) where {F!} =
#     traverse_mesh(f!, tp, dg.grid, m,
#         (
#             (Q, Prognostic()),
#             (dg.state_auxiliary.data, Auxiliary()),
#             (dg.state_gradient_flux.data, GradientFlux()),
#         ); execute = execute)

"""
# Example: state manipulation (for example, non-conserving filters)
```
traverse_mesh(RadiallyOutward(), grid, Q, state_auxiliary, args...) do grid_local, state, aux
    Q.ρ = 1
end
```
"""
function traverse_mesh(f!::F!,
    tp::Pointwise,
    grid::DiscontinuousSpectralElementGrid,
    m::BalanceLaw,
    states...;
    execute = true,
    event = nothing) where {F!}
    elems = grid.topology.realelems
    device = arraytype(grid) <: Array ? CPU() : CUDADevice()
    if event == nothing
        event = Event(device)
    end
    Np = dofs_per_element(grid)
    event = kernel_traverse_mesh!(device, min(Np, 1024))(
        tp,
        m,
        Val(dimensionality(grid)),
        Val(polynomialorder(grid)),
        f!,
        elems,
        grid.activedofs,
        eltype(grid),
        states...;
        ndrange = dofs_per_element(grid) * length(elems),
        dependencies = (event,),
    )
    execute && wait(event)
    return event
end

