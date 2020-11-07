module Fields

abstract type AbstractField end

"""
    field(domain, state, index)

Returns a view into the data in `state::MPIStateArray`
associated with `index` that represents a field on the `domain`.
"""
function field end

"""
    assemble(u::AbstractField)

Assemble `u.elements` into a single `element::eltype(u)`, averaging shared nodes.
"""
assemble(u::AbstractField) = assemble(u.elements)

end # module
