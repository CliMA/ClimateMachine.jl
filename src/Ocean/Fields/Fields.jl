module Fields

export SpectralElementField, assemble

using Printf

#####
##### SpectralElementField
#####

struct SpectralElementField{E, D}
    elements::E
    domain::D
end

function Base.collect(field::SpectralElementField)
    elements = [
        collect(field.elements[i, j, k])
        for
        i in 1:(field.domain.Ne.x),
        j in 1:(field.domain.Ne.y), k in 1:(field.domain.Ne.z)
    ]
    return SpectralElementField(elements, field)
end

Base.maximum(f, field::SpectralElementField) =
    maximum([maximum(f, el) for el in field.elements])
Base.minimum(f, field::SpectralElementField) =
    minimum([minimum(f, el) for el in field.elements])

Base.maximum(field::SpectralElementField) =
    maximum([maximum(el) for el in field.elements])
Base.minimum(field::SpectralElementField) =
    minimum([minimum(el) for el in field.elements])

Base.show(io::IO, field::SpectralElementField{FT}) where {FT} =
    print(io, "SpectralElementField{$FT}")

Base.@propagate_inbounds Base.getindex(field::SpectralElementField, i, j, k) =
    field.elements[i, j, k]

Base.size(field::SpectralElementField) = size(field.elements)

"""
    assemble(u::AbstractField)

Assemble `u.elements` into a single `element::eltype(u)`, averaging shared nodes.
"""
assemble(u::SpectralElementField) = assemble(u.elements)

#####
##### Domain-specific stuff
#####

include("rectangular_element.jl")
include("rectangular_spectral_element_fields.jl")

end # module
