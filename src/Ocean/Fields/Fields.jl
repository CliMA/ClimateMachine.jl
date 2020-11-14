module Fields

export SpectralElementField, assemble, data

using CUDA
using Printf

using ClimateMachine.MPIStateArrays: MPIStateArray
using ..Domains: AbstractDomain

#####
##### SpectralElementField
#####

struct SpectralElementField{R, D, E}
    realdata::R
    domain::D
    elements::E
end

"""
    SpectralElementField(domain::RectangularDomain, state::MPIStateArray, variable_index::Int)

Returns a Cartesian `view` into `state.realdata[:, variable_index, :]`,
assuming that `state.realdata` lives on `RectangularDomain`.

`SpectralElementField.elements` is a three-dimensional array of `RectangularElements`.
"""
function SpectralElementField(domain::AbstractDomain, state::MPIStateArray, variable_index::Int)
    data = view(state.realdata, :, variable_index, :)
    return SpectralElementField(domain, data)
end

const SEF = SpectralElementField

Base.@propagate_inbounds Base.getindex(field::SEF, i, j, k) = field.elements[i, j, k]
Base.size(field::SEF) = size(field.elements)
Base.eltype(field::SEF) = eltype(field.realdata)

Base.maximum(f, field::SEF) = maximum([maximum(f, el) for el in field.elements])
Base.minimum(f, field::SEF) = minimum([minimum(f, el) for el in field.elements])

Base.maximum(field::SEF) = maximum([maximum(el) for el in field.elements])
Base.minimum(field::SEF) = minimum([minimum(el) for el in field.elements])

Base.show(io::IO, field::SEF{FT}) where {FT} = print(io, "SpectralElementField{$FT}")

#####
##### Domain-specific stuff
#####

include("rectangular_element.jl")
include("rectangular_spectral_element_fields.jl")


#####
##### CPU and GPU-friendly element assembly
#####

"""
    assemble(u::SpectralElementField)

Assemble `u.elements` into a single `element::eltype(u)`, averaging shared nodes.
"""
assemble(u::SpectralElementField) = assemble(u.elements)

"""
    assemble_data(u::SpectralElementField{<:CuArray})

Assemble the data in `u.elements` into a single `Array`, averaging shared nodes.
"""
function assemble(u::SpectralElementField{<:CuArray})

    domain = u.domain
    index = u.variable_index

    cpudata = Array(u.realdata)

    u_cpu = SpectralElementField(cpudata, domain)

    return assemble(u_cpu)
end

end # module
