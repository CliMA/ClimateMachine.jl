mutable struct CplState{DT}
    CplStateBlob::DT
end

"""
    CplState(;)

Type for holding Coupler "state". This is the namespace through which coupled components
communicate. Its role is to provide a level of indirection so that components remain modular
and so that any data communication, interpolation, reindexing/unit conversions and filtering 
etc... can be embeded in the intermdediate coupling layer.

To start with we can just use a dictionary key and value table that holds labelled pointers to various fields.
A field is exported by one component and imported by one or more other components. Components
can select which fields are needed by using the Dict symbols.
"""
function CplState(coupler_fields...; CplStateBlob = Dict{Symbol, Array}())
    coupler = CplState(CplStateBlob)
    for field in coupler_fields
        register_cpl_field!(coupler, field)
    end
    return coupler
end

function register_cpl_field!(coupler::CplState, coupler_field_info)
    push!(coupler.CplStateBlob, coupler_field_info => [])
end

# Write to coupler
function put!(cplstate::CplState, key::Symbol, value)
    cplstate.CplStateBlob[key] = value
end

# Read from coupler
function get(cplstate::CplState, key::Symbol)
    return cplstate.CplStateBlob[key]
end
