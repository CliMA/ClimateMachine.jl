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
function CplState(;
                  CplStateBlob
                     )

        return CplState(
                        CplStateBlob
                            )
end

function cpl_register(coupler::CplState,coupler_field_info)
end
