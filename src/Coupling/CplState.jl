struct CplState{CST}
        CplStateTuple::CST
end

"""
    CplState(;)

Type for holding Coupler "state". This is the namespace through which coupled components
communicate. Its role is to provide a level of indirection so that components remain modular
and so that any data communication, interpolation and filtering can be embeded in the
intermdediate coupling layer.

To start with we just use a named tuple that holds labelled pointers to variuos fields
we want to pass between components.
"""
function CplState(;
                  CplStateTuple
                     )

        return CplState(
                        CplStateTuple
                            )
end
