using ..Atmos
using ..ConfigTypes
using ..DiagnosticsMachine

@diagnostics_group(
    "AtmosGCMDefault",          # name
    AtmosGCMConfigType,         # configuration type
    Nothing,                    # params type
    (_...) -> nothing,          # initialization function
    InterpolateAfterCollection, # if/when to interpolate
    # various pointwise variables
    u,
    v,
    w,
    rho,
    temp,
    pres,
    thd,
    et,
    ei,
    ht,
    hi,
    #vort, TODO
    # moisture related
    qt,
    ql,
    qv,
    qi,
    thv,
    thl,
)
