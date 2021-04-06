using ..Atmos
using ..ConfigTypes
using ..DiagnosticsMachine

@diagnostics_group(
    "AtmosLESDefault",      # name
    AtmosLESConfigType,     # configuration type
    Nothing,                # params type
    (_...) -> nothing,      # initialization function
    NoInterpolation,        # if/when to interpolate
    # various horizontal averages
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
    w_ht_sgs,
    # moisture related
    qt,
    ql,
    qi,
    qv,
    thv,
    thl,
    w_qt_sgs,
    # for variances and co-variances
    uu,
    vv,
    ww,
    www,
    eiei,
    wu,
    wv,
    wrho,
    wthd,
    wei,
    qtqt,
    thlthl,
    wqt,
    wql,
    wqi,
    wqv,
    wthv,
    wthl,
    qtthl,
    qtei,
    #cld_top, TODO
    #cld_base, TODO
    #cld_cover, TODO
    #lwp, TODO
    #rwp, TODO
)
