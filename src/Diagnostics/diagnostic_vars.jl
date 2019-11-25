"""
    Diagnostics variable template

Container for diagnostic variables of interest. Useful also for
post-processing.

"""

using CLIMA.VariableTemplates

function vars_diagnostic(FT)
  @vars begin
    # vertical coordinate
    z::FT
    # state and functions of state
    u::FT
    v::FT
    w::FT
    q_tot::FT
    e_tot::FT
    q_liq::FT
    thd::FT
    thl::FT
    thv::FT
    e_int::FT
    h_m::FT
    h_t::FT
    qt_sgs::FT
    ht_sgs::FT

    vert_eddy_mass_flx::FT
    vert_eddy_u_flx::FT
    vert_eddy_v_flx::FT
    vert_eddy_qt_flx::FT  #<w'q_tot'>
    vert_qt_flx::FT    #<w q_tot>
    vert_eddy_ql_flx::FT
    vert_eddy_qv_flx::FT
    vert_eddy_thd_flx::FT
    vert_eddy_thv_flx::FT
    vert_eddy_thl_flx::FT

    # variances
    uvariance::FT
    vvariance::FT
    wvariance::FT
    # skewness
    wskew::FT

    # turbulent kinetic energy
    TKE::FT
  end
end
num_diagnostic(FT) = varsize(vars_diagnostic(FT))
diagnostic_vars(array) = Vars{vars_diagnostic(eltype(array))}(array)

