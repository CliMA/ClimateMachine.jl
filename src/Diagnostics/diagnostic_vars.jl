
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
    thd::FT                # θ
    thl::FT                # θ_liq
    thv::FT                # θ_v
    e_int::FT
    h_m::FT
    h_t::FT
    qt_sgs::FT
    ht_sgs::FT

    # vertical fluxes
    vert_eddy_mass_flx::FT # <w′ρ′>
    vert_eddy_u_flx::FT    # <w′u′>
    vert_eddy_v_flx::FT    # <w′v′>
    vert_eddy_qt_flx::FT   # <w'q_tot'>
    vert_qt_flx::FT        # <w q_tot>
    vert_eddy_ql_flx::FT   # <w′q_liq′>
    vert_eddy_qv_flx::FT   # <w′q_vap′>
    vert_eddy_thd_flx::FT  # <w′θ′>
    vert_eddy_thv_flx::FT  # <w′θ_v′>
    vert_eddy_thl_flx::FT  # <w′θ_liq′>

    # variances
    uvariance::FT          # u′u′
    vvariance::FT          # v′v′
    wvariance::FT          # w′w′

    # skewness
    wskew::FT              # w′w′w′

    # turbulent kinetic energy
    TKE::FT
  end
end

num_diagnostic(FT) = varsize(vars_diagnostic(FT))
diagnostic_vars(array) = Vars{vars_diagnostic(eltype(array))}(array)

