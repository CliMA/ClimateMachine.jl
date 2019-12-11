"""
    Diagnostics variable template

Container for diagnostic variables of interest. Useful also for
post-processing.

"""

using CLIMA.VariableTemplates

function vars_diagnostic(FT)
  @vars begin
    z::FT                  # 1
    u::FT                  # 2
    v::FT                  # 3
    w::FT                  # 4
    q_tot::FT              # 5
    q_liq::FT              # 6
    e_tot::FT              # 7
    thd::FT                # 8
    thl::FT                # 9
    thv::FT                # 10
    e_int::FT              # 11
    h_m::FT                # 12
    h_t::FT                # 13
    qt_sgs::FT             # 14
    ht_sgs::FT             # 15
    vert_eddy_mass_flx::FT # 16
    vert_eddy_u_flx::FT    # 17
    vert_eddy_v_flx::FT    # 18
    vert_eddy_qt_flx::FT   # 19       #<w'q_tot'>
    vert_qt_flx::FT        # 20       #<w q_tot>
    vert_eddy_ql_flx::FT   # 21
    vert_eddy_qv_flx::FT   # 22
    vert_eddy_thd_flx::FT  # 23
    vert_eddy_thv_flx::FT  # 24
    vert_eddy_thl_flx::FT  # 25
    uvariance::FT          # 26
    vvariance::FT          # 27
    wvariance::FT          # 28
    wskew::FT              # 29
    TKE::FT                # 30    
    SGS::FT
    Ri::FT
    N::FT
  end
end

"""
    var_groups(FT)

Dictionary containing indexes of
variables to be plotted together
"""
function var_groups(FT)
  vars_diag = vars_diagnostic(FT)
  varnames_diag = fieldnames(vars_diag)
  D = Dict()
  D[:velocity] = findall(map(x-> x==:u || x==:v || x==:w, varnames_diag))
  D[:shum] = findall(map(x-> x==:q_tot || x==:q_liq || x==:q_ice, varnames_diag))
  D[:q_liq] = findall(map(x-> x==:q_liq, varnames_diag))
  D[:energy] = findall(map(x-> x==:e_tot || x==:e_int, varnames_diag))
  D[:pottemp] = findall(map(x-> x==:thd || x==:thl || x==:thv, varnames_diag))
  D[:UVariance] = findall(map(x-> x==:uvariance || x==:vvariance || x==:wvariance, varnames_diag))
  D[:VertEddyQ] = findall(map(x-> x==:vert_eddy_qt_flx || x==:vert_eddy_ql_flx || x==:vert_eddy_qv_flx, varnames_diag))
  D[:VertEddyPottemp] = findall(map(x-> x==:vert_eddy_thd_flx || x==:vert_eddy_thv_flx || x==:vert_eddy_thl_flx, varnames_diag))
  D[:SGS] = findall(map(x-> x==:SGS, varnames_diag))
  D[:Ricorrection]  = findall(map(x-> x==:Ri, varnames_diag))
  D[:buoyancyFreq] = findall(map(x-> x==:N, varnames_diag))
  D[:qt_sgs] = findall(map(x-> x==:qt_sgs, varnames_diag))
  D[:ht_sgs] = findall(map(x-> x==:ht_sgs, varnames_diag))
  D[:N] = findall(map(x-> x==:N, varnames_diag))
  return D
end


num_diagnostic(FT) = varsize(vars_diagnostic(FT))
diagnostic_vars(array) = Vars{vars_diagnostic(eltype(array))}(array)

