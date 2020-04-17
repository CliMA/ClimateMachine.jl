# CliMA Diagnostic Variable List

This document contains the diagnostic variables in CliMA.

## LES Diagnostics

### default

| code name            | short name | description                                                                |
|----------------------|------------|----------------------------------------------------------------------------|
| ρ                    | rho        | density                                                                    |
| u                    | u          | x_velocity                                                                 |
| v                    | v          | y_velocity                                                                 |
| w                    | w          | z_velocity                                                                 |
| q_tot                | qt         | total specific humidity                                                    |
| q_vap                | qv         | total specific energy                                                      |
| q_vap                | qv         | water vapor specific humidity                                              |
| q_liq                | ql         | liquid water specific humidity                                             |
| e_int                | ei         | specific internal energy                                                   |
| θ_dry                | thd        | potential temperature                                                      |
| θ_liq_ice            | thl        | liquid-ice potential temperature                                           |
| θ_vir                | thv        | virtual potential temperature                                              |
| h_moi                | hm         | specific enthalpy                                                          |
| h_tot                | ht         | total specific enthalpy                                                    |
| u′u′                 | var_u      | variance of x-velocity                                                     |
| v′v′                 | var_v      | variance of y-velocity                                                     |
| w'w'                 | var_w      | variance of z-velocity                                                     |
| tke                  | tke        | turbulence kinetic energy                                                  |
| q_tot'q_tot'         | var_qt     | variance of total specific humidity                                        |
| e_int'e_int'         | var_ei     | variance of specific internal energy                                       |
| θ_liq_ice'θ_liq_ice' | var_thl    | variance of liquid-ice potential temperature                               |
| w′w′w′               | w3         | the third moment of z-velocity                                             |
| q_tot'θ_liq_ice'     | cov_qt_thl | covariance of total specific humidity and liquid-ice potential temperature |
| q_tot'e_int'         | cov_qt_ei  | covariance of total specific humidity and specific internal energy         |
| w′ρ′                 | cov_w_rho  | vertical eddy flux of mass                                                 |
| w′u′                 | cov_w_u    | vertical eddy flux of x-velocity                                           |
| w′v′                 | cov_w_v    | vertical eddy flux of y-velocity                                           |
| wq_tot               | w_qt       | vertical flux of total specific humidity                                   |
| w'q_tot'             | cov_w_qt   | vertical eddy flux of total specific humidity                              |
| w′q_vap′             | cov_w_qv   | vertical eddy flux of water vapor specific humidity                        |
| w′q_liq′             | cov_w_ql   | vertical eddy flux of liuqid water specific humidity                       |
| w′θ_dry′             | cov_w_thd  | vertical eddy flux of potential temperature                                |
| w′θ_vir′             | cov_w_thv  | vertical eddy flux of virtual temperature                                  |
| w′θ_liq_ice′         | cov_w_thl  | vertical eddy flux of liquid-ice potential temperature                     |
| w′e_int′             | cov_w_ei   | vertical eddy flux of specific internal energy                             |
| d_q_tot              | w_qt_sgs   | vertical sgs flux of total specific humidity                               |
| d_h_tot              | w_ht_sgs   | vertical sgs flux of total specific enthalpy                               |
| cld_frac             | cld_frac   | cloud fraction                                                             |
| cld_cover            | cld_cover  | cloud cover                                                                |
| cld_top              | cld_top    | cloud top                                                                  |
| cld_base             | cld_base   | cloud base                                                                 |
| lwp                  | lwp        | liquid water path                                                          |

### core

| code name                 | short name      | description                                                                           |
|---------------------------|-----------------|---------------------------------------------------------------------------------------|
| ρ_core                    | rho_core        | cloud core density                                                                    |
| u_core                    | u_core          | cloud core x_velocity                                                                 |
| v_core                    | v_core          | cloud core y_velocity                                                                 |
| w_core                    | w_core          | cloud core z_velocity                                                                 |
| q_tot_core                | qt_core         | cloud core total specific humidity                                                    |
| e_int_core                | ei_core         | cloud core specific internal energy                                                   |
| θ_liq_ice                 | thl_core        | cloud core liquid-ice potential temperature                                           |
| u′u′_core                 | var_u_core      | cloud core variance of x-velocity                                                     |
| v′v′_core                 | var_v_core      | cloud core variance of y-velocity                                                     |
| w'w'_core                 | var_w_core      | cloud core variance of z-velocity                                                     |
| q_tot'q_tot'_core         | var_qt_core     | cloud core variance of total specific humidity                                        |
| e_int'e_int'_core         | var_ei_core     | cloud core variance of specific internal energy                                       |
| θ_liq_ice'θ_liq_ice'_core | var_thl_core    | cloud core variance of liquid-ice potential temperature                               |
| q_tot'θ_liq_ice'_core     | cov_qt_thl_core | cloud core covariance of total specific humidity and liquid-ice potential temperature |
| q_tot'e_int'_core         | cov_qt_ei_core  | cloud core covariance of total specific humidity and specific internal energy         |
| w′ρ′_core                 | cov_w_rho_core  | cloud core vertical eddy flux of mass                                                 |
| w'q_tot'_core             | cov_w_qt_core   | cloud core vertical eddy flux of total specific humidity                              |
| w′θ_liq_ice′_core         | cov_w_thl_core  | cloud core vertical eddy flux of liquid-ice potential temperature                     |
| w′e_int′_core             | cov_w_ei_core   | cloud core vertical eddy flux of specific internal energy                             |
| core_frac                 | core_frac       | cloud core (q_liq > 0 and w > 0) fraction                                             |
