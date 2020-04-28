# CliMA Diagnostic Variable List

This document contains the diagnostic variables in CliMA.

## LES Diagnostics

### default

| code name            | short name | description                                                                |
|:---------------------|:-----------|:---------------------------------------------------------------------------|
| u                    | u          | x-velocity                                                                 |
| v                    | v          | y-velocity                                                                 |
| w                    | w          | z-velocity                                                                 |
| ρ                    | avg_rho    | density                                                                    |
| ρ                    | rho        | (density-averaged) density                                                 |
| q\_tot                | qt         | total specific humidity                                                    |
| q\_liq                | ql         | liquid water specific humidity                                             |
| q\_vap                | qv         | water vapor specific humidity                                              |
| θ\_dry                | thd        | potential temperature                                                      |
| θ\_vir                | thv        | virtual potential temperature                                              |
| θ\_liq\_ice            | thl        | liquid-ice potential temperature                                           |
| e\_tot                | et         | total specific energy                                                      |
| e\_int                | ei         | specific internal energy                                                   |
| h\_tot                | ht         | total specific enthalpy                                                    |
| h\_moi                | hm         | specific enthalpy                                                          |
| u′u′                 | var\_u      | variance of x-velocity                                                     |
| v′v′                 | var\_v      | variance of y-velocity                                                     |
| w′w′                 | var\_w      | variance of z-velocity                                                     |
| w′w′w′               | w3         | the third moment of z-velocity                                             |
| tke                  | tke        | turbulence kinetic energy                                                  |
| q\_tot′q\_tot′         | var\_qt     | variance of total specific humidity                                        |
| θ\_liq\_ice′θ\_liq\_ice′ | var\_thl    | variance of liquid-ice potential temperature                               |
| e\_int′e\_int′         | var\_ei     | variance of specific internal energy                                       |
| w′u′                 | cov\_w\_u    | vertical eddy flux of x-velocity                                           |
| w′v′                 | cov\_w\_v    | vertical eddy flux of y-velocity                                           |
| w′ρ′                 | cov\_w\_rho  | vertical eddy flux of mass                                                 |
| w′q\_tot′             | cov\_w\_qt   | vertical eddy flux of total specific humidity                              |
| w′q\_liq′             | cov\_w\_ql   | vertical eddy flux of liuqid water specific humidity                       |
| w′q\_vap′             | cov\_w\_qv   | vertical eddy flux of water vapor specific humidity                        |
| w′θ\_dry′             | cov\_w\_thd  | vertical eddy flux of potential temperature                                |
| w′θ\_vir′             | cov\_w\_thv  | vertical eddy flux of virtual temperature                                  |
| w′θ\_liq\_ice′         | cov\_w\_thl  | vertical eddy flux of liquid-ice potential temperature                     |
| w′e\_int′             | cov\_w\_ei   | vertical eddy flux of specific internal energy                             |
| q\_tot′θ\_liq\_ice′     | cov\_qt\_thl | covariance of total specific humidity and liquid-ice potential temperature |
| q\_tot′e\_int′         | cov\_qt\_ei  | covariance of total specific humidity and specific internal energy         |
| d\_q\_tot              | w\_qt\_sgs   | vertical sgs flux of total specific humidity                               |
| d\_h\_tot              | w\_ht\_sgs   | vertical sgs flux of total specific enthalpy                               |
| cld\_frac             | cld\_frac   | cloud fraction                                                             |
| cld\_cover            | cld\_cover  | cloud cover                                                                |
| cld\_top              | cld\_top    | cloud top                                                                  |
| cld\_base             | cld\_base   | cloud base                                                                 |
| lwp                  | lwp        | liquid water path                                                          |

### core

| code name                 | short name      | description                                                                           |
|:--------------------------|:----------------|:--------------------------------------------------------------------------------------|
| ρ\_core                    | rho\_core        | cloud core density                                                                    |
| u\_core                    | u\_core          | cloud core x-velocity                                                                 |
| v\_core                    | v\_core          | cloud core y-velocity                                                                 |
| w\_core                    | w\_core          | cloud core z-velocity                                                                 |
| q\_tot\_core                | qt\_core         | cloud core total specific humidity                                                    |
| e\_int\_core                | ei\_core         | cloud core specific internal energy                                                   |
| θ\_liq\_ice                 | thl\_core        | cloud core liquid-ice potential temperature                                           |
| u′u′\_core                 | var\_u\_core      | cloud core variance of x-velocity                                                     |
| v′v′\_core                 | var\_v\_core      | cloud core variance of y-velocity                                                     |
| w′w′\_core                 | var\_w\_core      | cloud core variance of z-velocity                                                     |
| q\_tot′q\_tot′\_core         | var\_qt\_core     | cloud core variance of total specific humidity                                        |
| e\_int′e\_int′\_core         | var\_ei\_core     | cloud core variance of specific internal energy                                       |
| θ\_liq\_ice′θ\_liq\_ice′\_core | var\_thl\_core    | cloud core variance of liquid-ice potential temperature                               |
| q\_tot′θ\_liq\_ice′\_core     | cov\_qt\_thl\_core | cloud core covariance of total specific humidity and liquid-ice potential temperature |
| q\_tot′e\_int′\_core         | cov\_qt\_ei\_core  | cloud core covariance of total specific humidity and specific internal energy         |
| w′ρ′\_core                 | cov\_w\_rho\_core  | cloud core vertical eddy flux of mass                                                 |
| w′q\_tot′\_core             | cov\_w\_qt\_core   | cloud core vertical eddy flux of total specific humidity                              |
| w′θ\_liq\_ice′\_core         | cov\_w\_thl\_core  | cloud core vertical eddy flux of liquid-ice potential temperature                     |
| w′e\_int′\_core             | cov\_w\_ei\_core   | cloud core vertical eddy flux of specific internal energy                             |
| core\_frac                 | core\_frac       | cloud core (q\_liq > 0 and w > 0) fraction                                             |
