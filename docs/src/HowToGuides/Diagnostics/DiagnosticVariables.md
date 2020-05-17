# CliMA Diagnostic Variable List

This document contains the diagnostic variables in CliMA.

## LES Diagnostics

### default

| short name | description                                                                |
|:-----------|:---------------------------------------------------------------------------|
| u          | x-velocity                                                                 |
| v          | y-velocity                                                                 |
| w          | z-velocity                                                                 |
| avg_rho    | density                                                                    |
| rho        | (density-averaged) density                                                 |
| qt         | total specific humidity                                                    |
| ql         | liquid water specific humidity                                             |
| qv         | water vapor specific humidity                                              |
| thd        | potential temperature                                                      |
| thv        | virtual potential temperature                                              |
| thl        | liquid-ice potential temperature                                           |
| et         | total specific energy                                                      |
| ei         | specific internal energy                                                   |
| ht         | total specific enthalpy                                                    |
| hm         | specific enthalpy                                                          |
| var\_u      | variance of x-velocity                                                     |
| var\_v      | variance of y-velocity                                                     |
| var\_w      | variance of z-velocity                                                     |
| w3         | the third moment of z-velocity                                             |
| tke        | turbulence kinetic energy                                                  |
| var\_qt     | variance of total specific humidity                                        |
| var\_thl    | variance of liquid-ice potential temperature                               |
| var\_ei     | variance of specific internal energy                                       |
| cov\_w\_u    | vertical eddy flux of x-velocity                                           |
| cov\_w\_v    | vertical eddy flux of y-velocity                                           |
| cov\_w\_rho  | vertical eddy flux of mass                                                 |
| cov\_w\_qt   | vertical eddy flux of total specific humidity                              |
| cov\_w\_ql   | vertical eddy flux of liuqid water specific humidity                       |
| cov\_w\_qv   | vertical eddy flux of water vapor specific humidity                        |
| cov\_w\_thd  | vertical eddy flux of potential temperature                                |
| cov\_w\_thv  | vertical eddy flux of virtual temperature                                  |
| cov\_w\_thl  | vertical eddy flux of liquid-ice potential temperature                     |
| cov\_w\_ei   | vertical eddy flux of specific internal energy                             |
| cov\_qt\_thl | covariance of total specific humidity and liquid-ice potential temperature |
| cov\_qt\_ei  | covariance of total specific humidity and specific internal energy         |
| w\_qt\_sgs   | vertical sgs flux of total specific humidity                               |
| w\_ht\_sgs   | vertical sgs flux of total specific enthalpy                               |
| cld\_frac   | cloud fraction                                                             |
| cld\_cover  | cloud cover                                                                |
| cld\_top    | cloud top                                                                  |
| cld\_base   | cloud base                                                                 |
| lwp        | liquid water path                                                          |

### core

| short name      | description                                                                           |
|:----------------|:--------------------------------------------------------------------------------------|
| u\_core          | cloud core x-velocity                                                                 |
| v\_core          | cloud core y-velocity                                                                 |
| w\_core          | cloud core z-velocity                                                                 |
| avg\_rho\_core    | cloud core density                                                                    |
| rho\_core        | cloud core (density-averaged) density                                                 |
| qt\_core         | cloud core total specific humidity                                                    |
| ql\_core         | cloud core liquid water specific humidity                                             |
| thv\_core        | cloud core virtual potential temperature                                              |
| thl\_core        | cloud core liquid-ice potential temperature                                           |
| ei\_core         | cloud core specific internal energy                                                   |
| var\_u\_core      | cloud core variance of x-velocity                                                     |
| var\_v\_core      | cloud core variance of y-velocity                                                     |
| var\_w\_core      | cloud core variance of z-velocity                                                     |
| var\_qt\_core     | cloud core variance of total specific humidity                                        |
| var\_thl\_core    | cloud core variance of liquid-ice potential temperature                               |
| var\_ei\_core     | cloud core variance of specific internal energy                                       |
| cov\_w\_rho\_core  | cloud core vertical eddy flux of mass                                                 |
| cov\_w\_qt\_core   | cloud core vertical eddy flux of total specific humidity                              |
| cov\_w\_thl\_core  | cloud core vertical eddy flux of liquid-ice potential temperature                     |
| cov\_w\_ei\_core   | cloud core vertical eddy flux of specific internal energy                             |
| cov\_qt\_thl\_core | cloud core covariance of total specific humidity and liquid-ice potential temperature |
| cov\_qt\_ei\_core  | cloud core covariance of total specific humidity and specific internal energy         |
| core\_frac       | cloud core (q\_liq > 0 and w > 0) fraction                                             |
