#### Cases

export Case
export Soares, BOMEX, life_cycle_Tan2018
export Rico, TRMM_LBA, ARM_SGP, GATE_III
export DYCOMS_RF01, GABLS, SP

"""
    Case

An abstract case, which encompasses
all EDMF benchmark cases.
"""
abstract type Case end

"""
    Soares

The Barbados Oceanographic Meteorological
Experiment (BOMEX). Reference:

Key configuration features:

Reference:
  Soares, P. M. M., et al. "An eddy‐diffusivity/mass‐flux
  parametrization for dry and shallow cumulus convection."
  Quarterly Journal of the Royal Meteorological Society
  130.604 (2004): 3365-3383.
"""
struct Soares <: Case end

"""
    BOMEX

The Barbados Oceanographic Meteorological
Experiment (BOMEX).

Key configuration features:

Reference:
  Kuettner, Joachim P., and Joshua Holland.
  "The BOMEX project." Bulletin of the American
  Meteorological Society 50.6 (1969): 394-403.
"""
struct BOMEX <: Case end

struct life_cycle_Tan2018 <: Case end
struct Rico <: Case end
struct TRMM_LBA <: Case end
struct ARM_SGP <: Case end
struct GATE_III <: Case end
struct DYCOMS_RF01 <: Case end
struct GABLS <: Case end
struct SP <: Case end

