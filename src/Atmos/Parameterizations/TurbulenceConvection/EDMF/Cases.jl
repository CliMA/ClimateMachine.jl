#### Cases

export Case
export BOMEX

"""
    Case

An abstract case, which encompasses
all EDMF benchmark cases.
"""
abstract type Case end

"""
    BOMEX

The Barbados Oceanographic Meteorological
Experiment (BOMEX).

Reference:
  Kuettner, Joachim P., and Joshua Holland.
  "The BOMEX project." Bulletin of the American
  Meteorological Society 50.6 (1969): 394-403.
"""
struct BOMEX <: Case end

