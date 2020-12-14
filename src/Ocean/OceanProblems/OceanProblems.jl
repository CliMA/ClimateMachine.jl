module OceanProblems

export SimpleBox, Fixed, Rotating, HomogeneousBox, OceanGyre

using StaticArrays
using CLIMAParameters.Planet: grav

using ...Problems

using ..Ocean
using ..HydrostaticBoussinesq
using ..ShallowWater
using ..SplitExplicit01

import ..Ocean:
    ocean_init_state!,
    ocean_init_aux!,
    kinematic_stress,
    surface_flux,
    coriolis_parameter

HBModel = HydrostaticBoussinesqModel
SWModel = ShallowWaterModel

include("simple_box_problem.jl")
include("homogeneous_box.jl")

include("shallow_water_initial_states.jl")

function ocean_init_state!(
    m::SWModel,
    p::HomogeneousBox,
    Q,
    A,
    local_geometry,
    t,
)
    if t == 0
        null_init_state!(p, m.turbulence, Q, A, local_geometry, 0)
    else
        gyre_init_state!(m, p, m.turbulence, Q, A, local_geometry, t)
    end
end

@inline coriolis_parameter(m::SWModel, p::HomogeneousBox, y) =
    m.fₒ + m.β * (y - p.Lʸ / 2)

include("ocean_gyre.jl")

include("initial_value_problem.jl")

end # module
