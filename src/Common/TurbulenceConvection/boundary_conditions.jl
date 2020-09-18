#### Boundary conditions

export TurbConvBC,
    NoTurbConvBC,
    turbconv_bcs,
    turbconv_boundary_state!,
    turbconv_normal_boundary_flux_second_order!

abstract type TurbConvBC <: BoundaryCondition end

"""
    NoTurbConvBC <: TurbConvBC

Boundary conditions are not applied
"""
struct NoTurbConvBC <: TurbConvBC end

turbconv_bcs(::NoTurbConv) = NoTurbConvBC()
