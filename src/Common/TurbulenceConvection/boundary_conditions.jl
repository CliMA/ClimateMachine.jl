#### Boundary conditions

export TurbConvBC,
    NoTurbConvBC,
    turbconv_bcs,
    turbconv_boundary_state!,
    turbconv_normal_boundary_flux_second_order!

abstract type TurbConvBC end

"""
    NoTurbConvBC <: TurbConvBC

Boundary conditions are not applied
"""
struct NoTurbConvBC <: TurbConvBC end

turbconv_bcs(::NoTurbConv) = NoTurbConvBC()

function turbconv_boundary_state!(nf, bc_turbulence::NoTurbConvBC, bl, args...)
    nothing
end

function turbconv_normal_boundary_flux_second_order!(
    nf,
    bc_turbulence::NoTurbConvBC,
    bl,
    args...,
)
    nothing
end
