import ..BalanceLaws: projection

# Zero-out vertical momentum tendencies based on compressibility
function projection(
    atmos::AtmosModel,
    td::TendencyDef{TT, PV},
    args,
    x,
) where {TT, PV <: Momentum}
    return projection(atmos.compressibility, td, args, x)
end

# Zero-out vertical momentum fluxes for Anelastic1D:
function projection(
    ::Anelastic1D,
    ::TendencyDef{Flux{O}, PV},
    args,
    x,
) where {O, PV <: Momentum}
    return x .* SArray{Tuple{3, 3}}(1, 1, 1, 1, 1, 1, 0, 0, 0)
end

# Zero-out vertical momentum sources for Anelastic1D:
function projection(
    ::Anelastic1D,
    ::TendencyDef{Source, PV},
    args,
    x,
) where {PV <: Momentum}
    return x .* SVector(1, 1, 0)
end
