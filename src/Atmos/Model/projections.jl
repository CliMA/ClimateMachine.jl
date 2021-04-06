import ..BalanceLaws: projection

# Zero-out vertical momentum tendencies based on compressibility
function projection(pv::Momentum, atmos::AtmosModel, td::TendencyDef, args, x)
    return projection(pv, compressibility_model(atmos), td, args, x)
end

# Zero-out vertical momentum fluxes for Anelastic1D:
function projection(
    pv::Momentum,
    ::Anelastic1D,
    ::TendencyDef{Flux{O}},
    args,
    x,
) where {O}
    return x .* SArray{Tuple{3, 3}}(1, 1, 1, 1, 1, 1, 0, 0, 0)
end

# Zero-out vertical momentum sources for Anelastic1D:
function projection(::Momentum, ::Anelastic1D, ::TendencyDef{Source}, args, x)
    return x .* SVector(1, 1, 0)
end
