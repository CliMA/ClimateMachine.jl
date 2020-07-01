
export TurbConvSource, turbconv_source!, turbconv_sources

turbconv_sources(m::TurbulenceConvectionModel) = ()

function turbconv_source!(
    f::Function,
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    f(m, bl, source, state, diffusive, aux, t, direction)
end
function turbconv_source!(
    ::Nothing,
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) end

# sources are applied additively
@generated function turbconv_source!(
    stuple::Tuple,
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    N = fieldcount(stuple)
    return quote
        Base.Cartesian.@nexprs $N i -> turbconv_source!(
            stuple[i],
            m,
            bl,
            source,
            state,
            diffusive,
            aux,
            t,
            direction,
        )
        return nothing
    end
end

"""
    TurbConvSource

A super-type for non-conservative
source terms in the turbulent
convection model equations.
"""
abstract type TurbConvSource end
