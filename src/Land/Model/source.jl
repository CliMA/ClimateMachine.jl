#### Land sources

"""
    land_source!(
        f::Function,
        land::LandModel,
        source::Vars,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        direction,
)
"""
function land_source!(
    f::Function,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    f(land, source, state, diffusive, aux, t, direction)
end

"""
    land_source!(
        ::Nothing,
        land::LandModel,
        source::Vars,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        direction,
    ) end
"""
function land_source!(
    ::Nothing,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) end


# sources are applied additively
"""
    land_source!(
        stuple::Tuple,
        land::LandModel,
        source::Vars,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        direction,
    )
"""
@generated function land_source!(
    stuple::Tuple,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    N = fieldcount(stuple)
    return quote
        Base.Cartesian.@nexprs $N i -> land_source!(
            stuple[i],
            land,
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
