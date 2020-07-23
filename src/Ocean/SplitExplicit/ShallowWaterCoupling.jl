import ..ShallowWater: forcing_term!

@inline function forcing_term!(::SWModel, ::Coupled, S, Q, A, t)
    S.U += A.Gᵁ

    return nothing
end
