using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
end

@kernel function linearcombination!(Q, cs, Xs, increment::Bool)
    i = @index(Global, Linear)
    if !increment
        @inbounds Q[i] = -zero(eltype(Q))
    end
    @inbounds for j in 1:length(cs)
        Q[i] += cs[j] * Xs[j][i]
    end
end
