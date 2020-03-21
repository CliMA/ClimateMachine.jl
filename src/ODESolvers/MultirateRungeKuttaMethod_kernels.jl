
@kernel function update!(fast_dQ, slow_dQ, δ, slow_rka = nothing)
    i = @index(Global, Linear)
    @inbounds begin
        fast_dQ[i] += δ * slow_dQ[i]
        if slow_rka !== nothing
            slow_dQ[i] *= slow_rka
        end
    end
end
