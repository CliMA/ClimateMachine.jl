
@kernel function update!(dQ, Q, rka, rkb, dt, slow_δ, slow_dQ, slow_scaling)
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            dQ[i] += slow_δ * slow_dQ[i]
        end
        Q[i] += rkb * dt * dQ[i]
        dQ[i] *= rka
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end
    end
end
