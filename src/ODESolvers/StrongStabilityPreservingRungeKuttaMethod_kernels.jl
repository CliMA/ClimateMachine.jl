
@kernel function update!(
    dQ,
    Q,
    Qstage,
    rka1,
    rka2,
    rkb,
    dt,
    slow_δ,
    slow_dQ,
    slow_scaling,
)
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            dQ[i] += slow_δ * slow_dQ[i]
        end
        Qstage[i] = rka1 * Q[i] + rka2 * Qstage[i] + dt * rkb * dQ[i]
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end
    end
end
