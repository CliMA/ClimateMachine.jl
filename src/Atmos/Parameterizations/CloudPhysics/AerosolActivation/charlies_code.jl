mass_mx_rat = [[1, 2, 3, 4, 5],
                 [0.1, 0.2, 0.3, 0.4, 0.5],
                 [0.01, 0.02, 0.03, 0.04, 0.05]]
diss = [[1, 2, 3, 4, 5],
              [0.1, 0.2, 0.3, 0.4, 0.5],
              [0.01, 0.02, 0.03, 0.04, 0.05]]
mass_frac = [[1, 2, 3, 4, 5],
       [0.1, 0.2, 0.3, 0.4, 0.5],
       [0.01, 0.02, 0.03, 0.04, 0.05]]
aero_mm = [[1, 2, 3, 4, 5],
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.01, 0.02, 0.03, 0.04, 0.05]]
aero_ρ = [[1, 2, 3, 4, 5],
               [0.1, 0.2, 0.3, 0.4, 0.5],
               [0.01, 0.02, 0.03, 0.04, 0.05]]
struct SomeModeModel{T}
    diss::T
    mass_frac::T
    mass_mx_rat::T
    aero_ρ::T
    aero_mm::T
end
smm = ntuple(3) do j
    SomeModeModel(
        Tuple(diss[j]),
        Tuple(mass_frac[j]),
        Tuple(mass_mx_rat[j]),
        Tuple(aero_ρ[j]),
        Tuple(aero_mm[j]),
    )
end
struct AerosolModel{T}
    modes::T
    N::Int
    function AerosolModel(modes::T) where {T}
        return new{T}(modes, length(modes))
    end
end
n_modes(am::AerosolModel) = am.N
n_chems(smm::SomeModeModel) = length(smm.diss)
# TODO: move to CLIMAParameters?
# and remove closure over these parameters
# (i.e., compute `WTR_MM` & `WTR_ρ` inside `mean_hygroscopicity`)
WTR_MM = 18
WTR_ρ = 1000
function mean_hygroscopicity(am::AerosolModel)
    return ntuple(n_modes(am)) do j
        mode_j = am.modes[j]
        coeff = WTR_MM/WTR_ρ
        N = n_chems(mode_j)
        ∑num = sum(N) do i
            mode_j.mass_mx_rat[i] * mode_j.diss[i] * mode_j.mass_frac[i] * (1/mode_j.aero_mm[i])
        end
        ∑den = sum(N) do i
            mode_j.mass_mx_rat[i] / mode_j.aero_ρ[i]
        end
        coeff * ∑num / ∑den
    end
end