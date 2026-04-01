
# Equation 16 — closed-form plastic multiplier for von Mises + linear hardening
function compute_Δλ(σ_prev::SymmetricTensor{2, 2}, Δε::SymmetricTensor{2, 2},
                    mat::LinearElasticMaterial, h::Float64)
    n   = flow_direction(σ_prev)
    C   = mat.C
    num = n ⊡ (C ⊡ Δε)           # ∂F/∂σ · C · Δε  (scalar)
    den = (n ⊡ (C ⊡ n)) + h      # ∂F/∂σ · C · ∂F/∂σᵀ + h  (scalar)
    return num / den
end

# Plastic increment scheme — updates state in-place
function stress_update!(state::PlasticState, Δε::SymmetricTensor{2, 2},
                        mat::LinearElasticMaterial, h::Float64)
    σ_trial = state.σ + mat.C ⊡ Δε

    if yield_function(σ_trial, state.σ_y) < 0
        state.σ  = σ_trial
        state.Δλ = 0.0
    else
        Δλ = compute_Δλ(state.σ, Δε, mat, h)   # evaluated at σ_prev
        if Δλ <= 0
            state.σ  = σ_trial
            state.Δλ = 0.0
        else
            n         = flow_direction(state.σ)   # evaluated at σ_prev
            Δσ        = mat.C ⊡ (Δε - n * Δλ)
            state.σ   = state.σ + Δσ
            state.σ_y = state.σ_y + h * Δλ
            state.εp  = state.εp + Δλ
            state.Δλ  = Δλ
        end
    end
    return state
end