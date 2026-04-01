function material_response(Δε::SymmetricTensor{2,2}, state, mat::LinearElasticMaterial, h::Float64)

    σ_prev   = state.σ
    σ_y_prev = state.σ_y

    σ_trial = σ_prev + mat.C ⊡ Δε

    if yield_function(σ_trial, σ_y_prev) < 0
        # ── Elastic ──────────────────────────────────────────────────────────
        new_state = PlasticState(σ_trial, σ_y_prev, state.εp, 0.0)
        return σ_trial, mat.C, new_state
    else
        # ── Plastic ──────────────────────────────────────────────────────────
        Δλ = compute_Δλ(σ_prev, Δε, mat, h)

        if Δλ ≤ 0
            new_state = PlasticState(σ_trial, σ_y_prev, state.εp, 0.0)
            return σ_trial, mat.C, new_state
        end

        n       = flow_direction(σ_prev)
        σ_new   = σ_prev + mat.C ⊡ (Δε - n * Δλ)
        σ_y_new = σ_y_prev + h * Δλ
        C_ep    = compute_Cep(σ_prev, mat, h)

        new_state = PlasticState(σ_new, σ_y_new, state.εp + Δλ, Δλ)
        return σ_new, C_ep, new_state
    end
end