function run_simulation_plastic(dh, scheme, mat::LinearElasticMaterial, ch,
                                facetset, facetvalues, total_traction,
                                n_increments::Int, h::Float64, σ_y0::Float64;
                                imax::Int         = 20,
                                eps_stop::Float64 = 1e-6)

    nqp    = getnquadpoints(get_primary_cv(scheme))
    ncells = getncells(dh.grid)

    # ── Two buffers ───────────────────────────────────────────────────────────
    committed_states = [[init_plastic_state(σ_y0) for _ in 1:nqp] for _ in 1:ncells]
    trial_states     = [[init_plastic_state(σ_y0) for _ in 1:nqp] for _ in 1:ncells]

    u         = zeros(ndofs(dh))
    u_committed = zeros(ndofs(dh))
    u_trial     = zeros(ndofs(dh))
    P         = zeros(ndofs(dh))
    f_int     = zeros(ndofs(dh))
    K         = allocate_matrix(dh)

    # ── ΔP per increment ─────────────────────────────────────────────────────
    ΔP = zeros(ndofs(dh))
    assemble_external_forces!(ΔP, dh, facetset, facetvalues,
                               x -> total_traction(x) / n_increments)

    # ── Reference norm for convergence ───────────────────────────────────────
    P_final = zeros(ndofs(dh))
    assemble_external_forces!(P_final, dh, facetset, facetvalues, total_traction)
    P_final_norm = norm(P_final)
    P_final_norm == 0.0 && (P_final_norm = 1.0)

    # ── Outer loop: load increments ───────────────────────────────────────────
    for n in 1:n_increments

        P .+= ΔP

        u_committed .= u
        u_trial     .= u

        converged = false

        # ── Inner loop: equilibrium iterations ───────────────────────────────
        for k in 1:imax

            # Reset trial states to committed, then assemble K, f_int and
            # update trial_states — all in one cell loop
            for cell in 1:ncells
                for qp in 1:nqp
                    trial_states[cell][qp] = committed_states[cell][qp]
                end
            end

            fill!(K.nzval, 0.0)
            fill!(f_int, 0.0)
            assemble_plastic!(K, f_int, trial_states, dh, scheme, mat,
                               committed_states, u_trial, u_committed, h)

            # Residual
            R = f_int .- P
            apply_zero!(R, ch)

            #println("Increment $n, iter $k, ‖R‖ = $(norm(R))")

            if norm(R) ≤ eps_stop * P_final_norm # Is the residual smaller than one millionth of the final load norm?
                #println("Increment $n converged at iteration $k")
                converged = true
                break
            end

            apply!(K, ch)
            u_trial .+= K \ (-R)

        end  # equilibrium iterations

        if !converged
            #@warn "Increment $n did NOT converge after $imax iterations"
        end

        # Commit
        u .= u_trial
        for cell in 1:ncells
            for qp in 1:nqp
                committed_states[cell][qp] = trial_states[cell][qp]
            end
        end

    end  # load increments

    return u, committed_states
end


function run_simulation_material(dh, scheme, mat::LinearElasticMaterial, ch,
                                 facetset, facetvalues, total_traction;
                                 model::Symbol     = :elastic,
                                 n_increments::Int = 1,
                                 h::Float64        = 0.0,
                                 σ_y0::Float64     = 0.0)
    if model == :elastic
        return run_simulation(dh, scheme, mat, ch, facetset, facetvalues, total_traction), nothing
    elseif model == :perfect_plasticity
        return run_simulation_plastic(
            dh, scheme, mat, ch, facetset, facetvalues,
            total_traction, n_increments, 0.0, σ_y0
        )
    elseif model == :isotropic_hardening
        return run_simulation_plastic(
            dh, scheme, mat, ch, facetset, facetvalues,
            total_traction, n_increments, h, σ_y0
        )
    else
        throw(ArgumentError("Unknown material model: $model"))
    end
end