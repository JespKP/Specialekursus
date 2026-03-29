function run_simulation_plastic(dh, scheme, mat::LinearElasticMaterial, ch,
                                facetset, facetvalues, total_traction,
                                n_increments::Int, h::Float64, σ_y0::Float64;
                                imax::Int     = 20,
                                eps_stop::Float64 = 1e-8)

    nqp    = getnquadpoints(get_primary_cv(scheme))
    ncells = getncells(dh.grid)

    # ── Committed states: converged at end of previous increment ──────────────
    committed_states = [[init_plastic_state(σ_y0) for _ in 1:nqp] for _ in 1:ncells]

    u = zeros(ndofs(dh))   # Dⁿ — converged displacement
    P = zeros(ndofs(dh))   # Pⁿ — accumulated load

    # ΔP — load added per increment
    ΔP = zeros(ndofs(dh))
    assemble_external_forces!(ΔP, dh, facetset, facetvalues,
                               x -> total_traction(x) / n_increments)

    # ‖P_final‖ used as reference norm for convergence
    P_final = zeros(ndofs(dh))
    assemble_external_forces!(P_final, dh, facetset, facetvalues, total_traction)
    P_final_norm = norm(P_final)
    P_final_norm == 0.0 && (P_final_norm = 1.0)

    # ── Outer loop: load increments ───────────────────────────────────────────
    for n in 1:n_increments

        P .+= ΔP                           # Pⁿ = Pⁿ⁻¹ + ΔPⁿ

        u_committed  = copy(u)             # Dⁿ₀ = Dⁿ⁻¹  (start of increment)
        u_trial      = copy(u)             # current iterate Dⁿᵢ
        trial_states = deepcopy(committed_states)

        converged = false

        # ── Inner loop: equilibrium iterations ───────────────────────────────
        for i in 0:imax-1

            # 1. Rint(Dⁿᵢ)
            f_int = zeros(ndofs(dh))
            assemble_Rint!(f_int, dh, scheme, trial_states)

            # 2. Rⁿᵢ = Rint − Pⁿ
            R = f_int .- P

            # 3. Enforce BCs on Rⁿᵢ
            apply_zero!(R, ch)

            # 4. Convergence check
            if norm(R) ≤ eps_stop * P_final_norm
                println("Increment $n converged at iteration $i,  ‖R‖ = $(norm(R))")
                converged = true
                break
            end

            # 5. Kₜ(Dⁿᵢ) — tangent from current trial states
            K = allocate_matrix(dh)
            assemble_global_plastic!(K, dh, scheme, mat, committed_states, trial_states, h)


            # 6. Enforce BCs on Kₜ
            apply!(K, ch)

            # 7. Solve  ΔDⁿᵢ = −Kₜ⁻¹ Rⁿᵢ
            ΔD = K \ (-R)

            # 8. Update displacements  Dⁿᵢ₊₁ = Dⁿᵢ + ΔDⁿᵢ
            u_trial .+= ΔD

            # 9. Update stresses from committed base + total increment so far
            #    (always restarted from committed so path is consistent)
            trial_states = deepcopy(committed_states)
            for cell in CellIterator(dh)
                reinit_scheme!(scheme, cell)
                dofs    = celldofs(cell)
                Δu_cell = u_trial[dofs] .- u_committed[dofs]   # total Δu this increment
                for qp in 1:nqp
                    Δε = function_symmetric_gradient(get_primary_cv(scheme), qp, Δu_cell)
                    stress_update!(trial_states[cellid(cell)][qp], Δε, mat, h)
                end
            end

        end  # equilibrium iterations

        if !converged
            @warn "Increment $n did NOT converge after $imax iterations"
        end

        # Commit: Dⁿ = Dⁿᵢ
        u                = copy(u_trial)
        committed_states = deepcopy(trial_states)
    end  # load increments

    return u, committed_states
end
function run_simulation_material(dh, scheme, mat::LinearElasticMaterial, ch,
                                 facetset, facetvalues, total_traction;
                                 model::Symbol = :elastic,
                                 n_increments::Int = 1,
                                 h::Float64 = 0.0,
                                 σ_y0::Float64 = 0.0)
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

