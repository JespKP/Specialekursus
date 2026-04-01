

function assemble_global!(K, dh, scheme, mat::LinearElasticMaterial)
    n_basefuncs = getnbasefunctions(get_primary_cv(scheme))
    ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    
    for cell in CellIterator(dh)
        reinit_scheme!(scheme, cell)
        fill!(ke, 0.0)
        jp_test_package.assemble_cell!(ke, scheme, mat)
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end


function assemble_global_plastic!(K, dh, scheme, mat::LinearElasticMaterial,
                                   committed_states, trial_states, h)
    n_basefuncs = getnbasefunctions(get_primary_cv(scheme))
    ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit_scheme!(scheme, cell)
        fill!(ke, 0.0)
        # tangent from committed (σ_prev), Rint from trial
        jp_test_package.assemble_cell!(ke, scheme, mat,
                                       committed_states[cellid(cell)],
                                       trial_states[cellid(cell)], h)
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end