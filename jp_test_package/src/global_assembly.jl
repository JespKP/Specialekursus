

function assemble_global!(K, dh, scheme, mat::LinearElasticMaterial)
    n_basefuncs = getnbasefunctions(get_primary_cv(scheme))
    ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    
    for cell in CellIterator(dh)
        reinit_scheme!(scheme, cell)
        fill!(ke, 0.0)
        JespersPackage.assemble_cell!(ke, scheme, mat)
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end