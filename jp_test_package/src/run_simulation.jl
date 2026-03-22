function run_simulation(dh, scheme, mat::LinearElasticMaterial, ch, facetset, facetvalues, traction)
    K = allocate_matrix(dh)
    assemble_global!(K, dh, scheme, mat)
    
    f_ext = zeros(ndofs(dh))
    assemble_external_forces!(f_ext, dh, facetset, facetvalues, traction)
    
    apply!(K, f_ext, ch)
    
    u = K \ f_ext
    
    return u
end