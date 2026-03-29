function assemble_Rint!(f_int, dh, scheme, all_states)
    fe_int = zeros(getnbasefunctions(get_primary_cv(scheme)))
    for cell in CellIterator(dh)
        reinit_scheme!(scheme, cell)
        fill!(fe_int, 0.0)
        cell_states = all_states[cellid(cell)]
        for qp in 1:getnquadpoints(get_primary_cv(scheme))
            dΩ = getdetJdV(get_primary_cv(scheme), qp)
            σ  = cell_states[qp].σ
            for i in 1:getnbasefunctions(get_primary_cv(scheme))
                ∇ˢʸᵐNᵢ = shape_symmetric_gradient(get_primary_cv(scheme), qp, i)
                fe_int[i] += (∇ˢʸᵐNᵢ ⊡ σ) * dΩ   # Bᵀ σ dΩ
            end
        end
        assemble!(f_int, celldofs(cell), fe_int)  # reuse Ferrite's assemble!
    end
    return f_int
end