function calculate_stresses(grid, dh, scheme, u, mat::LinearElasticMaterial)
    cv = get_primary_cv(scheme)
    qp_stresses = [
        [zero(SymmetricTensor{2, 2}) for _ in 1:getnquadpoints(cv)]
            for _ in 1:getncells(grid)
    ]
    avg_cell_stresses = tuple((zeros(getncells(grid)) for _ in 1:3)...)

    for cell in CellIterator(dh)
        reinit_scheme!(scheme, cell)
        cell_stresses = qp_stresses[cellid(cell)]

        for q_point in 1:getnquadpoints(cv)
            ε = function_symmetric_gradient(cv, q_point, u, celldofs(cell))
            cell_stresses[q_point] = mat.C ⊡ ε
        end

        σ_avg = sum(cell_stresses) / getnquadpoints(cv)
        avg_cell_stresses[1][cellid(cell)] = σ_avg[1, 1]
        avg_cell_stresses[2][cellid(cell)] = σ_avg[2, 2]
        avg_cell_stresses[3][cellid(cell)] = σ_avg[1, 2]
    end
    return qp_stresses, avg_cell_stresses
end