using Ferrite, SparseArrays, JespersPackage

L = 20.0   # Length [mm]


function calculate_stresses(grid, dh, cv_dil, cv_dev, u, C_dil, C_dev)
    qp_stresses = [
        [zero(SymmetricTensor{2, 2}) for _ in 1:getnquadpoints(cv_dev)]
            for _ in 1:getncells(grid)
    ]
    avg_cell_stresses = tuple((zeros(getncells(grid)) for _ in 1:3)...)

    for cell in CellIterator(dh)
        reinit!(cv_dil, cell)
        reinit!(cv_dev, cell)
        cell_stresses = qp_stresses[cellid(cell)]

        # Volumetric part
        σ_vol = zero(SymmetricTensor{2, 2})
        for q_point in 1:getnquadpoints(cv_dil)
            ε_vol = function_symmetric_gradient(cv_dil, q_point, u, celldofs(cell))
            σ_vol += C_dil ⊡ ε_vol
        end
        σ_vol /= getnquadpoints(cv_dil)

        # Deviatoric part
        for q_point in 1:getnquadpoints(cv_dev)
            ε_dev = function_symmetric_gradient(cv_dev, q_point, u, celldofs(cell))
            σ_dev = C_dev ⊡ ε_dev
            cell_stresses[q_point] = σ_vol + σ_dev
        end

        σ_avg = sum(cell_stresses) / getnquadpoints(cv_dev)
        avg_cell_stresses[1][cellid(cell)] = σ_avg[1, 1]
        avg_cell_stresses[2][cellid(cell)] = σ_avg[2, 2]
        avg_cell_stresses[3][cellid(cell)] = σ_avg[1, 2]
    end
    return qp_stresses, avg_cell_stresses
end

traction(x) = Vec(0.0, 1.0e3);

grid = generate_grid(Quadrilateral, (4, 1), Vec(0.0, 0.0), Vec(20.0, 1.0)); #Creating a 5x2 grid of quadrilaterals.

# Fix left (x ≈ 0) boundary completely
addfacetset!(grid, "fixed", x -> abs(x[1]) < 1.0e-8)
addfacetset!(grid, "rightt", x -> abs(x[1]) ≈ L)

dim = 2
order_dil = 1
order_dev = 2

ip = Lagrange{RefQuadrilateral, 1}()^dim # This produces nodes

qr_dil = QuadratureRule{RefQuadrilateral}(order_dil) # 1-point quadrature for volumetric part (dilational)
qr_dev = QuadratureRule{RefQuadrilateral}(order_dev) # 4-point quadrature for deviatoric part 
qr_face = FacetQuadratureRule{RefQuadrilateral}(1)

cellvalues_dil = CellValues(qr_dil, ip)
cellvalues_dev = CellValues(qr_dev, ip)
facetvalues = FacetValues(qr_face, ip)

dh = DofHandler(grid)
add!(dh, :u, ip) # Vector-valued
close!(dh)

# Setup constraint handler - only fix left edge
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "fixed"), (x, t) -> [0.0, 0.0]))  # Fix left edges completely (vector-valued)
close!(ch)


# Material parameters
Emod = 200.0e3  # Young's modulus [MPa]

nu = 0.3  # Poisson's ratio


C_dil, C_dev, C, Gmod, Kmod = JespersPackage.plane_strain_tensors(Emod, nu)

# Solve
K = allocate_matrix(dh)
JespersPackage.SL_assemble_global!(K, dh, cellvalues_dil, cellvalues_dev, C_dil, C_dev)

f_ext = zeros(ndofs(dh))

JespersPackage.assemble_external_forces!(f_ext, dh, getfacetset(grid, "rightt"), facetvalues, traction)
#assemble_external_forces!(f_ext, dh, getfacetset(grid, "rightt"), facetvalues, traction)

apply!(K, f_ext, ch)

u = K \ f_ext

# Calculate stresses
qp_stresses, avg_cell_stresses = calculate_stresses(grid, dh, cellvalues_dil, cellvalues_dev, u, C_dil, C_dev)

# Calculate von Mises stresses
cell_vm = JespersPackage.cell_von_mises_plane_strain(grid, dh, cellvalues_dev, u, C)

# Stress field projection
proj = L2Projector(Lagrange{RefQuadrilateral, 1}(), grid)
stress_field = project(proj, qp_stresses, qr_dev)


VTKGridFile("shear_locking", dh) do vtk
    write_solution(vtk, dh, u)
    
    for (i, key) in enumerate(("11", "22", "12"))
        write_cell_data(vtk, avg_cell_stresses[i], "sigma_" * key)
    end
    write_cell_data(vtk, cell_vm, "von_mises")
    write_projection(vtk, proj, stress_field, "stress field")
    Ferrite.write_cellset(vtk, grid)
end





 

