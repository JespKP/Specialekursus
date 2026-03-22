using Ferrite, SparseArrays, JespersPackage

L = 20.0   # Length [mm]



traction(x) = Vec(0.0, 1.0e3);

grid = generate_grid(Quadrilateral, (4, 1), Vec(0.0, 0.0), Vec(20.0, 1.0)); #Creating a 5x2 grid of quadrilaterals.

# Fix left (x ≈ 0) boundary completely
addfacetset!(grid, "fixed", x -> abs(x[1]) < 1.0e-8)
addfacetset!(grid, "rightt", x -> abs(x[1]) ≈ L)
# Setup constraint handler - only fix left edge


dim = 2
order_dil = 1
order_dev = 2




ip = Lagrange{RefQuadrilateral, 1}()^dim # The 1 indicates the order of the shape functions (linear in this case), the 2 indicates vector-valued (2D) problem
qr = QuadratureRule{RefQuadrilateral}(order_dev) # 4-point quadrature for cell assembly and stress calculation
qr_dil = QuadratureRule{RefQuadrilateral}(order_dil) # 1-point quadrature for volumetric part (dilational)
qr_dev = QuadratureRule{RefQuadrilateral}(order_dev) # 4-point quadrature for deviatoric part 
qr_face = FacetQuadratureRule{RefQuadrilateral}(1)

cellvalues_dil = CellValues(qr_dil, ip)
cellvalues_dev = CellValues(qr_dev, ip)
cellvalues = CellValues(qr, ip)
facetvalues = FacetValues(qr_face, ip)
dh = DofHandler(grid)
add!(dh, :u, ip) # Vector-valued
close!(dh)


ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "fixed"), (x, t) -> [0.0, 0.0]))  # Fix left edges completely (vector-valued)
close!(ch)



# Material parameters
Emod = 200.0e3  # Young's modulus [MPa]

nu = 0.3  # Poisson's ratio
# Standard integration
scheme_standard = StandardIntegration(cellvalues)
mat = JespersPackage.plane_strain_tensors(Emod, nu)

K = allocate_matrix(dh)
JespersPackage.assemble_global!(K, dh, scheme_standard, mat)
f_ext = zeros(ndofs(dh))
JespersPackage.assemble_external_forces!(f_ext, dh, getfacetset(grid, "rightt"), facetvalues, traction)
apply!(K, f_ext, ch)
u_standard = K \ f_ext

VTKGridFile("standard_integration", dh) do vtk
    write_solution(vtk, dh, u_standard)
end

# Reduced integration - everything identical except the scheme
scheme_reduced = ReducedIntegration(cellvalues_dil, cellvalues_dev)

K = allocate_matrix(dh)
JespersPackage.assemble_global!(K, dh, scheme_reduced, mat)
f_ext = zeros(ndofs(dh))
JespersPackage.assemble_external_forces!(f_ext, dh, getfacetset(grid, "rightt"), facetvalues, traction)
apply!(K, f_ext, ch)
u_reduced = K \ f_ext

VTKGridFile("reduced_integration", dh) do vtk
    write_solution(vtk, dh, u_reduced)
end

u_standard = JespersPackage.run_simulation(dh, scheme_standard, mat, ch, getfacetset(grid, "rightt"), facetvalues, traction)
u_reduced  = JespersPackage.run_simulation(dh, scheme_reduced,  mat, ch, getfacetset(grid, "rightt"), facetvalues, traction)
#Test to see how to push code
foo = function()
    2+2
    return(4)
end