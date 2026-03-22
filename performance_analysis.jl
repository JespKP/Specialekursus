using BenchmarkTools
using Cthulhu

# Profiling 
using Ferrite, SparseArrays, JespersPackage
function run_reduced_simulation(nel_y)
    L = 20.0   # Length [mm]

    traction(x) = Vec(0.0, 1.0e3);

    grid = generate_grid(Quadrilateral, (4*nel_y, nel_y), Vec(0.0, 0.0), Vec(20.0, 1.0)); #Creating a 5x2 grid of quadrilaterals.

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
    mat = JespersPackage.plane_strain_tensors(Emod, nu)
    # Reduced integration - everything identical except the scheme
    scheme_reduced = ReducedIntegration(cellvalues_dil, cellvalues_dev)

    u_reduced  = JespersPackage.run_simulation(dh, scheme_reduced,  mat, ch, getfacetset(grid, "rightt"), facetvalues, traction)

    VTKGridFile("reduced_integration", dh) do vtk
        write_solution(vtk, dh, u_reduced)
    end
end

# general insight
@time run_reduced_simulation(10)
@profview for i in 1:3 run_reduced_simulation(100); end

# Benchmarking
function set_up_cellassembly()
    grid = generate_grid(Quadrilateral, (2, 2), Vec(0.0, 0.0), Vec(1.0, 1.0))

    ip = Lagrange{RefQuadrilateral, 1}()^2
    qr_dil  = QuadratureRule{RefQuadrilateral}(1)
    qr_dev  = QuadratureRule{RefQuadrilateral}(2)

    cv_dil  = CellValues(qr_dil, ip)
    cv_dev  = CellValues(qr_dev, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    Emod = 200.0e3
    nu   = 0.3
    mat  = JespersPackage.plane_strain_tensors(Emod, nu)

    scheme_red = ReducedIntegration(cv_dil, cv_dev)

    n = ndofs_per_cell(dh)
    ke_red = zeros(n, n)

    # Reinit to first cell so gradients are defined
    cell = first(CellIterator(dh))
    reinit!(cv_dil, cell)
    reinit!(cv_dev, cell)
    
    return ke_red, scheme_red, mat
end


ke_red, scheme_red, mat = set_up_cellassembly()
JespersPackage.assemble_cell!(ke_red, scheme_red, mat)

@time JespersPackage.assemble_cell!(ke_red, scheme_red, mat);

@btime JespersPackage.assemble_cell!($ke_red, $scheme_red, $mat);

@code_warntype JespersPackage.assemble_cell!(ke_red, scheme_red, mat);


# this comes from Cthulhu
@descend JespersPackage.assemble_cell!(ke_red, scheme_red, mat);
