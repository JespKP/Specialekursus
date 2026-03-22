using Test
using Ferrite
using JespersPackage
using LinearAlgebra

# ── Shared setup used by all tests ──────────────────────────────────────────
grid = generate_grid(Quadrilateral, (2, 2), Vec(0.0, 0.0), Vec(1.0, 1.0))

ip = Lagrange{RefQuadrilateral, 1}()^2
qr_dil  = QuadratureRule{RefQuadrilateral}(1)
qr_dev  = QuadratureRule{RefQuadrilateral}(2)
qr_face = FacetQuadratureRule{RefQuadrilateral}(1)

cv      = CellValues(qr_dev, ip)
cv_dil  = CellValues(qr_dil, ip)
cv_dev  = CellValues(qr_dev, ip)
fv      = FacetValues(qr_face, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0]))
close!(ch)

Emod = 200.0e3
nu   = 0.3
mat  = JespersPackage.plane_strain_tensors(Emod, nu)

scheme_std = StandardIntegration(cv)
scheme_red = ReducedIntegration(cv_dil, cv_dev)

# ── 1. Test plane_strain_tensors ─────────────────────────────────────────────
@testset "plane_strain_tensors" begin
    # Check it returns the right type
    @test mat isa LinearElasticMaterial

    # Check symmetry of C
    @test mat.C[1,1,2,2] ≈ mat.C[2,2,1,1]

    # Check C = C_dil + C_dev
    @test mat.C ≈ mat.C_dil + mat.C_dev

end

# ── 2. Test assemble_cell! ───────────────────────────────────────────────────
@testset "assemble_cell!" begin
    n = getnbasefunctions(cv)
    ke_std = zeros(n, n)
    ke_red = zeros(n, n)

    # Reinit to first cell so gradients are defined
    cell = first(CellIterator(dh))
    reinit!(cv, cell)
    reinit!(cv_dil, cell)
    reinit!(cv_dev, cell)

    JespersPackage.assemble_cell!(ke_std, scheme_std, mat)
    JespersPackage.assemble_cell!(ke_red, scheme_red, mat)

    # Stiffness matrix should be symmetric
    @test ke_std ≈ ke_std'
    @test ke_red ≈ ke_red'

    # Diagonal entries should be positive
    @test all(diag(ke_std) .> 0.0)
    
    @test all(diag(ke_red) .> 0.0)

    # Both schemes should give similar (not necessarily identical) results
    @test size(ke_std) == size(ke_red)
end

# ── 3. Test assemble_global! ───────────────────────────────────────────────────
@testset "assemble_global!" begin
    K_std = allocate_matrix(dh)
    K_red = allocate_matrix(dh)


    JespersPackage.assemble_global!(K_std, dh, scheme_std, mat)
    JespersPackage.assemble_global!(K_red, dh, scheme_red, mat)

    # Should be symmetric
    @test K_std ≈ K_std'
    @test K_red ≈ K_red'

    @test size(K_std,1) == dh.ndofs
    @test size(K_red,2) == dh.ndofs
end

    
# ── 4. Test assemble_external_forces! ─────────────────────────────────────────
@testset "assemble_external_forces!" begin
    f_ext = zeros(ndofs(dh))
    traction(x) = Vec(0.0, 1.0e3);
    facetvalues = FacetValues(qr_face, ip)

    f_ext = assemble_external_forces!(f_ext, dh, getfacetset(grid, "left"), facetvalues, traction)

    @test any(f_ext .!= 0)

    @test sum(f_ext) > 0
end

# ── 5. Test run_simulation ───────────────────────────────────────────────────
@testset "run_simulation" begin
    traction(x) = Vec(1.0e3, 0.0)

    u_std = JespersPackage.run_simulation(dh, scheme_std, mat, ch, getfacetset(grid, "right"), fv, traction)
    u_red = JespersPackage.run_simulation(dh, scheme_red, mat, ch, getfacetset(grid, "right"), fv, traction)

    @test length(u_std) == length(u_red)
    @test length(u_std) == dh.ndofs

    for dof in ch.prescribed_dofs
        @test abs(u_std[dof]) < 1.0e-10
        @test abs(u_red[dof]) < 1.0e-10
    end

    @test any(u_std .> 0.0)
    @test any(u_red .> 0.0)

end

# ── 6. Calculate_stresses ───────────────────────────────────────────────────
@testset "calculate_stresses" begin
    traction(x) = Vec(1.0e3, 0.0)
    u = JespersPackage.run_simulation(dh, scheme_std, mat, ch, getfacetset(grid, "right"), fv, traction)

    qp_stresses, avg_cell_stresses = JespersPackage.calculate_stresses(grid, dh, scheme_std, u, mat)

    length(qp_stresses) == getncells(grid)
end