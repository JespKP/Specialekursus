using Ferrite, SparseArrays
using BenchmarkTools
using Cthulhu
using jp_test_package

# Setup FE problem
# 3x2 elements -> 4 nodes horizontally, 3 nodes vertically (triangles)
nodes = Array{Node{2, Float64},1}([
    Node(Vec(0.0, 0.0)),  # 1
    Node(Vec(0.5, 0.0)),  # 2
    Node(Vec(1.0, 0.0)),  # 3
    Node(Vec(1.5, 0.0)),  # 4
    Node(Vec(0.0, 0.5)),  # 5
    Node(Vec(0.5, 0.5)),  # 6
    Node(Vec(1.0, 0.5)),  # 7
    Node(Vec(1.5, 0.5)),  # 8
    Node(Vec(0.0, 1.0)),  # 9
    Node(Vec(0.5, 1.0)),  # 10
    Node(Vec(1.0, 1.0)),  # 11
    Node(Vec(1.5, 1.0)),  # 12
])
cells = [
    # Bottom row
    Triangle((1, 2, 6)),   # (0,0)→(0.5,0)→(0.5,0.5) ✓
    Triangle((1, 6, 5)),   # (0,0)→(0.5,0.5)→(0,0.5) ✓
    Triangle((2, 3, 7)),   # (0.5,0)→(1,0)→(1,0.5)   ✓
    Triangle((2, 7, 6)),   # (0.5,0)→(1,0.5)→(0.5,0.5) ✓
    Triangle((3, 4, 8)),   # (1,0)→(1.5,0)→(1.5,0.5) ✓
    Triangle((3, 8, 7)),   # (1,0)→(1.5,0.5)→(1,0.5) ✓

    # Top row
    Triangle((5, 6, 10)),  # (0,0.5)→(0.5,0.5)→(0.5,1) ✓
    Triangle((5, 10, 9)),  # (0,0.5)→(0.5,1)→(0,1)     ✓
    Triangle((6, 7, 11)),  # (0.5,0.5)→(1,0.5)→(1,1)   ✓
    Triangle((6, 11, 10)), # (0.5,0.5)→(1,1)→(0.5,1)   ✓
    Triangle((7, 8, 12)),  # (1,0.5)→(1.5,0.5)→(1.5,1) ✓
    Triangle((7, 12, 11)), # (1,0.5)→(1.5,1)→(1,1)     ✓
]

grid = Grid(cells, nodes)


# Fix left (x ≈ 0) and bottom (y ≈ 0) boundaries completely
addfacetset!(grid, "fixed", x -> abs(x[1]) < 1.0e-8 || abs(x[2]) < 1.0e-8)
addfacetset!(grid, "topp", x -> abs(x[2] - 1.0) < 1e-8)


dim = 2
order = 1

ip = Lagrange{RefTriangle, order}()^dim

qr = QuadratureRule{RefTriangle}(2)
qr_face = FacetQuadratureRule{RefTriangle}(1)

cellvalues = CellValues(qr, ip)
facetvalues = FacetValues(qr_face, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)

close!(dh)

# Setup constraint handler - only fix left edge
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "fixed"), (x, t) -> [0.0, 0.0]))  # Fix left & bottom edges completely (vector-valued)
close!(ch)



traction(x) = Vec(0.0, 20.0e3 * x[1]);



# Material parameters: nearly incompressible to trigger volumetric locking
Emod = 200.0e3  # Young's modulus [MPa]



# Loop over ν to show volumetric locking
ν = 0.3


Gmod = Emod / (2(1 + ν))   # Shear modulus
Kmod = Emod / (3(1 - 2ν))  # Bulk modulus

# Constitutive tangent stiffness (elasticity tensor)
C = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2, 2}))
lambda = Emod * ν / ((1 + ν) * (1 - 2 * ν))
C3 = gradient(eps -> 2 * Gmod * dev(eps) + lambda * tr(eps) * one(eps), zero(SymmetricTensor{2, 3}))

# Assembly functions
function assemble_cell!(ke, cellvalues, C)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                ∇Nⱼ = shape_gradient(cellvalues, q_point, j)
                ∇ˢʸᵐNⱼ = symmetric(∇Nⱼ)
                ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function assemble_global!(K, dh, cellvalues, C)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        fill!(ke, 0.0)
        assemble_cell!(ke, cellvalues, C)
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

function calculate_stresses(grid, dh, cv, u, C)
    qp_stresses = [
        [zero(SymmetricTensor{2, 2}) for _ in 1:getnquadpoints(cv)]
            for _ in 1:getncells(grid)
    ]
    avg_cell_stresses = tuple((zeros(getncells(grid)) for _ in 1:3)...)
    
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        cell_stresses = qp_stresses[cellid(cell)]
        for q_point in 1:getnquadpoints(cv)
            ε = function_symmetric_gradient(cv, q_point, u, celldofs(cell))
            cell_stresses[q_point] = C ⊡ ε
        end
        σ_avg = sum(cell_stresses) / getnquadpoints(cv)
        avg_cell_stresses[1][cellid(cell)] = σ_avg[1, 1]
        avg_cell_stresses[2][cellid(cell)] = σ_avg[2, 2]
        avg_cell_stresses[3][cellid(cell)] = σ_avg[1, 2]
    end
    return qp_stresses, avg_cell_stresses
end

# Solve
K = allocate_matrix(dh)
assemble_global!(K, dh, cellvalues, C)

f_ext = zeros(ndofs(dh))

jp_test_package.assemble_external_forces!(f_ext, dh, getfacetset(grid, "topp"), facetvalues, traction)
println("Total applied force = ", sum(f_ext))
apply!(K, f_ext, ch)

u = K \ f_ext

# Calculate stresses
qp_stresses, avg_cell_stresses = calculate_stresses(grid, dh, cellvalues, u, C)

# Calculate von Mises stresses
cell_vm = zeros(getncells(grid))

# Hydrostatic pressure per cell
cell_pressure = zeros(getncells(grid))
cell_vol_strain = zeros(getncells(grid))

for cell in CellIterator(dh)
    reinit!(cellvalues, cell)
    p_int = 0.0
    ev_int = 0.0
    area = 0.0

    for qp in 1:getnquadpoints(cellvalues)
        ε = function_symmetric_gradient(cellvalues, qp, u, celldofs(cell))
        σ = C ⊡ ε
        
        # Plane strain: σ_33 = ν*(σ_11 + σ_22)
        σ_33 = ν * (σ[1,1] + σ[2,2])
        
        # Hydrostatic pressure (3D)
        p = -(σ[1,1] + σ[2,2] + σ_33) / 3.0
        
        # Volumetric strain (plane strain: ε_33 = 0)
        εvol = ε[1,1] + ε[2,2]
        
        dΩ = getdetJdV(cellvalues, qp)
        p_int  += p * dΩ
        ev_int += εvol * dΩ
        area   += dΩ
    end

    cell_pressure[cellid(cell)]   = p_int  / area
    cell_vol_strain[cellid(cell)] = ev_int / area
end


for cell in CellIterator(dh)
    reinit!(cellvalues, cell)
    area = 0.0
    vm_int = 0.0

    for qp in 1:getnquadpoints(cellvalues)
        ε = function_symmetric_gradient(cellvalues, qp, u, celldofs(cell))
        ε3 = SymmetricTensor{2, 3}((ε[1, 1], ε[2, 2], 0.0, ε[1, 2], 0.0, 0.0))
        vm_state = jp_test_package.von_mises_from_plane_strain_3d(ε3, C, return_components=true)
        σ_vm = vm_state.sigma_vm
        
        dΩ = getdetJdV(cellvalues, qp)
        vm_int += σ_vm * dΩ
        area += dΩ
    end

    cell_vm[cellid(cell)] = vm_int / area
end


# Stress field projection
proj = L2Projector(Lagrange{RefTriangle, 1}(), grid)
stress_field = project(proj, qp_stresses, qr)

# Write VTK output
VTKGridFile("volumetric_locking", dh) do vtk
    write_solution(vtk, dh, u)
    
    # Write displaced configuration
    displaced_x = zeros(length(grid.nodes))
    displaced_y = zeros(length(grid.nodes))
    for i in 1:length(grid.nodes)
        displaced_x[i] = grid.nodes[i].x[1] + u[2*i-1]
        displaced_y[i] = grid.nodes[i].x[2] + u[2*i]
    end
    write_node_data(vtk, displaced_x, "displaced_x")
    write_node_data(vtk, displaced_y, "displaced_y")
    
    # Write stresses
    for (i, key) in enumerate(("11", "22", "12"))
        write_cell_data(vtk, avg_cell_stresses[i], "sigma_" * key)
    end
    write_cell_data(vtk, cell_vm, "von_mises")
    write_projection(vtk, proj, stress_field, "stress field")
    write_cell_data(vtk, cell_pressure,   "pressure")
    write_cell_data(vtk, cell_vol_strain, "volumetric_strain")
    Ferrite.write_cellset(vtk, grid)
end

#println("\n=== ν = $ν ===")
#println("Max displacement: $(maximum(abs.(u)))")





@time run_simulation()

@profview for i in 1:10 run_simulation(); end



@time calculate_stresses(grid, dh, cellvalues, u, C);

@btime calculate_stresses($grid, $dh, $cellvalues, $u, $C);

@code_warntype calculate_stresses(grid, dh, cellvalues, u, C);

@benchmark calculate_stresses($grid, $dh, $cellvalues, $u, $C)