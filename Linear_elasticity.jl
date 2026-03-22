using Ferrite, FerriteGmsh, SparseArrays

using Downloads: download
logo_mesh = "logo.geo"
asset_url = "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/"
isfile(logo_mesh) || download(string(asset_url, logo_mesh), logo_mesh)

grid = togrid(logo_mesh);

addfacetset!(grid, "top", x -> x[2] ≈ 1.0) # facets for which x[2] ≈ 1.0 for all nodes
addfacetset!(grid, "left", x -> abs(x[1]) < 1.0e-6)
addfacetset!(grid, "bottom", x -> abs(x[2]) < 1.0e-6);



dim = 2
order = 2 # quadratic approximation for the solution - i think???
ip = Lagrange{RefTriangle, order}()^dim; 


qr = QuadratureRule{RefTriangle}(2) # I think this also has to change?
qr_face = FacetQuadratureRule{RefTriangle}(1); 

cellvalues = CellValues(qr, ip)
facetvalues = FacetValues(qr_face, ip);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, 1))
close!(ch);
traction_factors = [0.2, 0.4, 0.6, 0.8, 1.0]  # Ramp up from 20% to 100%



for t in 1:5
traction(x) = Vec(0.0, 50.0e3 * x[1] * traction_factors[t]) # Traction function that varies with x[1] and time step;
function assemble_external_forces!(f_ext, dh, facetset, facetvalues, prescribed_traction)
    # Create a temporary array for the facet's local contributions to the external force vector
    fe_ext = zeros(getnbasefunctions(facetvalues))
    for facet in FacetIterator(dh, facetset)
        # Update the facetvalues to the correct facet number
        reinit!(facetvalues, facet)
        # Reset the temporary array for the next facet
        fill!(fe_ext, 0.0)
        # Access the cell's coordinates
        cell_coordinates = getcoordinates(facet)
        for qp in 1:getnquadpoints(facetvalues)
            # Calculate the global coordinate of the quadrature point.
            x = spatial_coordinate(facetvalues, qp, cell_coordinates) # This function computes the global coordinate of the quadrature point based on the facet's geometry and the shape functions.
            tₚ = prescribed_traction(x) # Evaluate the prescribed traction at the quadrature point's global coordinate.
            # Get the integration weight for the current quadrature point.
            dΓ = getdetJdV(facetvalues, qp)
            for i in 1:getnbasefunctions(facetvalues)
                Nᵢ = shape_value(facetvalues, qp, i)
                fe_ext[i] += tₚ ⋅ Nᵢ * dΓ
            end
        end
        # Add the local contributions to the correct indices in the global external force vector
        assemble!(f_ext, celldofs(facet), fe_ext)
    end
    return f_ext
end

Emod = 200.0e3 # Young's modulus [MPa]
ν = 0.3        # Poisson's ratio [-]

Gmod = Emod / (2(1 + ν))  # Shear modulus
Kmod = Emod / (3(1 - 2ν)) # Bulk modulus
#
C = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2, 2})); #So this means that i take the gradient of the funcction with respect to \epsilon and insert it into a zero symmetric tensor to get the elasticity tensor C, is that correct?
#
function assemble_cell!(ke, cellvalues, C)
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the test function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the trial function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end



function assemble_global!(K, dh, cellvalues, C)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cellvalues, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        # Compute element contribution
        #
        assemble_cell!(ke, cellvalues, C) 
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

K = allocate_matrix(dh)
assemble_global!(K, dh, cellvalues, C); #Now we call the function we created earlier which then calls the assemble_cell which we also call.


f_ext = zeros(ndofs(dh))
assemble_external_forces!(f_ext, dh, getfacetset(grid, "top"), facetvalues, traction);

apply!(K, f_ext, ch)
#So because we have the exclamation mark, the above function mutates K and F_ext to account for the BC ?


u = K \ f_ext;

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

qp_stresses, avg_cell_stresses = calculate_stresses(grid, dh, cellvalues, u, C);

cell_vm = zeros(getncells(grid))

for cell in CellIterator(dh)
    reinit!(cellvalues, cell)
    area = 0.0
    vm_int = 0.0

    for qp in 1:getnquadpoints(cellvalues)
        ε = function_symmetric_gradient(cellvalues, qp, u, celldofs(cell))
        σ = C ⊡ ε
        σ_33    = ν * (σ[1, 1] + σ[2, 2]) # Plane strain assumption 
        σ3 = SymmetricTensor{2,3}((σ[1,1], σ[2,2], σ_33, σ[1,2], 0.0, 0.0))



        σ_vm = sqrt(1.5 * (dev(σ3) ⊡ dev(σ3)))

        dΩ = getdetJdV(cellvalues, qp)
        vm_int += σ_vm * dΩ
        area += dΩ
    end

    cell_vm[cellid(cell)] = vm_int / area
end



proj = L2Projector(Lagrange{RefTriangle, 1}(), grid)
stress_field = project(proj, qp_stresses, qr);


VTKGridFile("linear_elasticity", dh) do vtk
    write_solution(vtk, dh, u)
    for (i, key) in enumerate(("11", "22", "12"))
        write_cell_data(vtk, avg_cell_stresses[i], "sigma_" * key)
        
    end
    write_cell_data(vtk, cell_vm, "von_mises")
    write_projection(vtk, proj, stress_field, "stress field")
    Ferrite.write_cellset(vtk, grid)
end 


# Get the facet set for the top boundary
top_facet_set = getfacetset(grid, "top")
# Initialize a variable to store the reaction forces
reaction_forces = zeros(length(u))
# Compute internal forces
f_internal = K * u




ch_dummy = ConstraintHandler(dh)
add!(ch_dummy, Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> 0.0, 2))

close!(ch_dummy);

ch_dummy.prescribed_dofs
# Loop through the facets in  top boundary and sum the internal forces for  corresponding DOFs
for i in 1:length(ch_dummy.prescribed_dofs)
    dofs = ch_dummy.prescribed_dofs[i]
    reaction_forces[dofs] += f_internal[dofs]
end


total_reaction_force = sum(reaction_forces)
println("Total reaction force at the top boundary: ", total_reaction_force)
end
