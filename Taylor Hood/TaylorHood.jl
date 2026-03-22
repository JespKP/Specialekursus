using Ferrite
using Tensors
using JespersPackage
# ∫ (2G dev(𝛆) - p𝐈) : ∇v dΩ


# insert discretisations
# 𝛆 = ∑ ∇ˢʸᵐ𝐍ᵘᵢ uᵢ
# ∇v = ∑ ∇ˢʸᵐ𝐍ᵘᵢ vᵢ

# p = ∑ Nᵖᵢ pᵢ
# q = ∑ Nᵖᵢ qᵢ

# δWⁱⁿᵗᵤ = ∫ (2G dev(∇ˢʸᵐ𝐍ᵘᵢ uᵢ) - (Nᵖᵢ pᵢ)𝐈) : ∇ˢʸᵐ𝐍ᵘⱼ vⱼ dΩ
# ∫ (2G dev(∇ˢʸᵐ𝐍ᵘᵢ uᵢ) - (Nᵖᵢ pᵢ)𝐈) : ∇ˢʸᵐ𝐍ᵘⱼ vⱼ dΩ
# = [∫ (2G dev(∇ˢʸᵐ𝐍ᵘᵢ uᵢ) - (Nᵖᵢ pᵢ)𝐈) : ∇ˢʸᵐ𝐍ᵘᵢ dΩ] vⱼ 

# Fᵘⱼ = ∂(δW)∂(vⱼ) = ∫ (2G dev(∇ˢʸᵐ𝐍ᵘᵢ uᵢ) - (Nᵖᵢ pᵢ)𝐈) : ∇ˢʸᵐ𝐍ᵘⱼ dΩ
# Kᵘᵘⱼᵢ = ∫ 2G dev(∇ˢʸᵐ𝐍ᵘᵢ) : ∇ˢʸᵐ𝐍ᵘⱼ dΩ
# Kᵘᵖⱼᵢ = ∫ - Nᵖᵢ tr(∇ˢʸᵐ𝐍ᵘⱼ) dΩ # [12 x 3]


# ∫ (tr(𝛆) + 1/K p) * q dΩ
# ∫ (tr(∇ˢʸᵐ𝐍ᵘᵢ uᵢ) + 1/K (Nᵖᵢ pᵢ)) * (Nᵖⱼ qⱼ) dΩ
# δWⁱⁿᵗₚ = ∫ [(tr(∇ˢʸᵐ𝐍ᵘᵢ uᵢ) + 1/K (Nᵖᵢ pᵢ)) * Nᵖⱼ ] qⱼ dΩ

# Fᵖᵢ = ∂(δW)∂(qᵢ) = ∫ [(tr(∇ˢʸᵐ𝐍ᵘᵢ uᵢ) + 1/K (Nᵖᵢ pᵢ)) * Nᵖⱼ ] dΩ 
#Kᵖᵘⱼᵢ = ∫ Nᵖⱼ tr(∇ˢʸᵐNᵘᵢ) dΩ   
#Kᵖᵖⱼᵢ = ∫ 1/K Nᵖⱼ Nᵖᵢ dΩ        
#  Kᵖᵘ · u + Kᵖᵖ · p = 0

function create_cook_grid(nx, ny)
    corners = [
        Vec{2}((0.0, 0.0)),
        Vec{2}((48.0, 44.0)),
        Vec{2}((48.0, 60.0)),
        Vec{2}((0.0, 44.0)),
    ]
    grid = generate_grid(Triangle, (nx, ny), corners)
    # facesets for boundary conditions
    addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
    addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
    return grid
end;


function create_values(interpolation_u, interpolation_p)
    # quadrature rules
    qr = QuadratureRule{RefTriangle}(3)
    facet_qr = FacetQuadratureRule{RefTriangle}(3)

    # cell and FacetValues for u
    cv_u = CellValues(qr, interpolation_u)
    facetvalues_u = FacetValues(facet_qr, interpolation_u) 

    # cellvalues for p
    cv_p = CellValues(qr, interpolation_p)

    return cv_u, cv_p, facetvalues_u
end;


function assemble_cell!(ke, cv_u, cv_p, Gmod, Kmod, dofs_u, dofs_p)
# Kᵘᵖⱼᵢ = ∫ - Nᵖᵢ tr(∇ˢʸᵐ𝐍ᵘⱼ) dΩ # [12 x 3]
for qp in 1:getnquadpoints(cv_u)
    dΩ = getdetJdV(cv_u, qp)

    for i in 1:getnbasefunctions(cv_u)
        ∇symNi = shape_symmetric_gradient(cv_u, qp, i)

        for j in 1:getnbasefunctions(cv_p)
            Nj = shape_value(cv_p, qp, j)

            ke[dofs_u[i], dofs_p[j]] += -Nj * tr(∇symNi) * dΩ
        end
    end
end
# Kᵘᵘⱼᵢ = ∫ 2G dev(∇ˢʸᵐ𝐍ᵘᵢ) : ∇ˢʸᵐ𝐍ᵘⱼ dΩ
for qp in 1:getnquadpoints(cv_u) # doesn't matter which cellvalues
     dΩ = getdetJdV(cv_u, qp) # doesn't matter which cv
     for j in 1:getnbasefunctions(cv_u) # j in 1:12
        ∇ˢʸᵐ𝐍ᵘⱼ = shape_symmetric_gradient(cv_u, qp, j)
        for i in 1:getnbasefunctions(cv_u) # i in 1:12
            ∇ˢʸᵐ𝐍ᵘᵢ = shape_symmetric_gradient(cv_u, qp, i)
            dev_∇Nᵢ = dev(∇ˢʸᵐ𝐍ᵘᵢ)
            ke[dofs_u[j],dofs_u[i]] += 2Gmod * (dev_∇Nᵢ ⊡ ∇ˢʸᵐ𝐍ᵘⱼ) * dΩ
        end
    end
end

#Kᵖᵖⱼᵢ = ∫ 1/K Nᵖⱼ Nᵖᵢ dΩ        

#Kᵖᵘⱼᵢ = ∫ Nᵖⱼ tr(∇ˢʸᵐNᵘᵢ) dΩ  
for qp in 1:getnquadpoints(cv_u)
    dΩ = getdetJdV(cv_u, qp)

    for i in 1:getnbasefunctions(cv_p)
        Ni = shape_value(cv_p, qp, i)

        for j in 1:getnbasefunctions(cv_u)
            ∇symNj = shape_symmetric_gradient(cv_u, qp, j)

            ke[dofs_p[i], dofs_u[j]] += -Ni * tr(∇symNj) * dΩ
        end
    end
end
#Kᵖᵖⱼᵢ = ∫ 1/K Nᵖⱼ Nᵖᵢ dΩ     
for qp in 1:getnquadpoints(cv_u) # doesn't matter which cellvalues
     dΩ = getdetJdV(cv_u, qp) # doesn't matter which cv
     for j in 1:getnbasefunctions(cv_p) # j in 1:3
        Nᵖⱼ = shape_value(cv_p, qp, j)
        for i in 1:getnbasefunctions(cv_p) # i in 1:3
            Nᵖᵢ = shape_value(cv_p, qp, i)
            ke[dofs_p[j],dofs_p[i]] += -1/Kmod * Nᵖⱼ * Nᵖᵢ * dΩ
        end
    end
end

    return ke
end

function assemble_global!(K, dh, cv_u, cv_p, Gmod, Kmod)
    n_dofs = ndofs_per_cell(dh)
    ke = zeros(n_dofs, n_dofs)
    assembler = start_assemble(K)
    
    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        reinit!(cv_u, cell)
        reinit!(cv_p, cell)
         dofs_u = dof_range(dh, :u)
        dofs_p = dof_range(dh, :p)

        assemble_cell!(ke, cv_u, cv_p, Gmod, Kmod,dofs_u, dofs_p)
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

function assemble_external_forces!(f_ext, dh, facetset, facetvalues, prescribed_traction)
    n_dofs_cell = ndofs_per_cell(dh)
    fe_ext = zeros(n_dofs_cell)      # Local element force vector
    dofs_u = dof_range(dh, :u)       # Local indices for displacement (e.g., 1:12)
    
    for facet in FacetIterator(dh, facetset)
        fill!(fe_ext, 0.0)           # CRITICAL: Reset local force for each facet
        reinit!(facetvalues, facet)
        
        for qp in 1:getnquadpoints(facetvalues)
            dΓ = getdetJdV(facetvalues, qp)
            tₚ = prescribed_traction(spatial_coordinate(facetvalues, qp, getcoordinates(facet))) #
            
            # We only loop over displacement base functions (u)
            for i in 1:getnbasefunctions(facetvalues)
    Nᵢ = shape_value(facetvalues, qp, i) # This is a Vec{2}
    
    # Use ⋅ for dot product between the two vectors
    fe_ext[dofs_u[i]] += (Nᵢ ⋅ tₚ) * dΓ  
end
        end
        
        # Assemble the 15-element fe_ext into the global f_ext
        assemble!(f_ext, celldofs(facet), fe_ext)
    end
end


ip_u = Lagrange{RefTriangle, 2}()^2 
ip_p = Lagrange{RefTriangle, 1}()

cv_u, cv_p, facetvalues_u = create_values(ip_u, ip_p)
grid = create_cook_grid(50,50)


dh = DofHandler(grid)
add!(dh, :u, ip_u)

add!(dh, :p, ip_p)
close!(dh)


Emod = 1 # Young's modulus in MPa
ν =  0.5     # Poisson's ratio [-]

    Gmod = Emod / 2(1 + ν)
    Kmod = Emod * ν / (3 * (1 - 2ν))



dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1, 2]))
if ν == 0.5
    add!(dbc, Dirichlet(:p, [1], x -> 0.0))
end
close!(dbc)
update!(dbc, 0.0)    # 


traction = (x) -> Vec(0.0, 1/16) #
f_ext = zeros(ndofs(dh)) # 




assemble_external_forces!(f_ext, dh, getfacetset(grid, "traction"), facetvalues_u, traction)


# local dof ranges of the fields within the elements dofs
dofs_u = dof_range(dh, :u) # 1:12
dofs_p = dof_range(dh, :p) # 13:15

K = allocate_matrix(dh)
assemble_global!(K, dh, cv_u, cv_p, Gmod, Kmod);
println("Matrix symmetry error: ", norm(K - K'))
apply!(K, f_ext, dbc)


u = K \ f_ext


# Find the node at the top right corner (48.0, 60.0)
top_right_node = 0
for (i, node) in enumerate(grid.nodes)
    if norm(node.x - Vec(48.0, 60.0)) < 1e-3
        top_right_node = i
        break
    end
end

# Extract the Y-displacement
u_nodal = evaluate_at_grid_nodes(dh, u, :u)
tip_displacement_y = u_nodal[top_right_node][2]

println("Current Tip Displacement (Y): ", tip_displacement_y)
println("Target Benchmark Value: ~23.9")