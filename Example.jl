using Tensors, Ferrite, SparseArrays
E = 200e9
ν = 0.3
dim = 2

λ = E*ν / ((1 + ν) * (1 - 2ν))
μ = E / (2(1 + ν))
δ(i,j) = i == j ? 1.0 : 0.0
f = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))

C = SymmetricTensor{4, dim}(f)


using Ferrite, Test


grid = generate_grid(Triangle, (20, 20))
dh = DofHandler(grid)

interpolation_u = Lagrange{RefTriangle, 1}()
interpolation_v = Lagrange{RefTriangle, 1}() ^ 2

add!(dh, :u, interpolation_u)
add!(dh, :v, interpolation_v)
close!(dh)



# Quadratic scalar interpolation
ip = Lagrange{RefTriangle, 2}()

# DofHandler
const N = 100
grid = generate_grid(Triangle, (N, N))
const dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

# Global matrix and a corresponding assembler
const K = allocate_matrix(dh)
##

#Heat Equation
using Ferrite, SparseArrays

grid = generate_grid(Quadrilateral, (20, 20)); #Creating a 20x20 grid of quadrilaterals

ip = Lagrange{RefQuadrilateral, 1}() #I want continuous lagrange polynomials of degree 1 to describe the cell values of the walls
qr = QuadratureRule{RefQuadrilateral}(2) #I want to use a quadrature rule with 2 quadrature points in each direction to compute the integrals over the cells.
cellvalues = CellValues(qr, ip); # Creating a new object to hold the values of the shape functions and their gradients at the quadrature points for each cell.

dh = DofHandler(grid) #I open the dofhandler on the grid to manage all the dofs
add!(dh, :u, ip) #i tell it to add a field called :u with the interpolation ip
close!(dh);

K = allocate_matrix(dh) # Allocate matrix to stiffness matrix K

ch = ConstraintHandler(dh); #Starte a constraint handler to manage the boundary conditions

∂Ω = union(
    getfacetset(grid, "left"),
    getfacetset(grid, "right"),
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
);

dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0) #The value of the solution u is 0 on the boundary ∂Ω for all time t
add!(ch, dbc);

close!(ch)

function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δu = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Add contribution to fe
            fe[i] += δu * dΩ
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

function assemble_global(cellvalues::CellValues, K::SparseMatrixCSC, dh::DofHandler)
    # Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues) # I would like an explanation of what this function does and what it returns
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    # Allocate global force vector f
    f = zeros(ndofs(dh))
    # Create an assembler
    assembler = start_assemble(K, f) #I think i understand this function, it creates an assembler object that will be used to assemble the global stiffness matrix K and the global force vector f.
    # The assembler takes care of the mapping between local element contributions and the global system, so we don't have to worry about the details of how the local matrices and vectors are added to the global ones. SEE JESPERS QUESTIONS
    # Loop over all cels
    for cell in CellIterator(dh) # Don't udnerstand this - is it just a built-in function of Ferrite to loop over the cells? But why is it referring to dh then, that was the constraints?
        # Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        # Compute element contribution
        assemble_element!(Ke, fe, cellvalues)
        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

K, f = assemble_global(cellvalues, K, dh);

apply!(K, f, ch)
u = K \ f;

VTKGridFile("heat_equation", dh) do vtk
    write_solution(vtk, dh, u)
end